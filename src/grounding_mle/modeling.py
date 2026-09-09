from __future__ import annotations

import gc
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from .io import atomic_write_json, seed_everything
from .records import PromptRecord, TrainingExample


def _empty_device_cache() -> None:
    """Release accelerator caches only when the corresponding backend is usable."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and getattr(torch, "mps", None) is not None
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def instruction_text(prompt: str, domain: str = "code") -> str:
    if domain == "raw":
        return prompt
    if domain == "math":
        return (
            "Solve the problem carefully. End with a line of the form #### <answer>.\n\n"
            f"Problem:\n{prompt}\n\nSolution:\n"
        )
    return (
        "Solve the programming problem. Return only a complete Python solution.\n\n"
        f"Problem:\n{prompt}\n\nSolution:\n"
    )


def formatted_prompt(tokenizer: Any, prompt: str, domain: str) -> str:
    instruction = instruction_text(prompt, domain)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def strip_code_fences(text: str) -> str:
    value = text.strip()
    if "```" not in value:
        return value
    blocks = value.split("```")
    candidates = [blocks[index] for index in range(1, len(blocks), 2)]
    if not candidates:
        return value
    code = max(candidates, key=len).strip()
    if code.startswith("python"):
        code = code[len("python") :].lstrip("\n")
    return code.strip()


class ModelBackend(ABC):
    @abstractmethod
    def train(
        self,
        source_model: str,
        examples: Sequence[TrainingExample],
        output_dir: Path,
        config: dict[str, Any],
        seed: int,
        base_model: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        model_ref: str,
        prompts: Sequence[PromptRecord],
        config: dict[str, Any],
        seed: int,
        base_model: str,
        domain: str,
    ) -> list[str]:
        raise NotImplementedError


class AnalyticBackend(ModelBackend):
    """Fast deterministic backend for end-to-end protocol smoke tests only."""

    def train(
        self,
        source_model: str,
        examples: Sequence[TrainingExample],
        output_dir: Path,
        config: dict[str, Any],
        seed: int,
        base_model: str,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        real = sum(example.source == "real" for example in examples)
        synthetic = sum(example.source == "synthetic" for example in examples)
        inherited = 0.25
        source_state = Path(source_model) / "analytic_state.json"
        if source_state.exists():
            with source_state.open(encoding="utf-8") as handle:
                inherited = float(json.load(handle)["error"])
        total = max(1, real + synthetic)
        gamma = real / total
        noise = ((seed % 97) / 97 - 0.5) / math.sqrt(total)
        error = max(0.0, min(1.0, (1 - gamma) * inherited + abs(noise)))
        atomic_write_json(
            output_dir / "analytic_state.json",
            {"error": error, "real": real, "synthetic": synthetic, "seed": seed},
        )
        return str(output_dir)

    def generate(
        self,
        model_ref: str,
        prompts: Sequence[PromptRecord],
        config: dict[str, Any],
        seed: int,
        base_model: str,
        domain: str,
    ) -> list[str]:
        if domain == "math":
            return ["Reasoning omitted in analytic smoke mode.\n#### 0" for _ in prompts]
        return ["def solution(*args, **kwargs):\n    return None" for _ in prompts]


class _CausalSFTDataset:
    def __init__(
        self,
        examples: Sequence[TrainingExample],
        tokenizer: Any,
        max_length: int,
        domain: str,
        max_prompt_length: int,
    ):
        self.items: list[dict[str, list[int]]] = []
        eos = tokenizer.eos_token or ""
        for example in examples:
            prefix = formatted_prompt(tokenizer, example.prompt, domain)
            prefix_ids = tokenizer(
                prefix,
                truncation=True,
                max_length=min(max_prompt_length, max_length - 1),
                add_special_tokens=True,
            )["input_ids"]
            response_ids = tokenizer(
                example.response.rstrip() + eos,
                truncation=True,
                max_length=max_length - len(prefix_ids),
                add_special_tokens=False,
            )["input_ids"]
            if not response_ids and tokenizer.eos_token_id is not None:
                response_ids = [int(tokenizer.eos_token_id)]
            if not response_ids:
                raise ValueError("A training response produced no tokens")
            input_ids = list(prefix_ids) + list(response_ids)
            labels = [-100] * len(prefix_ids) + list(response_ids)
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class _CausalCollator:
    def __init__(self, tokenizer: Any, fixed_length: int | None = None):
        self.tokenizer = tokenizer
        self.fixed_length = fixed_length

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = self.fixed_length or max(len(feature["input_ids"]) for feature in features)
        if any(len(feature["input_ids"]) > max_length for feature in features):
            raise ValueError("A tokenized example exceeds the collator length")
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HuggingFaceBackend(ModelBackend):
    def _device(self) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, model_ref: str, base_model: str, trainable: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ref_path = Path(model_ref)
        adapter = ref_path.is_dir() and (ref_path / "adapter_config.json").exists()
        tokenizer_ref = str(ref_path) if ref_path.is_dir() and (ref_path / "tokenizer_config.json").exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if self._device() == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load this LoRA checkpoint") from exc
            model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
            model = PeftModel.from_pretrained(model, model_ref, is_trainable=trainable)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=dtype)
        model.config.use_cache = not trainable
        return model, tokenizer

    def train(
        self,
        source_model: str,
        examples: Sequence[TrainingExample],
        output_dir: Path,
        config: dict[str, Any],
        seed: int,
        base_model: str,
    ) -> str:
        if not examples:
            raise ValueError("Cannot fine-tune on an empty training set")
        seed_everything(seed)
        import torch
        from transformers import Trainer, TrainingArguments

        use_lora = bool(config.get("lora", {}).get("enabled", True))
        model, tokenizer = self._load_model(source_model, base_model, trainable=True)
        if use_lora and not (Path(source_model) / "adapter_config.json").exists():
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError("Install the llm extra to use LoRA training") from exc
            lora = config["lora"]
            model = get_peft_model(
                model,
                LoraConfig(
                    r=int(lora.get("rank", 16)),
                    lora_alpha=int(lora.get("alpha", 32)),
                    lora_dropout=float(lora.get("dropout", 0.05)),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora.get(
                        "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                    ),
                ),
            )
        if bool(config.get("gradient_checkpointing", True)):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.config.use_cache = False
        device = self._device()
        if device != "cuda":
            model.to(device)
        domain = str(config.get("domain", "code"))
        dataset = _CausalSFTDataset(
            examples,
            tokenizer,
            int(config.get("max_length", 1024)),
            domain,
            int(config.get("max_prompt_length", 768)),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            learning_rate=float(config.get("learning_rate", 2e-4)),
            max_steps=int(config.get("max_steps", 100)),
            per_device_train_batch_size=int(config.get("batch_size", 1)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
            warmup_ratio=float(config.get("warmup_ratio", 0.03)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
            logging_steps=int(config.get("logging_steps", 10)),
            save_strategy="no",
            report_to=[],
            seed=seed,
            data_seed=seed,
            fp16=False,
            bf16=device == "cuda" and torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=_CausalCollator(
                tokenizer,
                int(config.get("max_length", 1024))
                if bool(config.get("pad_to_max_length", True))
                else None,
            ),
        )
        result = trainer.train()
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        atomic_write_json(
            output_dir / "training_summary.json",
            {
                "source_model": source_model,
                "base_model": base_model,
                "seed": seed,
                "examples": len(examples),
                "prescribed_training_tokens": (
                    int(config.get("max_steps", 100))
                    * int(config.get("batch_size", 1))
                    * int(config.get("gradient_accumulation_steps", 8))
                    * int(config.get("max_length", 1024))
                ),
                "metrics": result.metrics,
                "device": device,
            },
        )
        del trainer, model
        gc.collect()
        _empty_device_cache()
        return str(output_dir)

    def generate(
        self,
        model_ref: str,
        prompts: Sequence[PromptRecord],
        config: dict[str, Any],
        seed: int,
        base_model: str,
        domain: str,
    ) -> list[str]:
        if not prompts:
            return []
        seed_everything(seed)
        import torch
        from transformers import set_seed

        set_seed(seed)
        model, tokenizer = self._load_model(model_ref, base_model, trainable=False)
        device = self._device()
        model.to(device)
        model.eval()
        batch_size = int(config.get("batch_size", 4))
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            texts = [formatted_prompt(tokenizer, row.prompt, domain) for row in batch]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(config.get("max_prompt_length", 1024)),
            ).to(device)
            do_sample = bool(config.get("do_sample", True))
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(config.get("max_new_tokens", 512)),
                "do_sample": do_sample,
                "num_return_sequences": 1,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update(
                    temperature=float(config.get("temperature", 0.8)),
                    top_p=float(config.get("top_p", 0.95)),
                )
            else:
                # Instruction models can ship sampling defaults. Clear them
                # explicitly for deterministic benchmark evaluation.
                generation_kwargs.update(temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            encoded_length = int(encoded["input_ids"].shape[1])
            for item in generated:
                decoded = tokenizer.decode(item[encoded_length:], skip_special_tokens=True)
                outputs.append(strip_code_fences(decoded) if domain == "code" else decoded.strip())
        del model
        gc.collect()
        _empty_device_cache()
        return outputs


def make_backend(name: str) -> ModelBackend:
    if name == "analytic":
        return AnalyticBackend()
    if name == "hf":
        return HuggingFaceBackend()
    raise ValueError(f"Unknown model backend: {name}")
