import sys
from types import SimpleNamespace

from grounding_mle.modeling import _CausalSFTDataset, _empty_device_cache, formatted_prompt
from grounding_mle.records import TrainingExample


class TinyTokenizer:
    eos_token = "!"
    eos_token_id = 1
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, truncation, max_length, add_special_tokens):
        prefix = [2] if add_special_tokens else []
        ids = (prefix + [3 + ord(char) % 31 for char in text])[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_sft_masks_only_prompt_and_reserves_response_tokens():
    tokenizer = TinyTokenizer()
    example = TrainingExample(
        example_id="one",
        prompt="p" * 100,
        response="answer",
        source="real",
        round_index=0,
        task_id="task",
    )
    dataset = _CausalSFTDataset(
        [example], tokenizer, max_length=16, domain="code", max_prompt_length=10
    )
    item = dataset[0]
    assert len(item["input_ids"]) <= 16
    assert item["labels"][:10] == [-100] * 10
    assert any(token != -100 for token in item["labels"])


def test_chat_template_is_used_when_available():
    tokenizer = TinyTokenizer()
    tokenizer.chat_template = "configured"
    tokenizer.apply_chat_template = lambda messages, **kwargs: "CHAT:" + messages[0]["content"]
    assert formatted_prompt(tokenizer, "hello", "code").startswith("CHAT:")


def test_empty_device_cache_does_not_call_unavailable_mps(monkeypatch):
    class UnavailableMPS:
        @staticmethod
        def is_available():
            return False

    class MPSNamespace:
        @staticmethod
        def empty_cache():
            raise AssertionError("MPS cache must not be used when its backend is unavailable")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=UnavailableMPS()),
        mps=MPSNamespace(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _empty_device_cache()
