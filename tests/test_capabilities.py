from grounding_mle.capabilities import capability_metadata


def test_capability_metadata_computes_prompt_length_without_dataset_import() -> None:
    metadata = capability_metadata("Return the sum of two integers.")

    assert metadata["algorithm_type"] == "number_theory"
    assert metadata["api_family"] == "builtins"
    assert metadata["prompt_complexity"] == "short"
    assert metadata["prompt_tokens_proxy"] == 6
