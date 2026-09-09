from grounding_mle.datasets import contamination_matches, normalize_prompt
from grounding_mle.records import PromptRecord


def record(task_id: str, prompt: str, response: str = "pass") -> PromptRecord:
    return PromptRecord(task_id=task_id, prompt=prompt, human_responses=[response])


def test_prompt_normalization_and_contamination_filter():
    assert normalize_prompt("Hello,  WORLD!") == "hello world"
    candidates = [
        record("train/1", "Write a function to add two integers."),
        record(
            "train/2",
            "Compute a graph's shortest path with Dijkstra's algorithm.",
            "def shortest_path(graph):\n    return []",
        ),
    ]
    evaluation = [record("eval/1", "Write a function to add two integers!")]
    clean, matches = contamination_matches(candidates, evaluation)
    assert [row.task_id for row in clean] == ["train/2"]
    assert matches[0]["kind"] in {"exact", "exact_python_ast"}
