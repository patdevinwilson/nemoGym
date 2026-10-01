from __future__ import annotations

import json

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

from resources_servers.geobench_vlm.scoring import score_answer
from resources_servers.geobench_vlm.task_data import (
    dataset_row,
    example_rows,
    load_qa_tree,
    materialize,
)


def test_letter_match_accepts_the_official_short_answer() -> None:
    scored = score_answer("B", "B", "Harbor")
    assert scored["reward"] == 1.0
    assert scored["geobench_vlm_first_letter_correct"] is True


def test_sentence_answer_counts_when_the_letter_is_explicit() -> None:
    scored = score_answer("The choice is C.", "C.", "3")
    assert scored["predicted_letter"] == "C"
    assert scored["reward"] == 1.0


def test_wrong_letter_scores_zero() -> None:
    scored = score_answer("A. Forest", "B", "Harbor")
    assert scored["reward"] == 0.0
    assert scored["gold_letter"] == "B"


def test_first_character_rule_rejects_a_leading_word() -> None:
    scored = score_answer("Answer: D", "D", "Drought")
    assert scored["geobench_vlm_correct"] is True
    assert scored["geobench_vlm_first_letter_correct"] is False


def test_example_rows_cover_single_and_temporal_prompts() -> None:
    rows = example_rows()
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"single", "temporal"}
    temporal = next(row for row in rows if row["split"] == "temporal")
    types = [block["type"] for block in temporal["responses_create_params"]["input"][0]["content"]]
    assert types.count("input_image") == 2
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        rows[0]["responses_create_params"]
    )
    assert params.temperature == 0
    text = params.input[0].content[-1]["text"]
    assert "A., B., C., D. or E." in text


def test_qa_tree_expands_every_prompt(tmp_path) -> None:
    single = tmp_path / "Single"
    single.mkdir()
    (single / "qa.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 9,
                    "task": "Scene Classification",
                    "image_path": "images/a.jpg",
                    "prompts": ["one", "two"],
                    "options": "A. Forest B. Harbor",
                    "ground_truth": "Harbor",
                    "ground_truth_option": "B",
                    "cls_description": "overhead",
                }
            ]
        )
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"not-a-real-jpeg")
    rows = load_qa_tree(tmp_path)
    assert len(rows) == 2
    assert rows[0]["prompt_index"] == 0
    assert rows[1]["prompt_index"] == 1
    assert "one" in rows[0]["responses_create_params"]["input"][0]["content"][-1]["text"]


def test_materialize_writes_the_example_set(tmp_path) -> None:
    path = materialize(tmp_path / "example.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["1", "2", "3", "4", "5", "5"]
    assert dataset_row({"prompts": ["q"], "ground_truth_option": "A", "task": "t"})["task"] == "t"
