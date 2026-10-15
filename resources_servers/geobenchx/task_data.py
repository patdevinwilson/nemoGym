"""Load and materialize the official GeoBenchX task set for NemoGym."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from resources_servers.geobenchx.catalog import response_tools


ROOT = Path(__file__).resolve().parent
UPSTREAM_TASKS = ROOT / "upstream" / "tasks_and_reference_solutions.json"
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "benchmark.jsonl"
VALIDATION_FRACTION = 0.26
# Upstream prompts use the run date. A fixed date keeps prompts identical across
# baseline, training, and post-training runs.
BENCHMARK_DATE = date(2026, 10, 1)

SYSTEM_PROMPT = """You are a geographer who answers questions with maps whenever possible.
Use the supplied GIS tools and snapshot catalogs. Do not provide Python code. Provide the
dataset names used, the resulting map or calculation, and concise explanations.

Rules:
- If the task is not geospatial and needs no supplied data, answer directly without tools.
- If a geospatial task cannot be solved with the available datasets and tools, call reject_task.
- Reject when a dataset covers only part of the requested geography.
- If no date is specified, use the latest period with non-empty data for over 70% of objects.
- Use the same year for both variables in a bivariate map.
- Do not ask for clarification. Infer reasonable values and complete the task.
- You may make at most 25 tool calls.
"""


def load_tasks() -> dict[str, dict[str, Any]]:
    records = json.loads(UPSTREAM_TASKS.read_text())["tasks"]
    return {record["task_ID"]: record for record in records}


TASKS = load_tasks()


def task_kind(task: dict[str, Any]) -> str:
    kinds = set()
    for solution in task["reference_solutions"]:
        names = [step["function_name"] for step in solution.get("steps") or []]
        kinds.add("control" if not names else "reject" if names == ["reject_task"] else "solve")
    return "+".join(sorted(kinds))


def primary_group(task: dict[str, Any]) -> str:
    return next(
        (
            label
            for label in task["task_labels"]
            if label
            in {
                "Merge, Visualize",
                "Process, Merge, Visualize",
                "Spatial operations",
                "Heatmaps, Contour Lines",
                "Control question",
            }
        ),
        task["task_labels"][0],
    )


def dataset_row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_ID"],
        "activity": "geobenchx",
        "group": primary_group(task),
        "task_kind": task_kind(task),
        "task_labels": task["task_labels"],
        "reference_solution_count": len(task["reference_solutions"]),
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\nBenchmark date: {BENCHMARK_DATE.isoformat()}.",
                },
                {"role": "user", "content": task["task_text"]},
            ],
            "tools": response_tools(),
            "parallel_tool_calls": False,
            "temperature": 0,
        },
    }


def split_of(tasks: dict[str, dict[str, Any]] = TASKS) -> dict[str, str]:
    """Fixed train/validation assignment, stratified by group and task kind.

    Within each stratum, tasks are ordered by a hash of the task ID, so the split
    does not depend on file order and is stable across rebuilds.
    """
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_id, task in tasks.items():
        strata[(primary_group(task), task_kind(task))].append(task_id)
    split = {}
    for ids in strata.values():
        ids.sort(key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest())
        n_validation = round(len(ids) * VALIDATION_FRACTION)
        for index, task_id in enumerate(ids):
            split[task_id] = "validation" if index < n_validation else "train"
    return split


def _write(path: Path, tasks: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for task in tasks:
            stream.write(json.dumps(dataset_row(task), ensure_ascii=False, separators=(",", ":")) + "\n")


def materialize(data_dir: Path = DATA_DIR) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    split = split_of()
    train = [task for task_id, task in TASKS.items() if split[task_id] == "train"]
    validation = [task for task_id, task in TASKS.items() if split[task_id] == "validation"]
    outputs = {
        "benchmark.jsonl": list(TASKS.values()),
        "train.jsonl": train,
        "validation.jsonl": validation,
        "example.jsonl": train[:5],
    }
    for name, tasks in outputs.items():
        _write(data_dir / name, tasks)
    return [data_dir / name for name in outputs]


if __name__ == "__main__":
    for path in materialize():
        print(path)
