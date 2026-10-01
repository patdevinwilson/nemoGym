# GEOBench-VLM NemoGym

NemoGym implementation of
[GEOBench-VLM](https://github.com/The-AI-Alliance/GEO-Bench-VLM), the
vision-language benchmark from "GEOBench-VLM: Benchmarking Vision-Language
Models for Geospatial Tasks"
([arXiv:2411.19325](https://arxiv.org/abs/2411.19325)).

This is a single-turn image benchmark. It is separate from the GeoBenchX
tool-calling Gym.

## Scope

- Official multiple-choice prompt: answer with one letter, A through E.
- Single-image and before/after temporal items.
- Reward `1` when the parsed letter matches `ground_truth_option`, otherwise `0`.
- `geobench_vlm_first_letter_correct` records the released temporal script's
  rule, which keeps only the first character.
- Headline paper metric: MCQ accuracy. LLaVA-OneVision is reported at 41.7%.
  Referring-expression IoU and segmentation masks are not scored here.

The committed dataset is `data/example.jsonl` (six rows from five synthetic
questions, because one question has two prompts). The 3,211-question Hugging
Face split and its images are not vendored. See `DATA_ATTESTATION.md`.

Training is not started. `training_sources.py` registers BigEarthNet v2,
EuroSAT, USDA Cropland Data Layer 2020–2023, NAIP, Microsoft building
footprints, OpenStreetMap, and SpaceNet 2 and 3. Each source stays
`registered` and `initiate` refuses, so nothing is downloaded.

## Build a local dataset

Place the official tree so `Single/qa.json` and, if you have it,
`Temporal/qa.json` exist, then:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from resources_servers.geobench_vlm.task_data import load_qa_tree, materialize
rows = load_qa_tree(Path("/path/to/GEOBench-VLM"))
materialize(Path("resources_servers/geobench_vlm/data/benchmark.jsonl"), rows)
print(len(rows))
PY
```

Point the dataset entry in `configs/geobench_vlm.yaml` at that file. Image
files are read and sent as data URLs when the row is built.

## Run the example

```bash
.venv/bin/gym env start \
  "+config_paths=[responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/geobench_vlm/configs/geobench_vlm.yaml]"

.venv/bin/gym eval run --no-serve \
  --agent geobench_vlm_simple_agent \
  --input resources_servers/geobench_vlm/data/example.jsonl \
  --output resources_servers/geobench_vlm/data/example_rollouts.jsonl \
  --limit 6 --num-repeats 1 --concurrency 1
```

`env.yaml` needs `policy_base_url`, `policy_api_key`, and `policy_model_name`
for a vision-capable model. Do not commit that file.
