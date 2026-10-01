# GeoBenchX NemoGym

NemoGym implementation of
[GeoBenchX](https://github.com/Solirinai/GeoBenchX), the benchmark from
“GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks”
([arXiv:2503.18129](https://arxiv.org/abs/2503.18129)).

## Scope

- All 202 official tasks and all annotated alternative reference solutions.
- Four paper groups: Merge/Visualize; Process/Merge/Visualize; Spatial Operations;
  Heatmaps/Contour Lines.
- The released interface's 23 GIS tools plus the additional `reject_task` tool
  (24 callable tools total).
- Temperature 0, at most 25 tool calls, and the paper's solvability rules.
- Match score `0` (no match), `1` (partial), or `2` (full), exposed as reward
  `0`, `0.5`, or `1`.
- Solution-length metrics for regression and efficiency analysis.

The paper used a panel of three LLM judges. This Gym defaults to a deterministic
reference matcher so repeated release-regression runs are reproducible. It ignores
temporary dataframe handle names, tolerates extra steps after a complete solution,
and treats colormap, NoData, and spatial-predicate differences as partial matches.
The official references remain vendored, so a panel-judge verifier can be added
without changing the dataset.

## Build the dataset

```bash
.venv/bin/python -m resources_servers.geobenchx.task_data
```

Expected output: 202 JSONL rows in `data/benchmark.jsonl`.

## Run

```bash
.venv/bin/gym env start \
  "+config_paths=[responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/geobenchx/configs/geobenchx.yaml]"

.venv/bin/gym eval run --no-serve \
  --agent geobenchx_simple_agent \
  --input resources_servers/geobenchx/data/benchmark.jsonl \
  --output resources_servers/geobenchx/data/rollouts.jsonl \
  --limit 202 --num-repeats 1
```

## Data mode

The default tools simulate catalog and dataflow state, which is sufficient for the
paper's matching-based evaluation and avoids redistributing uncleared external data.
It does not claim to reproduce final map pixels or numerical GIS outputs. See
`DATA_ATTESTATION.md` before using any records or external snapshots for training.

## Regression metrics

Each rollout returns:

- `geobenchx_match`
- normalized reward
- solvable/rejection behavior
- candidate and mean reference step counts
- relative solution-length delta
- matched-reference index and step coverage

Aggregate these by `group` and `task_kind` to reproduce the paper's main regression
cuts: task complexity, solvable tasks, and intentionally unsolvable tasks.
