# GeoBenchX source and data attestation

This file records provenance; it is not a legal approval for model training.

## Included benchmark records

- Source: `https://github.com/Solirinai/GeoBenchX`
- Pinned commit: `0edf6109ea5dca3e70fdfe081d39c9ccd0f832fb`
- Included files:
  - `upstream/tasks_and_reference_solutions.json`
  - `upstream/evaluator_tuning_set.json`
  - `upstream/LICENSE`
- Upstream license: MIT, copyright 2025 Varvara Krechetova.
- Integrity: the upstream commit is recorded in `upstream/COMMIT`.
- Upstream methodology: seed tasks came from practitioners and publications; additional
  tasks were generated using Claude 3.5 Sonnet, then manually reviewed. Reference solutions
  were drafted with GPT-4o or Claude 3.5 Sonnet and manually reviewed/rewritten.
- No claim is made here that model-generated content is approved for NVIDIA model training.
  That requires the relevant internal data review.

## Not included

The 18 statistical, 21 vector, and 11 raster snapshots used by the original executable
tools are **not redistributed**. Upstream publishes them as a separate Google Drive archive
and each underlying source has its own terms. They must receive source-by-source license,
commercial-use, privacy, and generated-content review before use in training.

The default Gym tool backend is therefore a deterministic catalog/dataflow simulator. This
matches what the paper evaluates—the semantic tool-call solution—without silently claiming
the external data is cleared. An execution backend can be added after the snapshot bundle
is reviewed and mounted.

## Generated outputs

`data/benchmark.jsonl` is a mechanical conversion of the 202 official task records into
NemoGym Responses API rows. It adds the paper's system rules and tool JSON schemas; it does
not add synthetic task or answer content.

## Paper

Krechetova, V. and Kochedykov, D. (2025), “GeoBenchX: Benchmarking LLMs in Agent Solving
Multistep Geospatial Tasks,” arXiv:2503.18129v2 / ACM GeoGenAgent 2025.
