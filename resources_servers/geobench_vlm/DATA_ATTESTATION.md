# GEOBench-VLM source and data attestation

This file records provenance. It is not a legal approval for model training.

## Code

- Source: `https://github.com/The-AI-Alliance/GEO-Bench-VLM`
- Pinned commit: `bd07b001cb7e19c83b34988b20e4d6905df077a1`
- Upstream license: Apache-2.0
- Paper: Danish et al., "GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks," ICCV 2025, arXiv:2411.19325

## Dataset

- Hugging Face: `aialliance/GEOBench-VLM`
- Single-image split in this release: 3,211 questions in `Single.parquet`, about 1.4 GB including images
- Additional archives on the dataset repo: `Single.zip`, `Temporal.zip`, `Captioning.zip`, `Ref-Det.zip`, `Ref-Seg.zip`
- The paper reports more than 10,000 manually verified instructions. The released single-image records expand to that count through multiple prompts per question.
- MCQ options and captions were generated with GPT-4o and then manually reviewed. No claim is made here that this generated text is approved for NVIDIA model training.
- Imagery comes from existing remote-sensing collections. Those collections have their own terms. This Gym does not redistribute them.

## What this Gym includes

`data/example.jsonl` is five synthetic questions with a 1-pixel placeholder image so the server, prompt, and letter scorer can be tested without the upstream archive. Point `load_qa_tree` at a local copy of the official dataset to build the real rows.

## Registered training sources

These rows are an inventory only. No source has been fetched, and training has not started. `training_sources.initiate` refuses until a later, explicit fetch step. Recording a license here is not a training approval. Imagery rights and model-generated text still need review before any source is used for training.

| Dataset | Training use | Prepared data | Upstream source(s) | License |
| --- | --- | --- | --- | --- |
| BigEarthNet v2 | Multilabel land-cover classification, scene QA, and grounded captions | 6,000 Sentinel-2 RGB patches (B02/B03/B04) with matching reference maps | Modified Copernicus Sentinel-2; CLC2018 / Copernicus Land Monitoring Service reference labels | CDLA-Permissive-1.0 |
| EuroSAT | Ten-class scene classification and class-grounded captions | 27,000 paired RGB and 13-band multispectral patches | Copernicus Sentinel-2 imagery | MIT |
| USDA Cropland Data Layer 2020–2023 | Crop and land-cover labels paired with date-compatible imagery | Four national 30 m annual rasters with metadata and overviews | USDA National Agricultural Statistics Service | Public domain; free to redistribute |
| NAIP | High-resolution captions and building/road geometry; later object tasks where labels permit | 89 mosaics across 43 AOIs | USDA FPAC Business Center, Geospatial Enterprise Operations; Planetary Computer and USDA ImageServer acquisitions | Public domain; provider credit requested |
| Microsoft Global ML Building Footprints | Model-derived building polygons joined to eligible NAIP imagery | 550,698 polygon features across 43 AOIs | Microsoft Global ML Building Footprints; underlying imagery rights remain separate | CDLA-Permissive-2.0 for the identified release |
| OpenStreetMap | Road, building, water, aeroway, and contextual geometry joined to eligible imagery | Extracts for 43 AOIs | OpenStreetMap contributors | ODbL 1.0 |
| SpaceNet 2 | Building localization and geometry QA | 10,592 complete image/label pairs | SpaceNet Partners and original imagery/annotation contributors | CC BY-SA 4.0 |
| SpaceNet 3 | Road-centerline localization and geometry QA | 2,549 complete image/label pairs; 231 additional images without labels are audit-only | SpaceNet Partners and original imagery/annotation contributors | CC BY-SA 4.0 |
