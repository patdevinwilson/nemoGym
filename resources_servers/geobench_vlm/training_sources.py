"""Registered training sources. Nothing here is downloaded or trained on.

The catalog records the prepared corpus so a later step can initiate each
source after review. Calling ``initiate`` refuses until that step is explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class TrainingNotStarted(RuntimeError):
    """Training and source initiation stay off."""


@dataclass(frozen=True)
class TrainingSource:
    source_id: str
    dataset: str
    training_use: str
    prepared_data: str
    upstream_sources: tuple[str, ...]
    license: str
    status: str = "registered"
    initiated: bool = False

    def as_record(self) -> dict:
        record = asdict(self)
        record["upstream_sources"] = list(self.upstream_sources)
        return record


TRAINING_SOURCES: tuple[TrainingSource, ...] = (
    TrainingSource(
        source_id="bigearthnet_v2",
        dataset="BigEarthNet v2",
        training_use="Multilabel land-cover classification, scene QA, and grounded captions",
        prepared_data="6,000 Sentinel-2 RGB patches (B02/B03/B04) with matching reference maps",
        upstream_sources=(
            "Modified Copernicus Sentinel-2",
            "CLC2018 / Copernicus Land Monitoring Service reference labels",
        ),
        license="CDLA-Permissive-1.0",
    ),
    TrainingSource(
        source_id="eurosat",
        dataset="EuroSAT",
        training_use="Ten-class scene classification and class-grounded captions",
        prepared_data="27,000 paired RGB and 13-band multispectral patches",
        upstream_sources=("Copernicus Sentinel-2 imagery",),
        license="MIT",
    ),
    TrainingSource(
        source_id="usda_cdl_2020_2023",
        dataset="USDA Cropland Data Layer 2020–2023",
        training_use="Crop and land-cover labels paired with date-compatible imagery",
        prepared_data="Four national 30 m annual rasters with metadata and overviews",
        upstream_sources=("USDA National Agricultural Statistics Service",),
        license="Public domain; free to redistribute",
    ),
    TrainingSource(
        source_id="naip",
        dataset="NAIP",
        training_use="High-resolution captions and building/road geometry; later object tasks where labels permit",
        prepared_data="89 mosaics across 43 AOIs",
        upstream_sources=(
            "USDA FPAC Business Center, Geospatial Enterprise Operations",
            "Planetary Computer and USDA ImageServer acquisitions",
        ),
        license="Public domain; provider credit requested",
    ),
    TrainingSource(
        source_id="microsoft_building_footprints",
        dataset="Microsoft Global ML Building Footprints",
        training_use="Model-derived building polygons joined to eligible NAIP imagery",
        prepared_data="550,698 polygon features across 43 AOIs",
        upstream_sources=(
            "Microsoft Global ML Building Footprints",
            "underlying imagery rights remain separate",
        ),
        license="CDLA-Permissive-2.0 for the identified release",
    ),
    TrainingSource(
        source_id="openstreetmap",
        dataset="OpenStreetMap",
        training_use="Road, building, water, aeroway, and contextual geometry joined to eligible imagery",
        prepared_data="Extracts for 43 AOIs",
        upstream_sources=("OpenStreetMap contributors",),
        license="ODbL 1.0",
    ),
    TrainingSource(
        source_id="spacenet_2",
        dataset="SpaceNet 2",
        training_use="Building localization and geometry QA",
        prepared_data="10,592 complete image/label pairs",
        upstream_sources=("SpaceNet Partners and original imagery/annotation contributors",),
        license="CC BY-SA 4.0",
    ),
    TrainingSource(
        source_id="spacenet_3",
        dataset="SpaceNet 3",
        training_use="Road-centerline localization and geometry QA",
        prepared_data="2,549 complete image/label pairs; 231 additional images without labels are audit-only",
        upstream_sources=("SpaceNet Partners and original imagery/annotation contributors",),
        license="CC BY-SA 4.0",
    ),
)


def catalog() -> tuple[TrainingSource, ...]:
    return TRAINING_SOURCES


def get_source(source_id: str) -> TrainingSource:
    for source in TRAINING_SOURCES:
        if source.source_id == source_id:
            return source
    known = ", ".join(source.source_id for source in TRAINING_SOURCES)
    raise KeyError(f"Unknown training source {source_id!r}. Known sources: {known}")


def initiate(source_id: str) -> None:
    """Refuse to fetch or train on a registered source."""
    source = get_source(source_id)
    raise TrainingNotStarted(
        f"{source.dataset} is registered and not initiated. "
        "Training stays off until that source is reviewed and fetched in a later step."
    )
