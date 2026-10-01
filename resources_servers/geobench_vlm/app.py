"""NemoGym resources server for GEOBench-VLM."""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from resources_servers.geobench_vlm.scoring import score_answer


class GeoBenchVLMResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class GeoBenchVLMVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    activity: str = "geobench_vlm"
    split: str = "single"
    task: str = ""
    question_id: str = ""
    prompt_index: int = 0
    ground_truth: str = ""
    ground_truth_option: str = ""


class GeoBenchVLMVerifyResponse(BaseVerifyResponse):
    geobench_vlm_correct: bool
    geobench_vlm_first_letter_correct: bool
    predicted_letter: str
    gold_letter: str
    task: str
    split: str


class GeoBenchVLMResourcesServer(SimpleResourcesServer):
    config: GeoBenchVLMResourcesServerConfig

    async def verify(self, body: GeoBenchVLMVerifyRequest) -> GeoBenchVLMVerifyResponse:
        scored = score_answer(
            body.response.output_text if body.response is not None else "",
            body.ground_truth_option,
            body.ground_truth,
        )
        return GeoBenchVLMVerifyResponse(
            **body.model_dump(),
            reward=scored["reward"],
            geobench_vlm_correct=scored["geobench_vlm_correct"],
            geobench_vlm_first_letter_correct=scored["geobench_vlm_first_letter_correct"],
            predicted_letter=scored["predicted_letter"],
            gold_letter=scored["gold_letter"],
            task=body.task,
            split=body.split,
        )


if __name__ == "__main__":
    GeoBenchVLMResourcesServer.run_webserver()
