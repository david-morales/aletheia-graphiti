"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pydantic import BaseModel, ConfigDict, Field

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.tracer import Tracer
from graphiti_core.validation import EdgeValidator, EdgeValidationObserver


class GraphitiClients(BaseModel):
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
    tracer: Tracer
    # Edge validators: empty list means no hooks; behavior identical to pre-Task 8.
    # Populated by Graphiti.__init__ from its edge_validators parameter.
    edge_validators: list[EdgeValidator] = Field(default_factory=list)
    # Validation observer: when set, the validator hook calls record()/record_error()
    # after each decision. None means no metrics are collected (default, pre-existing behavior).
    validation_observer: EdgeValidationObserver | None = None
    # Dry-run flag: when True, validators run and observer records decisions,
    # but drop decisions are converted to keep (no edges actually removed).
    validation_dry_run: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
