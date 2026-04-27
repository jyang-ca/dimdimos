#!/usr/bin/env python3

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dimos.agents.utils import _chat_openai_kwargs
from dimos.core.global_config import global_config


def test_chat_openai_kwargs_use_placeholder_key_for_local_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    original_base_url = global_config.openai_base_url
    global_config.update(openai_base_url="http://127.0.0.1:10531/v1")

    try:
        kwargs = _chat_openai_kwargs()
    finally:
        global_config.update(openai_base_url=original_base_url)

    assert kwargs["base_url"] == "http://127.0.0.1:10531/v1"
    assert kwargs["api_key"] == "dummy"


def test_chat_openai_kwargs_do_not_invent_key_for_openai_host(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    original_base_url = global_config.openai_base_url
    global_config.update(openai_base_url="https://api.openai.com/v1")

    try:
        kwargs = _chat_openai_kwargs()
    finally:
        global_config.update(openai_base_url=original_base_url)

    assert kwargs == {"base_url": "https://api.openai.com/v1"}
