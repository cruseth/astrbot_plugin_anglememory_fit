from __future__ import annotations

import logging
import sys
import types
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot-test")

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_astrbot_stubs()

PACKAGE_NAME = "astrbot_plugin_angel_memory"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PACKAGE_NAME] = package

CORE_PACKAGE = f"{PACKAGE_NAME}.core"
if CORE_PACKAGE not in sys.modules:
    package = types.ModuleType(CORE_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "core")]
    sys.modules[CORE_PACKAGE] = package

LLM_MEMORY_PACKAGE = f"{PACKAGE_NAME}.llm_memory"
if LLM_MEMORY_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory")]
    sys.modules[LLM_MEMORY_PACKAGE] = package

LLM_MEMORY_UTILS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.utils"
if LLM_MEMORY_UTILS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_UTILS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "utils")]
    sys.modules[LLM_MEMORY_UTILS_PACKAGE] = package

LLM_MEMORY_COMPONENTS_PACKAGE = f"{LLM_MEMORY_PACKAGE}.components"
if LLM_MEMORY_COMPONENTS_PACKAGE not in sys.modules:
    package = types.ModuleType(LLM_MEMORY_COMPONENTS_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llm_memory" / "components")]
    sys.modules[LLM_MEMORY_COMPONENTS_PACKAGE] = package

from astrbot_plugin_angel_memory.core.plugin_context import PluginContext


class _FakeAstrBotContext:
    def get_all_providers(self):
        return []

    def get_all_embedding_providers(self):
        return []


def _context(tmp_path, config=None) -> PluginContext:
    return PluginContext(
        _FakeAstrBotContext(),
        config or {},
        str(tmp_path),
    )


def test_unmapped_conversation_uses_private_scope(tmp_path):
    context = _context(tmp_path)

    scope, matched_by, matched_key = context.resolve_memory_scope_with_source(
        "aiocqhttp:group:12345",
        persona_name="same-persona",
    )

    assert scope == PluginContext.build_private_scope("aiocqhttp:group:12345")
    assert scope.startswith("session_")
    assert matched_by == "private_conversation"
    assert matched_key == "aiocqhttp:group:12345"


def test_shared_scopes_allow_multiple_conversations_to_share(tmp_path):
    config = {
        "memory_scope": {
            "shared_scopes": {
                "aiocqhttp:group:12345": "family",
                "aiocqhttp:private:10001": "family",
            }
        }
    }
    context = _context(tmp_path, config)

    assert context.resolve_memory_scope("aiocqhttp:group:12345") == "family"
    assert context.resolve_memory_scope("aiocqhttp:private:10001") == "family"


def test_legacy_public_scope_can_restore_old_default(tmp_path):
    context = _context(tmp_path, {"memory_scope": {"legacy_public_scope": True}})

    scope, matched_by, matched_key = context.resolve_memory_scope_with_source("umo:1")

    assert scope == "public"
    assert matched_by == "legacy_public"
    assert matched_key == "public"


def test_legacy_conversation_scope_map_is_compatible_but_not_persona_first(tmp_path):
    context = _context(
        tmp_path,
        {
            "conversation_scope_map": {
                "aiocqhttp:group:12345": "family",
                "same-persona": "persona_scope",
            }
        },
    )

    assert context.resolve_memory_scope("aiocqhttp:group:12345") == "family"
    assert (
        context.resolve_memory_scope(
            "aiocqhttp:group:99999",
            persona_name="same-persona",
        )
        == PluginContext.build_private_scope("aiocqhttp:group:99999")
    )
