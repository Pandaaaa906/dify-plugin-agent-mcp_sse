"""
Agent scratchpad storage utilities for short-term memory.

Uses Dify Plugin's storage API to persist agent_scratchpad across invocations,
keyed by conversation_id to isolate different conversations.
"""

import logging
from typing import Any, Tuple

import orjson
from dify_plugin.core.runtime import Session
from dify_plugin.entities.agent import AgentRuntime

logger = logging.getLogger(__name__)


class AgentScratchpadStorageMixin:
    """
    Mixin for agent strategies to persist scratchpad across invocations.

    Uses Dify Plugin's storage API to save agent_scratchpad keyed by
    app_id and conversation_id, enabling short-term memory within the same conversation.

    Required Attributes:
        runtime: AgentRuntime - The agent runtime instance
        session: Session - The Dify plugin session with storage access

    Usage:
        class MyStrategy(AgentScratchpadStorageMixin, AgentStrategy):
            def _invoke(self, parameters):
                # Load existing scratchpad
                scratchpad = self.agent_scratchpad

                # Append new entry
                self.append_agent_scratchpad({"role": "assistant", "content": "..."})

                # Or replace entire scratchpad
                self.agent_scratchpad = new_scratchpad

                # Clear when done
                self.clear_agent_scratchpad()

    Storage Key Format:
        agent_scratchpad:{app_id}:{conversation_id}

    Limits:
        - Maximum 5 entries retained (older entries truncated)
        - Data persists only within the same conversation
    """
    runtime: AgentRuntime
    session: Session
    _max_scratchpad: int = 5
    __key: str = None
    _agent_scratchpad: Tuple[dict[str, Any]] | None = None

    @property
    def _key(self) -> str:
        """Build storage key from app_id and conversation_id.

        Returns:
            Storage key string in format: agent_scratchpad:{app_id}:{conversation_id}

        Note:
            Key is cached after first access to avoid rebuilding.
        """
        if self.__key is not None:
            return self.__key
        self.__key = f"agent_scratchpad:{self.session.app_id}:{self.session.conversation_id}"
        return self.__key

    @property
    def agent_scratchpad(self) -> Tuple[dict[str, Any]]:
        """Get scratchpad from storage.

        Loads scratchpad entries from persistent storage on first access,
        then caches them in memory for subsequent accesses within the
        same invocation.

        Returns:
            Tuple of scratchpad entries (empty tuple if not found or on error)

        Note:
            Returns immutable tuple to prevent accidental in-place modification.
            Use setter or append_agent_scratchpad() to modify.
        """
        if self._agent_scratchpad is not None:
            return self._agent_scratchpad
        try:
            if not self.session.storage.exist(self._key):
                logger.debug("No existing scratchpad found for key: %s", self._key)
                self._agent_scratchpad = tuple()
                return self._agent_scratchpad

            data = self.session.storage.get(self._key)
            scratchpad = orjson.loads(data)

            if not isinstance(scratchpad, list):
                logger.warning(
                    "Invalid scratchpad format, expected list got %s",
                    type(scratchpad).__name__
                )
                self._agent_scratchpad = tuple()
                return self._agent_scratchpad

            scratchpad = tuple(scratchpad)
            logger.debug(
                "Loaded scratchpad with %d entries for conversation: %s",
                len(scratchpad),
                self.session.conversation_id
            )
            self._agent_scratchpad = scratchpad
            return self._agent_scratchpad

        except Exception as e:
            logger.error("Failed to load scratchpad: %s", e)
            self._agent_scratchpad = None
            return tuple()

    @agent_scratchpad.setter
    def agent_scratchpad(self, scratchpad: Tuple[dict[str, Any]]) -> None:
        """Set scratchpad entries.

        Replaces the entire scratchpad with new entries. Automatically
        truncates to _max_scratchpad entries (oldest entries removed).

        Args:
            scratchpad: New scratchpad entries as tuple of dicts
        """
        self._set_agent_scratchpad(scratchpad[-self._max_scratchpad:])

    def append_agent_scratchpad(self, item: dict[str, Any]):
        """Append a single entry to scratchpad.

        Loads existing scratchpad, appends the new item, and saves back.
        Automatically truncates to _max_scratchpad entries if exceeded.

        Args:
            item: New scratchpad entry dict to append
        """
        scratchpad = tuple((
            *(self.agent_scratchpad or ()),
            item
        ))
        self._set_agent_scratchpad(scratchpad)

    def _set_agent_scratchpad(self, scratchpad: Tuple[dict[str, Any]]):
        """Persist scratchpad to storage.

        Internal method that serializes and saves scratchpad to persistent
        storage, then updates the in-memory cache.

        Args:
            scratchpad: Scratchpad entries to persist (already truncated)

        Raises:
            Exception: If storage operation fails
        """
        try:
            data = orjson.dumps(scratchpad[-self._max_scratchpad:])
            self.session.storage.set(self._key, data)

            logger.debug(
                "Saved scratchpad with %d entries for conversation: %s",
                len(scratchpad),
                self.session.conversation_id
            )
            self._agent_scratchpad = scratchpad
        except Exception as e:
            logger.error("Failed to save scratchpad: %s", e)
            raise

    def clear_agent_scratchpad(self) -> None:
        """Remove scratchpad from storage.

        Deletes the persisted scratchpad from storage for the current
        conversation. Safe to call even if no scratchpad exists.
        """
        try:
            if self.session.storage.exist(self._key):
                self.session.storage.delete(self._key)
                logger.debug(
                    "Cleared scratchpad for conversation: %s",
                    self.session.conversation_id
                )
        except Exception as e:
            logger.error("Failed to clear scratchpad: %s", e)

