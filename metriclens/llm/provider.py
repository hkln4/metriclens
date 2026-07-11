"""Pluggable LLM provider interface.

v0 ships only the Anthropic implementation, but the interpreter depends on this
abstract interface rather than on Anthropic directly, so swapping in another
provider later is a single new file with no changes elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Minimal contract an LLM backend must satisfy for the interpreter.

    Deliberately tiny: the interpreter only needs to send a constructed prompt
    and get back text. Nothing about a specific vendor leaks into this contract.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text response for the given system+user prompt."""
        raise NotImplementedError
