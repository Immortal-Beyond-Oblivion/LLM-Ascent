"""A small prompt-chain agent: persistent memory plus a safe calculator tool.

The language model is abstracted as ``generate(prompt) -> continuation`` so the
chain logic is testable with a stub and works with any checkpoint.

Chain per user message:

1. Record the user turn.
2. ``/calc <expr>`` is executed directly (deterministic, no model call).
3. Otherwise build ``preamble + recent memory + "Assistant:"`` and generate.
4. If the reply contains ``CALC(<expr>)``, run the calculator, record the call
   and its ``Tool:`` result as turns, and generate again (at most
   ``max_tool_rounds`` times, so a looping model cannot spin forever).
5. Record the final reply.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .calculator import CalculatorError, evaluate, find_tool_call, format_number
from .memory import ConversationMemory

GenerateFn = Callable[[str], str]

DEFAULT_PREAMBLE = "Answer briefly. For arithmetic write CALC(expression)."
_STOP_MARKERS = ("\nUser:", "\nTool:", "\nAssistant:")
_PROMPT_SUFFIX = "\nAssistant:"


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    tool_limit_reached: bool = False


def _clean(reply: str) -> str:
    for marker in _STOP_MARKERS:
        index = reply.find(marker)
        if index != -1:
            reply = reply[:index]
    return reply.strip()


class Agent:
    def __init__(
        self,
        generate: GenerateFn,
        memory: ConversationMemory | None = None,
        *,
        max_context_chars: int = 1000,
        max_tool_rounds: int = 2,
        preamble: str = DEFAULT_PREAMBLE,
    ) -> None:
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        if max_context_chars <= len(preamble) + len(_PROMPT_SUFFIX) + 1:
            raise ValueError("max_context_chars must exceed the preamble and prompt suffix length")
        self.generate = generate
        self.memory = memory if memory is not None else ConversationMemory()
        self.max_context_chars = max_context_chars
        self.max_tool_rounds = max_tool_rounds
        self.preamble = preamble

    def build_prompt(self) -> str:
        budget = self.max_context_chars - len(self.preamble) - len(_PROMPT_SUFFIX) - 1
        return f"{self.preamble}\n{self.memory.render(max_chars=budget)}{_PROMPT_SUFFIX}"

    @staticmethod
    def _run_calculator(expression: str, log: list[dict[str, str]]) -> str:
        try:
            result = format_number(evaluate(expression))
            log.append({"expression": expression.strip(), "result": result})
            return f"{expression.strip()} = {result}"
        except CalculatorError as error:
            log.append({"expression": expression.strip(), "error": str(error)})
            return f"error: {error}"

    def respond(self, user_text: str) -> AgentResponse:
        text = user_text.strip()
        if not text:
            raise ValueError("user message must not be empty")
        self.memory.add("user", text)
        response = AgentResponse(text="")

        if text.lower().startswith("/calc "):
            outcome = self._run_calculator(text[len("/calc "):], response.tool_calls)
            response.text = outcome if not outcome.startswith("error:") else f"Calculator {outcome}"
            self.memory.add("assistant", response.text)
            return response

        for round_index in range(self.max_tool_rounds + 1):
            prompt = self.build_prompt()
            response.prompts.append(prompt)
            reply = _clean(self.generate(prompt))
            call = find_tool_call(reply)
            if call is None:
                response.text = reply
                break
            if round_index == self.max_tool_rounds:
                response.text = reply
                response.tool_limit_reached = True
                break
            self.memory.add("assistant", reply)
            self.memory.add("tool", self._run_calculator(call[0], response.tool_calls))
        self.memory.add("assistant", response.text or "(no response)")
        return response
