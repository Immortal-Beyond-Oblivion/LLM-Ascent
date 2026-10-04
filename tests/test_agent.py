from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent import Agent, CalculatorError, ConversationMemory, evaluate, find_tool_call, format_number

try:
    import torch
except ModuleNotFoundError:
    torch = None


class CalculatorTests(unittest.TestCase):
    def test_valid_arithmetic(self) -> None:
        cases = {"(2+3)*4": 20, "7/2": 3.5, "2**10": 1024, "-3 + +5": 2, "10 // 3": 3, "10 % 4": 2, " 6 * 7 ": 42, "0.5 + 0.25": 0.75}
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(evaluate(expression), expected)

    def test_formatting(self) -> None:
        self.assertEqual(format_number(20), "20")
        self.assertEqual(format_number(3.5), "3.5")
        self.assertEqual(format_number(4.0), "4")
        self.assertEqual(format_number(1 / 3), "0.3333333333")

    def test_unsafe_or_invalid_expressions_are_rejected(self) -> None:
        rejected = [
            "__import__('os').system('echo hi')", "open('x')", "().__class__", "abs(-1)", "x + 1",
            "'a' * 3", "True + 1", "1 if 1 else 2", "[1, 2]", "lambda: 1", "1 < 2",
            "2 ** 1000", "9**9**9", "(-8) ** 0.5", "1 / 0", "0 ** -1", "1 // 0", "1e308 * 10",
            "", "   ", "1 +", "9" * 150, "1+" * 150 + "1", "-" * 60 + "1", "1; 2", "1j",
        ]
        for expression in rejected:
            with self.subTest(expression=expression[:40]):
                with self.assertRaises(CalculatorError):
                    evaluate(expression)

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaises(CalculatorError):
            evaluate(5)  # type: ignore[arg-type]

    def test_tool_call_parsing_handles_nesting_and_incomplete_calls(self) -> None:
        text = "It is CALC((2+3)*4) ok"
        expression, start, end = find_tool_call(text)  # type: ignore[misc]
        self.assertEqual(expression, "(2+3)*4")
        self.assertEqual(text[start:end], "CALC((2+3)*4)")
        self.assertIsNone(find_tool_call("CALC(1+"))
        self.assertIsNone(find_tool_call("no call here"))


class MemoryTests(unittest.TestCase):
    def test_render_labels_and_budget_keep_newest_turns(self) -> None:
        memory = ConversationMemory()
        memory.add("user", "first question")
        memory.add("assistant", "first answer")
        memory.add("user", "second question")
        self.assertEqual(memory.render(), "User: first question\nAssistant: first answer\nUser: second question")
        trimmed = memory.render(max_chars=46)
        self.assertEqual(trimmed, "Assistant: first answer\nUser: second question")
        self.assertLessEqual(len(trimmed), 46)
        self.assertEqual(memory.render(max_chars=40), "User: second question")
        self.assertEqual(memory.render(max_chars=8), "question")

    def test_validation(self) -> None:
        memory = ConversationMemory()
        with self.assertRaises(ValueError):
            memory.add("system", "x")
        with self.assertRaises(ValueError):
            memory.add("user", "   ")
        with self.assertRaises(ValueError):
            memory.render(max_chars=0)

    def test_json_round_trip(self) -> None:
        memory = ConversationMemory()
        memory.add("user", "hello")
        memory.add("tool", "1+1 = 2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "memory.json"
            memory.save(path)
            restored = ConversationMemory.load(path)
        self.assertEqual(restored.turns, memory.turns)


class AgentTests(unittest.TestCase):
    @staticmethod
    def scripted(replies: list[str]):
        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return replies[min(len(prompts) - 1, len(replies) - 1)]

        return generate, prompts

    def test_context_persists_across_turns_and_files(self) -> None:
        generate, prompts = self.scripted(["Nice to meet you, Ada.", "Your name is Ada."])
        agent = Agent(generate)
        agent.respond("My name is Ada")
        agent.respond("What is my name?")
        self.assertIn("User: My name is Ada", prompts[1])
        self.assertIn("Assistant: Nice to meet you, Ada.", prompts[1])
        self.assertTrue(prompts[1].endswith("\nAssistant:"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            agent.memory.save(path)
            generate2, prompts2 = self.scripted(["ok"])
            Agent(generate2, ConversationMemory.load(path)).respond("Again?")
        self.assertIn("User: My name is Ada", prompts2[0])

    def test_model_requested_calculator_call_is_executed_and_fed_back(self) -> None:
        generate, prompts = self.scripted(["CALC((2+3)*4)", "The answer is 20."])
        agent = Agent(generate)
        response = agent.respond("What is five plus five times four?")
        self.assertEqual(response.tool_calls, [{"expression": "(2+3)*4", "result": "20"}])
        self.assertEqual(response.text, "The answer is 20.")
        self.assertEqual(len(prompts), 2)
        self.assertIn("Tool: (2+3)*4 = 20", prompts[1])
        self.assertEqual([turn.role for turn in agent.memory.turns], ["user", "assistant", "tool", "assistant"])

    def test_direct_calc_command_never_calls_the_model(self) -> None:
        def forbidden(prompt: str) -> str:
            raise AssertionError("model must not be called for /calc")

        agent = Agent(forbidden)
        self.assertEqual(agent.respond("/calc 2**10").text, "2**10 = 1024")
        unsafe = agent.respond("/calc __import__('os').system('echo hi')")
        self.assertTrue(unsafe.text.startswith("Calculator error:"))
        self.assertIn("error", unsafe.tool_calls[0])
        self.assertEqual(len(agent.memory), 4)

    def test_unsafe_model_generated_call_is_contained(self) -> None:
        generate, _ = self.scripted(["CALC(__import__('os').system('echo hi'))", "I cannot do that."])
        agent = Agent(generate)
        response = agent.respond("do something")
        self.assertIn("error", response.tool_calls[0])
        self.assertIn("Tool: error:", agent.memory.render())
        self.assertEqual(response.text, "I cannot do that.")

    def test_tool_rounds_are_bounded(self) -> None:
        generate, prompts = self.scripted(["CALC(1+1)"])
        response = Agent(generate, max_tool_rounds=2).respond("loop")
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len(response.tool_calls), 2)
        self.assertTrue(response.tool_limit_reached)

    def test_reply_is_cut_at_hallucinated_next_turn_and_empty_is_recorded(self) -> None:
        generate, _ = self.scripted([" Hello\nUser: fake follow-up"])
        self.assertEqual(Agent(generate).respond("hi").text, "Hello")
        empty, _ = self.scripted(["\nUser: only a fake turn"])
        agent = Agent(empty)
        self.assertEqual(agent.respond("hi").text, "")
        self.assertEqual(agent.memory.turns[-1].text, "(no response)")

    def test_prompt_respects_character_budget_and_keeps_latest_turn(self) -> None:
        generate, prompts = self.scripted(["a fairly long assistant reply to fill memory"])
        agent = Agent(generate, max_context_chars=120)
        for index in range(4):
            agent.respond(f"older question number {index}")
        agent.respond("Latest question?")
        self.assertLessEqual(len(prompts[-1]), 120)
        self.assertIn("User: Latest question?", prompts[-1])
        self.assertNotIn("older question number 0", prompts[-1])

    def test_invalid_configuration_and_input(self) -> None:
        generate, _ = self.scripted(["x"])
        with self.assertRaises(ValueError):
            Agent(generate, max_context_chars=10)
        with self.assertRaises(ValueError):
            Agent(generate, max_tool_rounds=-1)
        with self.assertRaises(ValueError):
            Agent(generate).respond("   ")


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ModelAdapterTests(unittest.TestCase):
    def test_agent_runs_end_to_end_with_a_real_model(self) -> None:
        from src.agent.chat import make_generate_fn
        from src.model import GPTConfig, MiniGPT
        from src.tokenizer import CharacterTokenizer

        tokenizer = CharacterTokenizer.from_text("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .:(),\n")
        torch.manual_seed(0)
        model = MiniGPT(GPTConfig(tokenizer.vocabulary_size, context_length=32, embedding_dim=16, num_heads=4, num_layers=1))
        agent = Agent(make_generate_fn(model, tokenizer, max_new_tokens=6), max_context_chars=300)
        first = agent.respond("Hello there")
        second = agent.respond("And again")
        self.assertIsInstance(first.text, str)
        self.assertEqual(len(agent.memory), 4)
        self.assertIn("User: Hello there", second.prompts[0])


if __name__ == "__main__":
    unittest.main()
