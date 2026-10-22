"""Tests for Act 8d: tool use as a RATE, not a binary.

The expectation is a measured negative -- a 6.3M base model probably cannot
emit a well-formed ``CALC(...)`` call. That is a legitimate result, and it has
to be reported as a rate with the generations shown, not as a single boolean.

The two rates are distinct and both are required:
  * well-formed call rate  -- the generation contains a parseable CALC(...)
  * correct evaluation    -- of those, the expression evaluates to the answer

Collapsing them into one number would hide whether a model that CAN call the
tool calls it correctly.
"""

from __future__ import annotations

import unittest


class TestToolCallParsing(unittest.TestCase):
    def test_finds_a_balanced_call(self):
        from src.eval.act8_tools import parse_tool_call

        expr, value = parse_tool_call("The answer is CALC((2+3)*4).")
        self.assertEqual(expr, "(2+3)*4")
        self.assertEqual(value, 20)

    def test_ignores_an_unterminated_call(self):
        from src.eval.act8_tools import parse_tool_call

        self.assertIsNone(parse_tool_call("CALC(1+"))
        self.assertIsNone(parse_tool_call("no call at all"))

    def test_rejects_a_call_the_calculator_refuses(self):
        """A syntactically complete but unsafe expression is NOT a well-formed
        tool call for our purposes: it could not have produced an answer."""
        from src.eval.act8_tools import parse_tool_call

        self.assertIsNone(parse_tool_call("CALC(__import__('os').system('ls'))"))

    def test_returns_none_when_absent(self):
        from src.eval.act8_tools import parse_tool_call

        self.assertIsNone(parse_tool_call("The answer is 42."))


class TestToolUseRates(unittest.TestCase):
    def test_rates_are_computed_over_all_generations(self):
        from src.eval.act8_tools import tool_use_summary

        gens = [
            {"generation": "CALC(2+2)", "seed": 42, "item": 0},
            {"generation": "CALC(2+3)", "seed": 42, "item": 0},   # well-formed, wrong
            {"generation": "no call", "seed": 42, "item": 0},     # no call
            {"generation": "CALC(3+3)", "seed": 43, "item": 0},   # well-formed, wrong
        ]
        s = tool_use_summary(gens, expected_by_item={0: 4})
        self.assertEqual(s["n_generations"], 4)
        self.assertEqual(s["well_formed_calls"], 3)
        self.assertAlmostEqual(s["well_formed_rate"], 3 / 4)
        # of the 3 well-formed calls, only the first evaluates to 4
        self.assertEqual(s["correct_calls"], 1)
        self.assertAlmostEqual(s["correct_of_well_formed_rate"], 1 / 3, places=3)

    def test_direct_answer_rate_is_separate_from_tool_use(self):
        """A model that just says the number is not using the tool."""
        from src.eval.act8_tools import tool_use_summary

        gens = [{"generation": "The answer is 4", "seed": 42, "item": 0}]
        s = tool_use_summary(gens, expected_by_item={0: 4})
        self.assertEqual(s["well_formed_calls"], 0)
        self.assertEqual(s["direct_correct"], 1)
        self.assertAlmostEqual(s["direct_correct_rate"], 1.0)

    def test_empty_generations_is_rejected(self):
        from src.eval.act8_tools import tool_use_summary

        with self.assertRaises(ValueError):
            tool_use_summary([], expected_by_item={0: 1})

    def test_unknown_label_is_rejected(self):
        from src.eval.act8_tools import tool_use_summary

        with self.assertRaises(ValueError):
            tool_use_summary([{"generation": "x", "seed": 1, "item": 0,
                                "label": "bogus"}],
                             expected_by_item={0: 1})


class TestToolPrompts(unittest.TestCase):
    def test_prompt_uses_the_shared_instruction_format(self):
        """8a fixed D2: RAG, SFT and tool use must share one prompt contract."""
        from src.adaptation.instructions import InstructionExample, InstructionFormatter
        from src.eval.act8_tools import build_tool_prompt

        prompt = build_tool_prompt("What is 35 + 12?")
        expected = InstructionFormatter().prompt(
            InstructionExample(instruction="What is 35 + 12?", response="",
                               input_text=""))
        self.assertEqual(prompt, expected)
        self.assertIn("### Response:\n", prompt)

    def test_expression_appears_in_the_prompt(self):
        from src.eval.act8_tools import build_tool_prompt

        self.assertIn("35 + 12", build_tool_prompt("What is 35 + 12?"))


if __name__ == "__main__":
    unittest.main()
