from __future__ import annotations

import unittest
from pathlib import Path

from src.prompting import (
    COT_CUE,
    EXEMPLARS,
    STYLES,
    build_prompt,
    clip_generation,
    extract_answer,
    load_tasks,
    normalize,
    score_answer,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

TASKS_PATH = Path(__file__).resolve().parent.parent / "data" / "prompt_tasks.jsonl"


class PromptTemplateTests(unittest.TestCase):
    def test_three_styles_have_expected_structure(self) -> None:
        question = "What is four plus three?"
        self.assertEqual(build_prompt("zero_shot", question), f"Q: {question}\nA:")
        few = build_prompt("few_shot", question, shots=2)
        self.assertIn(f"Q: {EXEMPLARS[0].question}\nA: {EXEMPLARS[0].answer}", few)
        self.assertNotIn(COT_CUE, few)
        self.assertTrue(few.endswith(f"Q: {question}\nA:"))
        cot = build_prompt("cot", question, shots=2)
        self.assertIn(f"The answer is {EXEMPLARS[1].answer}.", cot)
        self.assertTrue(cot.endswith(f"Q: {question}\nA: {COT_CUE}"))
        self.assertEqual(build_prompt("few_shot", question, shots=0), build_prompt("zero_shot", question))

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("tree_of_thought", "x?")
        with self.assertRaises(ValueError):
            build_prompt("cot", "x?", shots=len(EXEMPLARS) + 1)
        with self.assertRaises(ValueError):
            build_prompt("zero_shot", "   ")

    def test_prompts_are_deterministic(self) -> None:
        self.assertEqual(build_prompt("cot", "Why?", shots=3), build_prompt("cot", "Why?", shots=3))

    def test_exemplars_do_not_leak_task_questions_or_answers(self) -> None:
        tasks = load_tasks(TASKS_PATH)
        self.assertGreaterEqual(len(tasks), 5)
        exemplar_text = normalize(" ".join(f"{e.question} {e.reasoning} {e.answer}" for e in EXEMPLARS))
        padded = f" {exemplar_text} "
        for task in tasks:
            self.assertNotIn(normalize(task.question), exemplar_text)
            for answer in task.answers:
                self.assertNotIn(f" {normalize(answer)} ", padded)

    def test_task_and_exemplar_text_avoids_digits(self) -> None:
        text = " ".join(task.question for task in load_tasks(TASKS_PATH))
        text += " ".join(f"{e.question}{e.reasoning}{e.answer}" for e in EXEMPLARS)
        self.assertFalse(any(character.isdigit() for character in text))


class ScoringTests(unittest.TestCase):
    def test_clip_and_extract(self) -> None:
        self.assertEqual(clip_generation(" seven\nQ: next?"), "seven")
        self.assertEqual(extract_answer("few_shot", " seven\nQ: next?"), "seven")
        self.assertEqual(extract_answer("zero_shot", ""), "")
        cot = " Four and three make seven. The answer is Seven.\n\nQ: more"
        self.assertEqual(extract_answer("cot", cot), "Seven")
        self.assertEqual(extract_answer("cot", " I am not sure about this"), "")

    def test_exact_and_contains_scores(self) -> None:
        self.assertEqual(score_answer("Seven.", " Seven.", ["seven"]), {"exact": True, "contains": True})
        self.assertEqual(score_answer("", " It is seven I think", ["seven"]), {"exact": False, "contains": True})
        self.assertEqual(score_answer("sevens", " sevens", ["seven"]), {"exact": False, "contains": False})
        self.assertEqual(score_answer("x", "x", ["three", "3"]), {"exact": False, "contains": False})


@unittest.skipIf(torch is None, "PyTorch unavailable")
class PromptingEvaluationTests(unittest.TestCase):
    def test_evaluation_logs_every_style_with_truncation_flags(self) -> None:
        from src.model import GPTConfig, MiniGPT
        from src.prompting.evaluate import _markdown, evaluate_prompting
        from src.prompting import PromptTask
        from src.tokenizer import CharacterTokenizer

        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ?:.',\n"
        tokenizer = CharacterTokenizer.from_text(alphabet)
        torch.manual_seed(0)
        model = MiniGPT(GPTConfig(tokenizer.vocabulary_size, context_length=16, embedding_dim=16, num_heads=4, num_layers=1))
        tasks = [PromptTask("What is four plus three?", ("seven",)), PromptTask("Which planet is red?", ("mars",))]
        result = evaluate_prompting(model, tokenizer, tasks, shots=1, max_new_tokens=5)
        self.assertEqual(len(result["items"]), len(STYLES) * len(tasks))
        self.assertEqual(set(result["summary"]), set(STYLES))
        few = [item for item in result["items"] if item["style"] == "few_shot"]
        self.assertTrue(all(item["prompt_truncated"] for item in few))
        self.assertTrue(all(item["unknown_prompt_tokens"] == 0 for item in result["items"]))
        self.assertEqual(result["summary"]["few_shot"]["truncated_fraction"], 1.0)
        self.assertIn("| cot |", _markdown(result["summary"]))
        again = evaluate_prompting(model, tokenizer, tasks, shots=1, max_new_tokens=5)
        self.assertEqual([item["generation"] for item in again["items"]], [item["generation"] for item in result["items"]])


if __name__ == "__main__":
    unittest.main()
