"""Tests for Act 7: task splits, SFT mechanics, and the PEFT family.

Two of these guard defects this act had to fix:
  * D4 full-batch SFT  -> batch_size must actually batch and shuffle
  * D5 no terminator   -> EOS must be a supervised target
  * split leakage      -> train/held-out/few-shot must be provably disjoint

and one guards a bug found while building it: LoRA's target list was
("qkv", "projection") but MiniGPT has SEPARATE q and kv projections, so LoRA was
silently adapting one matrix out of the intended three.
"""

from __future__ import annotations

import unittest

import torch

from src.adaptation.peft import (LORA_TARGETS, METHODS, PEFTSpec, apply_peft,
                                 clamp_to_context, n_trainable_for, size_to_budget)
from src.adaptation.sft import EOS_ID, SFTConfig, collate, encode_example, sft_train
from src.data.instruction_tasks import TASKS, assert_disjoint, build_split, make_splits


class _Tok:
    """Minimal deterministic tokenizer for shape-level tests."""

    def encode(self, text: str) -> list[int]:
        # Must stay inside the 64-token vocab these tests build.
        return [(ord(c) % 60) + 1 for c in text][:64]

    def decode(self, ids) -> str:
        return "".join(chr((i % 60) + 31) for i in ids)


class TestTaskSplits(unittest.TestCase):
    def test_splits_are_disjoint(self):
        s = make_splits(n_train=120, n_held_out=40, n_icl=8)
        self.assertEqual((len(s.train), len(s.held_out), len(s.icl)), (120, 40, 8))
        assert_disjoint({"train": s.train, "held_out": s.held_out, "icl": s.icl})

    def test_no_add_item_appears_in_two_splits(self):
        """'add' has only 100 instances, so it must be partitioned, not reshuffled."""
        s = make_splits(n_train=200, n_held_out=60, n_icl=8)
        add = lambda xs: [e.instruction for e in xs if "plus" in e.instruction]
        tr, ho = set(add(s.train)), set(add(s.held_out))
        self.assertFalse(tr & ho, f"add leakage: {sorted(tr & ho)[:5]}")

    def test_splits_are_deterministic(self):
        a = make_splits(n_train=50, n_held_out=20, n_icl=8)
        b = make_splits(n_train=50, n_held_out=20, n_icl=8)
        self.assertEqual([e.response for e in a.train], [e.response for e in b.train])

    def test_duplicate_within_split_is_allowed_across_splits_is_not(self):
        """The guard must distinguish the two, not fire on harmless oversampling."""
        ex = make_splits(n_train=20, n_held_out=10, n_icl=4).train[0]
        assert_disjoint({"a": [ex, ex], "b": []})  # within-split: fine
        with self.assertRaises(AssertionError):
            assert_disjoint({"a": [ex], "b": [ex]})

    def test_every_task_is_present_and_answered(self):
        pool = [(a, b) for a in range(10) for b in range(10)]
        s = build_split(TASKS, len(TASKS) * 4, seed=7, add_pool=pool)
        self.assertEqual(len(s), len(TASKS) * 4)
        for e in s:
            self.assertTrue(e.response, f"empty response for {e.instruction!r}")

    def test_input_text_is_visible_in_the_prompt(self):
        """THE BUG THAT INVALIDATED THE FIRST ACT 7 RUN.

        ``InstructionExample`` is (instruction, response, input_text). Passing
        the payload positionally bound it to `response` and left `input_text`
        empty, so the rendered prompt showed NO input while scoring still
        compared against it. Every input-dependent task became unanswerable,
        and the run emitted near-zero accuracies that read exactly like a real
        negative result.

        The first version of this test simply skipped any example with an empty
        input_text, so reverting ONE task still passed -- it skipped that task
        instead of flagging it. Hence the explicit expected set below: a task
        that should carry input but does not must fail, not vanish.
        """
        from src.adaptation.instructions import InstructionFormatter
        fmt = InstructionFormatter()
        pool = [(a, b) for a in range(10) for b in range(10)]
        s = build_split(TASKS, len(TASKS) * 4, seed=11, add_pool=pool)

        INPUT_DEPENDENT = {"copy", "upper", "first_word", "last_word",
                           "reverse_words", "count_words"}
        INSTRUCTION_TEXT = {
            "Copy the text exactly.": "copy",
            "Rewrite the text in capital letters.": "upper",
            "Give the first word of the text.": "first_word",
            "Give the last word of the text.": "last_word",
            "Write the words in reverse order.": "reverse_words",
            "How many words are in the text?": "count_words",
        }

        seen: set[str] = set()
        for e in s:
            name = INSTRUCTION_TEXT.get(e.instruction)
            if name is None:
                # Unmapped instructions must be the 'add' task, which carries
                # its operands in the instruction and needs no input_text.
                self.assertIn("plus", e.instruction, "unrecognised instruction")
                self.assertFalse(e.input_text, "add must not carry an input_text")
                continue
            seen.add(name)
            self.assertTrue(e.input_text,
                            f"{name} produced an EMPTY input_text -- the payload "
                            "was almost certainly passed positionally into the "
                            "response field")
            self.assertIn(e.input_text, fmt.prompt(e),
                          f"input {e.input_text!r} missing from prompt for {name}")

        self.assertEqual(seen, INPUT_DEPENDENT,
                         f"expected tasks {INPUT_DEPENDENT - seen} never exercised")


class TestSFTMechanics(unittest.TestCase):
    def test_prompt_tokens_are_masked_and_eos_is_supervised(self):
        """D5: the terminator must be a target, so the model learns to stop."""
        from src.adaptation.instructions import InstructionExample
        ex = InstructionExample("Copy the text exactly.", "abc")
        ids, targets = encode_example(ex, _Tok(), max_length=64)
        self.assertEqual(targets[-1], EOS_ID, "EOS must be the final supervised target")
        self.assertTrue(any(t == -100 for t in targets), "prompt must be masked")
        supervised = [t for t in targets if t != -100]
        self.assertEqual(len(supervised), len(ex.response) + 1,
                         "supervision covers the response plus its terminator")

    def test_collate_pads_to_max_length(self):
        batch = [(list(range(10)), list(range(10))), (list(range(5)), list(range(5)))]
        x, y = collate(batch, max_length=16)
        self.assertEqual(x.shape, (2, 16))
        self.assertEqual(y.shape, (2, 16))
        self.assertTrue(torch.equal(y[1, 5:], torch.full((11,), -100)))

    def test_minibatching_actually_batches(self):
        """D4: one optimizer step per minibatch, not one per whole split."""
        from src.adaptation.instructions import InstructionExample
        from src.model.gpt import GPTConfig, MiniGPT
        torch.manual_seed(0)
        model = MiniGPT(GPTConfig(vocabulary_size=64, context_length=64, embedding_dim=32,
                                  num_heads=4, num_layers=2, dropout=0.0,
                                  normalization="layernorm"))
        examples = [InstructionExample("Copy the text.", f"x{i}") for i in range(8)]
        seen: list[int] = []
        out = sft_train(model, examples, _Tok(),
                        SFTConfig(epochs=1, batch_size=4, warmup_steps=1),
                        on_step=lambda s, l: seen.append(s))
        self.assertEqual(out["steps"], 2.0, "8 examples at batch_size 4 = 2 steps")
        self.assertEqual(len(seen), 2)

    def test_empty_training_set_is_rejected(self):
        from src.model.gpt import GPTConfig, MiniGPT
        model = MiniGPT(GPTConfig(vocabulary_size=64, context_length=64, embedding_dim=32,
                                  num_heads=4, num_layers=2, dropout=0.0,
                                  normalization="layernorm"))
        with self.assertRaises(ValueError):
            sft_train(model, [], _Tok())


class TestPEFTFamily(unittest.TestCase):
    def _model(self):
        from src.model.gpt import GPTConfig, MiniGPT
        torch.manual_seed(3)
        return MiniGPT(GPTConfig(vocabulary_size=64, context_length=64, embedding_dim=32,
                                 num_heads=4, num_kv_heads=4, num_layers=2,
                                 dropout=0.0, normalization="layernorm")).eval()

    def test_lora_targets_actually_exist(self):
        """LoRA must wrap q, kv AND projection — MiniGPT has no fused 'qkv'."""
        for suffix in LORA_TARGETS:
            base = getattr(self._model().blocks[0].attention, suffix, None)
            self.assertIsInstance(base, torch.nn.Linear,
                                  f"LoRA target {suffix!r} does not exist on MiniGPT")

    def test_lora_freezes_the_base(self):
        model = self._model()
        wrapped = apply_peft(model, PEFTSpec(method="lora", size=4))
        trainable = {id(p) for p in wrapped.trainable_parameters()}
        base_trainable = [n for n, p in model.named_parameters()
                          if p.requires_grad and id(p) not in trainable]
        self.assertEqual(base_trainable, [], "LoRA left base parameters trainable")

    def test_every_method_produces_a_forward_and_counts_its_parameters(self):
        for method in ("lora", "prompt_tuning", "prefix_tuning", "adapters"):
            with self.subTest(method=method):
                wrapped = apply_peft(self._model(), PEFTSpec(method=method, size=4))
                x = torch.randint(0, 64, (2, 16))
                y = torch.randint(0, 64, (2, 16))
                logits, loss, _ = wrapped(x, y)
                # prompt_tuning prepends virtual tokens, so length grows.
                self.assertEqual(logits.shape[0], 2)
                self.assertGreaterEqual(logits.shape[1], 16)
                self.assertIsNotNone(loss)
                self.assertGreater(wrapped.n_trainable(), 0)

    def test_prefix_tuning_does_not_change_base_weights(self):
        model = self._model()
        before = model.blocks[0].attention.q.weight.clone()
        wrapped = apply_peft(model, PEFTSpec(method="prefix_tuning", size=4))
        logits, _, _ = wrapped(torch.randint(0, 64, (2, 16)))
        self.assertGreater(float(logits.abs().sum()), 0.0)
        self.assertTrue(torch.equal(before, model.blocks[0].attention.q.weight))

    def test_matched_budget_picks_comparable_sizes(self):
        model = self._model()
        budget = 2000
        counts = {}
        for method in ("lora", "prompt_tuning", "prefix_tuning", "adapters"):
            spec = size_to_budget(model, method, budget)
            counts[method] = n_trainable_for(model, spec)
        self.assertEqual(set(counts), set(METHODS) - {"full"})
        for method, count in counts.items():
            self.assertLess(abs(count - budget) / budget, 0.75,
                            f"{method} landed at {count}, far from budget {budget}")

    def test_prompt_tuning_size_is_clamped_to_context(self):
        spec = clamp_to_context(PEFTSpec(method="prompt_tuning", size=500), 128)
        self.assertLessEqual(spec.size, 64)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_peft(self._model(), PEFTSpec(method="magic", size=1))


class TestPEFTLossAlignment(unittest.TestCase):
    """Guards the double-shift that made three PEFT methods score 0.0000.

    MiniGPT computes ``CE(logits, targets)`` with NO internal shift, because the
    SFT loop passes labels already aligned to the inputs. The first PEFT
    implementations used ``CE(logits[:, :-1], targets[:, 1:])``, shifting a
    second time. They trained to predict the wrong position and -- because
    scoring was then done at the correct positions -- reported exactly 0.000
    accuracy while the loss was genuinely decreasing.
    """

    def _model(self, seed=1):
        from src.model.gpt import GPTConfig, MiniGPT
        torch.manual_seed(seed)
        return MiniGPT(GPTConfig(vocabulary_size=64, context_length=64,
                                 embedding_dim=32, num_heads=4, num_kv_heads=4,
                                 num_layers=2, dropout=0.0,
                                 normalization="layernorm")).eval()

    def test_loss_is_plain_cross_entropy_over_the_returned_logits(self):
        import torch.nn.functional as F
        x = torch.randint(1, 64, (3, 14))
        y = torch.randint(1, 64, (3, 14))
        for method, size in (("lora", 4), ("prompt_tuning", 8),
                             ("prefix_tuning", 4), ("adapters", 4), ("full", 1)):
            with self.subTest(method=method):
                wrapped = apply_peft(self._model(), PEFTSpec(method=method, size=size))
                with torch.no_grad():
                    logits, _, _ = wrapped(x)
                    extra = logits.shape[1] - x.shape[1]
                    padded = (torch.cat([torch.full((y.size(0), extra), -100,
                                                    dtype=y.dtype), y], dim=1)
                              if extra else y)
                    expected = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        padded.reshape(-1), ignore_index=-100)
                    _, loss, _ = wrapped(x, y)
                self.assertTrue(
                    torch.allclose(loss, expected, atol=1e-5),
                    f"{method}: loss {float(loss):.6f} != CE(logits, targets) "
                    f"{float(expected):.6f} -- a second shift has been applied")

    def test_response_slice_predicts_the_response_for_every_method(self):
        """The scoring indices must land on the response, whatever the offset."""
        from src.eval.act7_sft import response_slice
        n_prompt, n_resp, seq = 10, 4, 14
        for method, size in (("lora", 4), ("prompt_tuning", 8),
                             ("prefix_tuning", 4), ("adapters", 4), ("full", 1)):
            with self.subTest(method=method):
                wrapped = apply_peft(self._model(), PEFTSpec(method=method, size=size))
                x = torch.randint(1, 64, (1, seq))
                with torch.no_grad():
                    logits, _, _ = wrapped(x)
                extra = logits.shape[1] - x.shape[1]
                start, stop = response_slice(logits.shape[1], x.shape[1],
                                             n_prompt, n_resp)
                self.assertEqual(stop - start + 1, n_resp)
                predicted = [i - extra + 1 for i in range(start, stop + 1)]
                self.assertEqual(predicted, list(range(n_prompt, n_prompt + n_resp)),
                                 f"{method}: scoring window lands off the response")
                self.assertGreaterEqual(start, 0)

    def test_response_slice_matches_the_naive_slice_when_extra_is_zero(self):
        """Pin the sign against the case that was already correct."""
        from src.eval.act7_sft import response_slice
        n_prompt, n_resp, seq = 10, 3, 12
        logits_len = input_len = seq
        start, stop = response_slice(logits_len, input_len, n_prompt, n_resp)
        full_len = n_prompt + n_resp
        self.assertEqual((start, stop), (full_len - 1 - n_resp, full_len - 2))
        self.assertEqual((start, stop), (input_len - n_resp, input_len - 1))


class TestICLPositiveControl(unittest.TestCase):
    """Permanent record of the control that gated Act 7's write-up.

    7a (zero-shot) and 7b (ICL) both read exactly 0.0000, which is the D11
    signature -- so before either could be written up, the control had to settle
    whether the scoring path could register a positive at all.

    Outcome: the harness was sound (an oracle scoring through the exact ICL path
    scores 1.0000), but the *prompt* was defective -- D13. The conventional
    4-shot prompt is ~328 tokens against a 128-token context, so truncation
    removed the demonstrations and the model had been scored on a truncated
    tail. The oracle control is retained here as a regression test; the fitting
    assertions below make D13 impossible to reintroduce quietly.
    """

    @classmethod
    def setUpClass(cls):
        from src.adaptation.instructions import InstructionFormatter
        from src.eval.act7_sft import load_canonical
        from src.tokenizer.bpe import BPETokenizer

        cls.tok = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
        cls.splits = make_splits(n_train=64, n_held_out=8, n_icl=8)
        cls.fmt = InstructionFormatter()
        cls.ctx = load_canonical("6.3M").config.context_length

    def test_four_shot_prompt_does_not_fit_the_context(self):
        """The premise of D13, asserted so the numbers stay honest.

        If this ever fails the window changed and the finding needs re-checking.
        """
        from src.eval.act7_sft import icl_prompt

        ex = self.splits.held_out[0]
        four = self.tok.encode(icl_prompt(self.fmt, self.splits.icl[:4], ex))
        self.assertGreater(len(four), self.ctx,
                           "a 4-shot prompt now fits; D13's premise is gone")

    def test_fitted_shots_always_fit_and_are_present(self):
        from src.eval.act7_sft import fit_icl_shots, icl_prompt

        for ex in self.splits.held_out:
            shots = fit_icl_shots(self.tok, self.fmt, self.splits.icl, ex, self.ctx)
            self.assertGreaterEqual(len(shots), 1,
                                    f"no demonstration fits for {ex.instruction!r}")
            resp = len(self.tok.encode(ex.response)) + 1
            self.assertLessEqual(
                len(self.tok.encode(icl_prompt(self.fmt, shots, ex))) + resp,
                self.ctx, "fitted prompt overflows the window")

            # and every selected demonstration survives into the model input
            body = self.tok.decode(
                (self.tok.encode(icl_prompt(self.fmt, shots, ex))
                 + self.tok.encode(ex.response) + [0])[-(self.ctx + 1):-1])
            for shot in shots:
                self.assertIn(shot.response, body,
                              f"demonstration for {shot.instruction!r} truncated")

    def test_oracle_scores_one_on_the_icl_path(self):
        """THE positive control: the ICL scoring path CAN emit a non-zero.

        The oracle predicts the token that actually follows each position, so
        teacher-forced alignment must recover the response exactly. A harness
        that could not register a positive would score 0 here, and 7a/7b would
        be void.
        """
        from src.eval.act7_sft import EOS_ID, fit_icl_shots, icl_prompt, measure_icl

        class ShiftOracle(torch.nn.Module):
            """Next-token identity: predicts x[j+1], EOS at the final position."""

            def __init__(self, ctx):
                super().__init__()
                self._probe = torch.nn.Parameter(torch.zeros(1))
                self._ctx = ctx

            @property
            def config(self):
                return type("Cfg", (), {"context_length": self._ctx})()

            def forward(self, token_ids, targets=None, cache=None):
                b, L = token_ids.shape
                logits = torch.full((b, L, 6487), -50.0)
                for j in range(L - 1):
                    logits[0, j, token_ids[0, j + 1]] = 50.0
                logits[0, L - 1, EOS_ID] = 50.0  # responses always end in EOS
                return logits, None, None

        # Prove the premise the oracle relies on: the scored slice is the
        # response, ending in EOS, immediately after the prompt.
        ex = self.splits.held_out[0]
        shots = fit_icl_shots(self.tok, self.fmt, self.splits.icl, ex, self.ctx)
        resp = self.tok.encode(ex.response) + [EOS_ID]
        full = self.tok.encode(icl_prompt(self.fmt, shots, ex)) + resp
        self.assertEqual(full[-len(resp):], resp)
        self.assertEqual(full[-1], EOS_ID)

        out = measure_icl(ShiftOracle(self.ctx), self.tok, self.splits,
                          device="cpu", seeds=[42])
        self.assertEqual(out["accuracy_mean"], 1.0,
                         "harness cannot register a positive -> 7a/7b would be void")
        self.assertEqual(out["accuracy_runs"], [1.0])

    def test_icl_is_measured_once_not_per_training_size(self):
        """ICL's prompt never depends on n, so per-size rows were a constant
        drawn four times and dressed up as a curve."""
        import json
        from pathlib import Path

        p = Path("results/v2/act7/6.3M_sft_peft.json")
        if not p.exists():
            self.skipTest("results file not present")
        d = json.loads(p.read_text())
        self.assertIn("icl", d, "ICL must be stored as its own single entry")
        self.assertNotIn("icl_4shot",
                         {r.get("method") for r in d.get("icl_vs_sft", [])},
                         "duplicated ICL rows reintroduced")
        self.assertIn("shots_used", d["icl"])

    def test_positive_control_is_recorded_in_results(self):
        import json
        from pathlib import Path

        p = Path("results/v2/act7/6.3M_sft_peft.json")
        if not p.exists():
            self.skipTest("results file not present")
        d = json.loads(p.read_text())
        pc = d.get("positive_control")
        self.assertIsNotNone(pc, "the control's outcome must ship with the numbers")
        self.assertTrue(pc["harness_can_score_nonzero"])
        self.assertEqual(pc["oracle_teacher_forced_scored"], 1)
        self.assertEqual(pc["oracle_greedy_scored"], 1)


if __name__ == "__main__":
    unittest.main()
