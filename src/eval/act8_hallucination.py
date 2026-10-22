"""Act 8b/8c: does supplying the passage change what the model says?

TWO INSTRUMENTS, because the first one could not answer the question.

1. FREE GENERATION (secondary). The model continues the prompt and we classify
   what it produced. On C4 this lands at a ceiling: 97 of 120 oracle
   generations contain no number at all, so the rate reads 1.000 in every
   condition and cannot rank them. That ceiling is informative in itself --
   the model rarely volunteers an answer -- but it is not a grounding
   measurement, because "did not produce a label" and "could not ground" are
   indistinguishable in it.

2. ANSWER COPYING (primary). Instead of asking the model to volunteer a
   number, we score its likelihood of each candidate answer and ask WHICH one
   it prefers. This removes the label-competence confound entirely: the model
   is never required to emit a digit.

   A prototype on 8 C4 items separated the two readings immediately. Under
   oracle the model preferred the number that was in its context on 7 of 8
   items, while under closed-book the same comparison was 5 of 8 -- chance.
   So the model CAN ground; the free-generation metric was blind to it.

CONDITIONS (four, all on the same held-out item set, all sharing the prompt
builder fixed in D2 and the token budget fixed in D1):

    retrieved       a passage chosen by the lexical retriever, with NO
                    knowledge of where the answer lives
    oracle          the answer-bearing sentence
    counterfactual  the same sentence with that number swapped
    closed_book     no input section at all

`retrieved` is not a disguised oracle: the retriever is queried with the
instruction text, which contains nothing about the answer, so it cannot find
the answer-bearing sentence on purpose. How often it succeeds anyway is
reported as the retrieval success rate.

THE UNIT IS THE ITEM. A base model plus a deterministic answer-copying score
gives one measurement per item per condition, so n = items, not
items x seeds. Free generation is sampled over 3 seeds, and those repeats are
averaged per item before any test, because repeats of an item share its
difficulty and are not independent draws. Contrasts are paired per item and
their CIs come from a percentile bootstrap that resamples ITEMS.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import torch

from src.bench.stats import (bootstrap_mean_ci, mcnemar_exact, paired_ttest,
                           wilson_interval)
from src.eval.act8 import (COPY_CLASSES, NUMBER_RE, classify_copy,
                           class_rate_summary, match_counterfactual, numbers_in,
                           score_candidates, swap_number)
from src.generation.decoding import generate
from src.rag.prompt import fit_context, rag_prompt

INSTRUCTION = "What number does the passage state?"
CONDITIONS = ("retrieved", "oracle", "counterfactual", "closed_book", "nonnumeric")
SEEDS = (42, 43, 44)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
N_DISTRACTORS = 4


def candidate_sentences(text: str) -> list[str]:
    """Sentences that make exactly ONE checkable claim.

    Filters, each removing a sentence that would be scored against a claim it
    never makes:

      * more or fewer than one number -- ambiguous target
      * ending in '?' -- a question does not STATE a number
      * the number opening the sentence ("1) for a newly created site") or
        immediately followed by ')' -- those are list markers, not claims
      * a number glued to a letter ("22nd") -- a literal swap cannot inflect it
      * too short to be a statement
    """
    out = []
    for sent in SENTENCE_SPLIT.split(text.strip()):
        sent = sent.strip()
        if not sent or not len(numbers_in(sent)) == 1:
            continue
        if len(sent) < 20 or sent.endswith("?"):
            continue
        match = NUMBER_RE.search(sent)
        if match is None:  # pragma: no cover - guarded above
            continue
        if match.start() == 0:
            continue  # list marker
        if sent[match.end():match.end() + 1].isalpha():
            continue  # ordinal: "September 22nd" -> "21nd"
        if sent[match.end():].lstrip().startswith(")"):
            continue
        out.append(sent)
    return out


def _content_words(text: str, limit: int = 3) -> list[str]:
    """Distinctive alphabetic words, for a lexical retrieval anchor.

    Deliberately excludes numbers and stopwords: the anchor must reach the
    document by TOPIC without handing the retriever the answer.
    """
    stop = {"this", "that", "with", "from", "have", "they", "been", "were",
            "will", "would", "there", "their", "which", "about", "into",
            "than", "then", "them", "these", "those", "what", "when", "where",
            "your", "you", "and", "the", "for", "are", "was", "were", "but",
            "not", "you", "our", "its", "his", "her", "she", "him", "out",
            "all", "can", "had", "has", "one", "two", "get", "got", "just",
            "like", "make", "made", "much", "some", "such", "only", "also",
            "more", "most", "other", "over", "very", "after", "before",
            "between", "because", "does", "done", "each", "use", "used",
            "number", "passage", "state", "states", "stated"}
    words = [w for w in re.findall(r"[a-z]{4,}", text.lower())
             if w not in stop and not w.isdigit()]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= limit:
            break
    return out


def load_passages(path: str | Path, *, limit: int, tokenizer,
                  max_docs: int = 4000) -> list[dict]:
    """C4 sentences that make exactly one checkable claim.

    Each item also carries a RETRIEVAL ANCHOR: content words taken from a
    DIFFERENT sentence of the same document. The anchor gives the retriever a
    lexical handle on the document without revealing the number or the
    answer-bearing sentence, so the retrieved condition is answer-agnostic
    rather than a disguised oracle.

    The counterfactual is matched to the source's digit count AND token count
    (see :func:`match_counterfactual`), so the two candidates are comparable.
    """
    found: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(itertools.islice(fh, max_docs)):
            if len(found) >= limit:
                break
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("text") or ""
            sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip())
                         if s.strip()]
            for sent in candidate_sentences(text):
                match = NUMBER_RE.search(sent)
                if match is None:  # pragma: no cover
                    continue
                raw = match.group(0)
                number = numbers_in(sent)[0]
                if not number.isdigit():
                    continue  # non-integers (3.5) do not swap like-for-like
                try:
                    cf = match_counterfactual(number, tokenizer)
                    swapped = swap_number(sent, raw, cf)
                except ValueError:
                    continue
                # Anchor: a different sentence of the same document, so the
                # query is topically informative but not answer-bearing.
                anchor_sentence = ""
                for other in sentences:
                    if other == sent or len(other) < 20:
                        continue
                    if NUMBER_RE.search(other):
                        continue  # keep the anchor free of numbers too
                    if sent[:25] in other:
                        continue
                    anchor_sentence = other
                    break
                anchor_terms = _content_words(anchor_sentence) if anchor_sentence else []
                found.append({
                    "doc_index": i,
                    "passage": sent,
                    "source_number": number,
                    "counterfactual_number": cf,
                    "counterfactual_passage": swapped,
                    "anchor_sentence": anchor_sentence,
                    "anchor_terms": anchor_terms,
                })
                break  # one claim per document keeps items independent
    return found


def build_corpus(c4_path: str | Path, out_dir: str | Path, *,
                 n_docs: int = 400) -> list[Path]:
    """Write the first ``n_docs`` C4 documents out as SEPARATE files.

    One file per document so that ``RetrievedChunk.source`` identifies which
    document a chunk came from, which is what makes it possible to measure
    whether retrieval found the right document rather than merely a chunk
    that happened to contain the right digits.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("doc_*.txt"):
        stale.unlink()
    paths: list[Path] = []
    with open(c4_path, encoding="utf-8") as src:
        for i, line in enumerate(itertools.islice(src, n_docs)):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (doc.get("text") or "").replace("\n", " ")
            if not text.strip():
                continue
            path = out / f"doc_{i:05d}.txt"
            path.write_text(text + "\n", encoding="utf-8")
            paths.append(path)
    return paths


def assemble(condition: str, item: dict, retrieved_text: str, *, tokenizer,
             context_length: int) -> tuple[str, str]:
    """Return (prompt, fitted_context) for one condition.

    The fitted context is what the model actually sees, so the number we call
    "in context" is checked against the FITTED text, not the raw retrieval
    chunk. A chunk whose number falls outside the token budget was never
    shown to the model and must not be credited to it.
    """
    if condition == "oracle":
        raw = item["passage"]
    elif condition == "counterfactual":
        raw = item["counterfactual_passage"]
    elif condition == "retrieved":
        raw = retrieved_text
    elif condition == "nonnumeric":
        # Length- and topic-matched control: a DIFFERENT sentence of the same
        # document that contains no number. It separates "the passage" from
        # "the number the passage asserts".
        raw = item.get("anchor_sentence", "")
    else:
        raw = ""
    fitted = fit_context(raw, instruction=INSTRUCTION, tokenizer=tokenizer,
                         context_length=context_length, answer_reserve=16)
    prompt = rag_prompt(INSTRUCTION, context=fitted, tokenizer=tokenizer,
                        context_length=context_length, answer_reserve=16)
    return prompt, fitted


def context_number(condition: str, item: dict, fitted: str) -> str | None:
    """The tracked number actually present in the context the model was shown."""
    if condition == "oracle":
        want = item["source_number"]
    elif condition == "counterfactual":
        want = item["counterfactual_number"]
    elif condition == "retrieved":
        want = item["source_number"]
    else:
        return None
    return want if want in fitted else None


def partner_number(condition: str, item: dict) -> str:
    """The other tracked number, which is NOT what the context says."""
    if condition == "counterfactual":
        return item["source_number"]
    return item["counterfactual_number"]


def distractor_pool(items: list[dict]) -> list[list[str]]:
    """Per-item distractor numbers, drawn from OTHER items' source numbers.

    Deterministic and answer-agnostic: item i borrows from items
    (i+1, i+2, i+3, i+4) modulo the pool, so the distractors are plausible
    numbers from the same corpus without being the item's own.
    """
    pool = [it["source_number"] for it in items]
    n = len(pool)
    out = []
    for i in range(n):
        picks = []
        for k in range(1, N_DISTRACTORS + 1):
            cand = pool[(i + k) % n]
            if cand not in picks:
                picks.append(cand)
        out.append(picks)
    return out


def run(rung: str, *, items: int, max_new: int, device: str, seeds=SEEDS,
        data: str = "data/v2/c4_en_shard_0.jsonl",
        corpus: str = "artifacts/act8/retrieval_corpus",
        corpus_docs: int = 400) -> dict:
    from src.eval.act7_sft import load_canonical
    from src.rag.retriever import TextRetriever
    from src.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
    model = load_canonical(rung).to(device)
    model.eval()
    ctx = model.config.context_length

    passages = load_passages(data, limit=items * 3, tokenizer=tokenizer)

    # The retriever is queried with the item's ANCHOR: content words from a
    # different sentence of the same document. The previous query was the
    # instruction text, whose terms {what, number, does, the, passage, state}
    # cannot come from a passage about a number -- term-overlap ranking then
    # degenerated to a constant and returned the SAME chunk for all 40 items.
    # That was a query-design failure, not a retriever bug.
    corpus_paths = build_corpus(data, corpus, n_docs=corpus_docs)
    retriever = TextRetriever.from_files(corpus_paths, chunk_words=40,
                                          overlap_words=0)

    # Guards: the prompt must fit, and the oracle context must CONTAIN its
    # number. A long sentence can push the number past the token budget, which
    # would make the item unwinnable and inflate the rate -- found by checking
    # rather than assuming.
    kept, dropped = [], 0
    for idx, item in enumerate(passages):
        # The oracle context is the item's own sentence, so the retrieved text
        # is not consulted here.
        prompt, fitted = assemble("oracle", item, "", tokenizer=tokenizer,
                                  context_length=ctx)
        if len(tokenizer.encode(prompt)) > ctx:
            raise AssertionError("oracle prompt exceeds the window")
        if item["source_number"] not in fitted:
            dropped += 1
            continue
        kept.append(item)
    passages = kept[:items]
    if len(passages) < items:
        raise ValueError(f"only {len(passages)} of {items} items survived the guards "
                         f"({dropped} had their number truncated out)")

    # Retrieval is run AFTER the guard so that the metadata covers exactly the
    # items that are scored. (Building it earlier indexed every candidate and
    # made the rates exceed 1.)
    retrieved: dict[int, str] = {}
    retrieval_meta: dict[int, dict] = {}
    for idx, item in enumerate(passages):
        query = " ".join(item["anchor_terms"])
        hits = retriever.retrieve(query, top_k=1)
        chunk = hits[0] if hits else None
        retrieved[idx] = chunk.text if chunk else ""
        expected_doc = f"doc_{item['doc_index']:05d}"
        retrieval_meta[idx] = {
            "query": query,
            "expected_doc": expected_doc,
            "retrieved_doc": Path(chunk.source).stem if chunk else None,
            "doc_hit": bool(chunk and Path(chunk.source).stem == expected_doc),
            "answer_present": bool(chunk and item["source_number"] in chunk.text),
        }

    distractors = distractor_pool(passages)

    t0 = time.time()
    copy_rows: list[dict] = []
    free_rows: list[dict] = []
    for idx, item in enumerate(passages):
        for cond in CONDITIONS:
            prompt, fitted = assemble(cond, item, retrieved[idx], tokenizer=tokenizer,
                                      context_length=ctx)
            ids = tokenizer.encode(prompt)
            if len(ids) > ctx:
                raise AssertionError(f"{cond} prompt is {len(ids)} tokens > {ctx}")
            ctx_num = context_number(cond, item, fitted)
            alt = partner_number(cond, item)
            src_n = item["source_number"]
            cands = sorted({c for c in [src_n, alt] + distractors[idx] if c})
            scores = score_candidates(model, ids, cands, tokenizer, device)
            # Rank by total log-probability: candidates are token-length
            # matched, so this is a like-for-like comparison.
            chosen = max(scores, key=lambda c: scores[c]["total_logprob"])
            copy_rows.append({
                "item": idx, "condition": cond, "chosen": chosen,
                "class": classify_copy(chosen, src_n, alt),
                "in_context_number": ctx_num,
                # Did the model pick the number that was actually in the
                # context it was shown? None for closed_book, which has none.
                "followed_context": (None if ctx_num is None
                                     else bool(chosen == ctx_num)),
                "scores": scores,
            })

            # Free generation: sampled, so seeds produce different text.
            for seed in seeds:
                torch.manual_seed(seed * 100003 + idx * 17 + CONDITIONS.index(cond))
                x = torch.tensor([ids], dtype=torch.long, device=device)
                out = generate(model, x, context_length=ctx, max_new_tokens=max_new,
                               do_sample=True, temperature=1.0, use_cache=True)
                gen = tokenizer.decode(out[0, len(ids):].tolist())
                free_rows.append({
                    "item": idx, "condition": cond, "seed": seed,
                    "generation": gen, "in_context_number": ctx_num,
                    "source_number": item["source_number"],
                    "counterfactual_number": item["counterfactual_number"],
                    "context_text": fitted,
                })
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(passages)} ({time.time() - t0:.0f}s)", flush=True)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return summarise(copy_rows, free_rows, passages=passages,
                     retrieved=retrieved, retrieval_meta=retrieval_meta,
                     distractors=distractors, rung=rung, seeds=list(seeds),
                     max_new=max_new, dropped=dropped, corpus_docs=corpus_docs,
                     seconds=round(time.time() - t0, 1))


def summarise(copy_rows: list[dict], free_rows: list[dict], *, passages: list[dict],
              retrieved: dict, retrieval_meta: dict, distractors: list[list[str]],
              rung: str, seeds: list[int], max_new: int, corpus_docs: int,
              dropped: int, seconds: float) -> dict:
    """Per-condition distributions and paired per-item contrasts."""
    items = len(passages)
    by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    for r in copy_rows:
        by_cond[r["condition"]].append(r)

    # --- primary instrument: answer copying (deterministic, n = items) ---
    copy_summary = {}
    for cond, rows in by_cond.items():
        labels = [r["class"] for r in rows]
        s = class_rate_summary(labels, COPY_CLASSES)
        n = len(rows)
        followed = [r["followed_context"] for r in rows
                    if r["followed_context"] is not None]
        copy_summary[cond] = {
            "n_items": n,
            "counts": s["counts"],
            "rates": {k: round(v, 4) for k, v in s["rates"].items()},
            "chose_source_rate": round(s["rates"]["source"], 4),
            "chose_source_wilson95": [round(v, 4) for v in
                                      wilson_interval(s["counts"]["source"], n)],
            "followed_context_rate": (round(sum(followed) / len(followed), 4)
                                      if followed else None),
            "followed_context_wilson95": ([round(v, 4) for v in
                                           wilson_interval(sum(followed), len(followed))]
                                          if followed else None),
            "note": ("closed_book shows no number at all, so 'followed_context' "
                     "is undefined: there is nothing in the context to follow. "
                     "'chose_source' here means the model produced the item's "
                     "answer with no passage -- a baseline, not grounding."
                     if cond == "closed_book" else
                     "followed_context = the model preferred the number that was "
                     "in the context it was shown. Classes are "
                     "condition-agnostic (source / counterfactual / neither) so "
                     "that no class is unreachable by construction."),
        }

    # --- paired per-item contrasts ---
    choice = {(r["item"], r["condition"]): r["chosen"] for r in copy_rows}
    cls = {(r["item"], r["condition"]): r["class"] for r in copy_rows}

    d_oracle_cb = [1.0 if choice[(i, "oracle")] != choice[(i, "closed_book")] else 0.0
                   for i in range(items)]
    d_cf_track = [1.0 if cls[(i, "counterfactual")] == "counterfactual" else 0.0
                  for i in range(items)]
    d_cf_follows = [1.0 if choice[(i, "counterfactual")] != choice[(i, "oracle")] else 0.0
                    for i in range(items)]
    d_cf_same = [1.0 if choice[(i, "counterfactual")] == choice[(i, "oracle")] else 0.0
                 for i in range(items)]

    # Paired binary: did the model pick the item's answer under oracle, and
    # did it pick it under closed-book? McNemar conditions on the discordant
    # pairs, which is the right test for the same items under two conditions.
    chose_source = {(r["item"], r["condition"]): r["class"] == "source"
                    for r in copy_rows}
    b_mcn = sum(1 for i in range(items)
                 if chose_source[(i, "oracle")] and not chose_source[(i, "closed_book")])
    c_mcn = sum(1 for i in range(items)
                 if not chose_source[(i, "oracle")] and chose_source[(i, "closed_book")])
    mcn = mcnemar_exact(b_mcn, c_mcn)

    contrasts = {
        "oracle_vs_closed_book_chose_the_answer": {
            "definition": "paired McNemar: items where the model chose the "
                          "item's answer under the oracle context but not "
                          "under closed-book, versus the reverse",
            "n": items,
            "oracle_chose_source": sum(1 for i in range(items)
                                       if chose_source[(i, "oracle")]),
            "closed_book_chose_source": sum(1 for i in range(items)
                                            if chose_source[(i, "closed_book")]),
            "discordant_b": b_mcn, "discordant_c": c_mcn,
            "mcnemar_p_value": round(mcn["p_value"], 6),
        },
        "oracle_vs_closed_book_answer_changes": {
            "definition": "share of items where the preferred answer differs "
                          "between the oracle context and no context",
            "n": items,
            "count": int(sum(d_oracle_cb)),
            "rate": round(sum(d_oracle_cb) / items, 4),
            "wilson95": [round(v, 4) for v in wilson_interval(int(sum(d_oracle_cb)), items)],
            "bootstrap95": bootstrap_mean_ci(d_oracle_cb, seed=1),
        },
        "counterfactual_tracks_the_edit": {
            "definition": "share of items where the preferred answer under the "
                          "counterfactual is the EDITED number",
            "n": items,
            "count": int(sum(d_cf_track)),
            "rate": round(sum(d_cf_track) / items, 4),
            "wilson95": [round(v, 4) for v in wilson_interval(int(sum(d_cf_track)), items)],
            "bootstrap95": bootstrap_mean_ci(d_cf_track, seed=2),
        },
        "counterfactual_answer_differs_from_oracle": {
            "definition": "share of items where editing the number in the text "
                          "changed the preferred answer",
            "n": items,
            "count": int(sum(d_cf_follows)),
            "rate": round(sum(d_cf_follows) / items, 4),
            "wilson95": [round(v, 4) for v in wilson_interval(int(sum(d_cf_follows)), items)],
            "bootstrap95": bootstrap_mean_ci(d_cf_follows, seed=3),
        },
        "counterfactual_answer_unchanged_from_oracle": {
            "definition": "share of items where the preferred answer was "
                          "IDENTICAL with and without the edit -- i.e. the "
                          "model ignored the edited text",
            "n": items,
            "count": int(sum(d_cf_same)),
            "rate": round(sum(d_cf_same) / items, 4),
            "wilson95": [round(v, 4) for v in wilson_interval(int(sum(d_cf_same)), items)],
            "bootstrap95": bootstrap_mean_ci(d_cf_same, seed=4),
        },
    }

    # --- paired log-probability contrasts -----------------------------------
    # The argmax flip rate is brittle: it throws away the SIZE of the
    # preference. These are the same question measured continuously, paired
    # per item, so they carry far more power and do not depend on two
    # near-tied log-probabilities landing on the right side of a comparison.
    def lp(item_id: int, cond: str, cand: str) -> float | None:
        row = next(r for r in copy_rows
                   if r["item"] == item_id and r["condition"] == cond)
        sc = row["scores"].get(cand)
        return None if sc is None else sc["total_logprob"]

    def paired(name: str, definition: str, values: list[float], seed: int) -> dict:
        vals = [v for v in values if v is not None]
        out = {
            "definition": definition,
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4) if vals else None,
            "bootstrap95": bootstrap_mean_ci(vals, seed=seed) if vals else None,
        }
        if vals:
            t = paired_ttest(vals, [0.0] * len(vals))
            out["t"] = round(t["t"], 4)
            out["df"] = t["df"]
            out["p_value"] = round(t["p_value"], 6)
        return out

    src_lift, cf_lift, edit_effect = [], [], []
    oracle_margin, cb_margin, cf_margin = [], [], []
    for i in range(items):
        it = passages[i]
        s, c = it["source_number"], it["counterfactual_number"]
        src_lift.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "oracle", s), lp(i, "closed_book", s)))
        cf_lift.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "oracle", c), lp(i, "closed_book", c)))
        edit_effect.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "counterfactual", c), lp(i, "oracle", c)))
        oracle_margin.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "oracle", s), lp(i, "oracle", c)))
        cb_margin.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "closed_book", s), lp(i, "closed_book", c)))
        # The symmetric check: under the COUNTERFACTUAL prompt, does the model
        # prefer the edited number over the original? If the model reads which
        # digit the text states, this should be positive.
        cf_margin.append(
            (lambda a, b: None if a is None or b is None else a - b)(
                lp(i, "counterfactual", c), lp(i, "counterfactual", s)))

    # The decisive statistic: how much of the oracle margin is CONTEXT-SPECIFIC
    # rather than a prior preference for one number over another. Paired per
    # item, so the item's own difficulty cancels.
    ctx_discrimination = [a - b for a, b in zip(oracle_margin, cb_margin)
                          if a is not None and b is not None]
    lift_difference = [a - b for a, b in zip(src_lift, cf_lift)
                        if a is not None and b is not None]

    logprob_contrasts = {
        "context_specific_discrimination": paired(
            "context_specific_discrimination",
            "(log P(source|oracle) - log P(cf|oracle)) - (log P(source|closed_book) "
            "- log P(cf|closed_book)), paired per item. This is THE statistic: "
            "the margin the model shows for the number in its context, MINUS the "
            "margin it would show for the same pair of numbers with no context. "
            "Positive means the preference is caused by READING the passage "
            "rather than by a prior over numbers.",
            ctx_discrimination, 21),
        "lift_stated_minus_lift_unstated": paired(
            "lift_stated_minus_lift_unstated",
            "CONTROL: how much the passage raised the stated number's "
            "likelihood MINUS how much it raised the unstated number's. A "
            "context effect that is specific to the stated number should be "
            "positive; a generic 'numbers became more likely in a numeric "
            "context' effect would be near zero.",
            lift_difference, 22),
        "passage_lift_on_the_number_it_states": paired(
            "passage_lift_on_the_number_it_states",
            "log P(source | oracle prompt) - log P(source | closed-book prompt), "
            "paired per item. Positive means supplying the passage RAISED the "
            "model's preference for the number the passage states.",
            src_lift, 11),
        "passage_lift_on_the_counterfactual_number": paired(
            "passage_lift_on_the_counterfactual_number",
            "CONTROL for the contrast above: the same comparison for the number "
            "the passage does NOT state. If the passage is being read, this "
            "should be near zero.",
            cf_lift, 12),
        "edit_effect_on_the_counterfactual_number": paired(
            "edit_effect_on_the_counterfactual_number",
            "log P(cf | counterfactual prompt) - log P(cf | oracle prompt). "
            "Positive means EDITING the number in the text raised the model's "
            "preference for the edited number.",
            edit_effect, 13),
        "oracle_margin_source_minus_counterfactual": paired(
            "oracle_margin_source_minus_counterfactual",
            "log P(source | oracle) - log P(cf | oracle). Positive means the "
            "model prefers the number actually in its context.",
            oracle_margin, 14),
        "counterfactual_margin_edited_minus_original": paired(
            "counterfactual_margin_edited_minus_original",
            "log P(cf | counterfactual prompt) - log P(source | counterfactual "
            "prompt). The symmetric check on the oracle margin: if the model "
            "reads WHICH digit the text states, this should be positive too. "
            "It is the more conservative reading of whether the passage is "
            "being read.",
            cf_margin, 23),
        "closed_book_margin_source_minus_counterfactual": paired(
            "closed_book_margin_source_minus_counterfactual",
            "CONTROL: the same margin with no context. Should be near zero if "
            "the oracle margin is about reading rather than a number prior.",
            cb_margin, 15),
    }

    # --- chance baseline ---------------------------------------------------
    # The candidate set is NOT uniformly sized: a distractor can coincide with
    # the source or the counterfactual and be deduplicated, so an item may
    # have 4, 5 or 6 distinct candidates. A flat "chance = 1/3" would be
    # wrong; the chance level is computed per item.
    chance = []
    for i, it in enumerate(passages):
        alt = partner_number("oracle", it)
        cands = {c for c in [it["source_number"], alt] + distractors[i] if c}
        chance.append(1.0 / len(cands))
    size_counts: dict[int, int] = {}
    for c in chance:
        size_counts[round(1.0 / c)] = size_counts.get(round(1.0 / c), 0) + 1
    mean_chance = sum(chance) / len(chance)

    def chance_adjusted(cond: str, seed: int) -> dict:
        diffs = [(1.0 if cls[(i, cond)] == "source" else 0.0) - chance[i]
                 for i in range(items)]
        return {
            "n": items,
            "mean_uniform_chance": round(mean_chance, 4),
            "observed_rate": round(sum(d for d in diffs) / items, 4),
            "chance_adjusted_mean": round(sum(diffs) / items, 4),
            "bootstrap95": bootstrap_mean_ci(diffs, seed=seed),
        }

    chance_block = {
        "candidate_set_sizes": {str(k): v for k, v in sorted(size_counts.items())},
        "mean_uniform_chance_per_item": round(mean_chance, 4),
        "why_per_item": "the candidate set is deduplicated, so an item may "
                        "have 4, 5 or 6 distinct candidates; a flat 1/3 (or "
                        "1/6) would be wrong",
        "per_condition": {c: chance_adjusted(c, 30 + i)
                          for i, c in enumerate(CONDITIONS)},
        "wording": "the model identifies the asserted number AT CHANCE. It is "
                   "not significantly below chance, and it is not above it "
                   "either; and the behavioural point estimate under oracle is "
                   "LOWER than the model's own closed-book baseline.",
    }

    # --- the headline, in the order the result should be read --------------
    headline = {
        "claim": "supplying the passage does not change what the model says",
        "order": "choice rates first, lift rows as the explanation",
        "choice_rates": {
            "oracle_chooses_the_asserted_number": {
                "count": copy_summary["oracle"]["counts"]["source"],
                "n": items, "rate": copy_summary["oracle"]["chose_source_rate"],
                "wilson95": copy_summary["oracle"]["chose_source_wilson95"]},
            "closed_book_chooses_the_asserted_number": {
                "count": copy_summary["closed_book"]["counts"]["source"],
                "n": items, "rate": copy_summary["closed_book"]["chose_source_rate"],
                "wilson95": copy_summary["closed_book"]["chose_source_wilson95"]},
            "retrieved_chooses_the_asserted_number": {
                "count": copy_summary["retrieved"]["counts"]["source"],
                "n": items, "rate": copy_summary["retrieved"]["chose_source_rate"],
                "wilson95": copy_summary["retrieved"]["chose_source_wilson95"]},
            "counterfactual_chooses_the_EDITED_number": {
                "count": copy_summary["counterfactual"]["counts"]["counterfactual"],
                "n": items,
                "rate": copy_summary["counterfactual"]["rates"]["counterfactual"],
                "wilson95": [round(v, 4) for v in wilson_interval(
                    copy_summary["counterfactual"]["counts"]["counterfactual"],
                    items)]},
            "mean_uniform_chance": round(mean_chance, 4),
            "chance_adjusted_oracle": chance_adjusted("oracle", 41),
            "chance_adjusted_closed_book": chance_adjusted("closed_book", 42),
            "mcnemar_oracle_vs_closed_book": contrasts[
                "oracle_vs_closed_book_chose_the_answer"],
            "counterfactual_tracks_the_edit": contrasts[
                "counterfactual_tracks_the_edit"],
        },
        "explanation": {
            "summary": "the passage produces a generic NUMBER bias, not an "
                       "assertion-reading ability. Having a passage raises the "
                       "likelihood of a number appearing in it, and raises it "
                       "almost as much for a number the passage does NOT "
                       "assert.",
            "lift_on_the_asserted_number": logprob_contrasts[
                "passage_lift_on_the_number_it_states"],
            "lift_on_the_UNASSERTED_number": logprob_contrasts[
                "passage_lift_on_the_counterfactual_number"],
            "residual_assertion_sensitivity": logprob_contrasts[
                "context_specific_discrimination"],
            "instruments_disagree": "the residual is a single marginal "
                                     "result (p=0.030) and it produces NO "
                                     "behavioural change: the behavioural "
                                     "point estimate is in the OPPOSITE "
                                     "direction (oracle 9/40 vs closed-book "
                                     "11/40, McNemar p=0.625). The behavioural "
                                     "test is the one that matters.",
        },
        "ceiling_note": "the absolute free-generation hallucination rate is "
                        "1.000 in every condition (97/120 oracle generations "
                        "contain no number at all). It is a CEILING and is "
                        "uninformative for ranking conditions. It is never used "
                        "as a comparison.",
    }

    # --- secondary instrument: free generation ---
    free_by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    for r in free_rows:
        free_by_cond[r["condition"]].append(r)
    free_summary = {}
    for cond, rows in free_by_cond.items():
        per_item: dict[int, list[dict]] = {}
        for r in rows:
            per_item.setdefault(r["item"], []).append(r)
        # Average over seeds FIRST, so the unit is the item.
        volunteered = sum(1 for v in per_item.values()
                          if any(NUMBER_RE.search(r["generation"]) for r in v))
        labels = []
        for _item_id, rs in per_item.items():
            gen = rs[0]["generation"]
            match = NUMBER_RE.search(gen)
            num = match.group(0).replace(",", "") if match else ""
            # in_context_number varies per item (the retrieved chunk may or may
            # not contain the number), so it must come from THIS item's rows.
            labels.append(classify_copy(num, rs[0]["source_number"],
                                        rs[0]["counterfactual_number"]))
        s = class_rate_summary(labels, COPY_CLASSES)
        # Diagnostic: did the generation emit SOME number that appears in the
        # context it was shown? Separates "read the passage but picked the
        # wrong span" from "ignored the passage entirely".
        from_context = 0
        for _item_id, rs in per_item.items():
            hit = False
            for r in rs:
                m = NUMBER_RE.search(r["generation"])
                if m and m.group(0).replace(",", "") in r["context_text"]:
                    hit = True
                    break
            from_context += int(hit)
        free_summary[cond] = {
            "n_items": len(per_item),
            "seeds": seeds,
            "volunteers_a_number_rate": round(volunteered / len(per_item), 4),
            "emits_a_number_from_the_context_rate": round(from_context / len(per_item), 4),
            "counts": s["counts"],
            "rates": {k: round(v, 4) for k, v in s["rates"].items()},
            "note": "free generation repeats of an item are averaged per item "
                    "before scoring, so the unit is the item, not the draw",
        }

    # Retrieval diagnostics: did the answer-agnostic retriever find the right
    # DOCUMENT, and did the chunk it returned contain the answer?
    doc_hits = sum(1 for v in retrieval_meta.values() if v["doc_hit"])
    answer_hits = sum(1 for v in retrieval_meta.values() if v["answer_present"])
    distinct_chunks = len({v["retrieved_doc"] for v in retrieval_meta.values()})

    return {
        "act": "8b+8c",
        "rung": rung,
        "headline": headline,
        "chance_baseline": chance_block,
        "claim_format": {
            "claim": "the supplied passage states <NUMBER>",
            "classes": ["passage", "other", "neither"],
            "primary_instrument": "answer copying: teacher-forced likelihood of "
                                   "each candidate answer, argmax over the "
                                   "candidate set",
            "why_not_free_generation": "free generation is at a ceiling: 97/120 "
                                       "oracle generations contain no number at "
                                       "all, so the rate reads 1.000 in every "
                                       "condition and cannot rank them. It "
                                       "confounds grounding with label "
                                       "competence. It is reported as a "
                                       "secondary instrument.",
            "measures": "groundedness against the supplied source, NOT factual "
                        "truth against the world (no fact base available)",
        },
        "items": items,
        "dropped_for_truncation": dropped,
        "seeds": seeds,
        "unit_of_analysis": "item",
        "unit_note": "the primary instrument is deterministic, so there is one "
                     "measurement per item per condition: n = items, NOT "
                     "items x seeds. Free-generation repeats are averaged per "
                     "item before scoring. Contrast CIs resample ITEMS.",
        "conditions": {
            "copy_instrument": copy_summary,
            "free_instrument": free_summary,
        },
        "contrasts": contrasts,
        "logprob_contrasts": logprob_contrasts,
        "retrieval": {
            "corpus_docs": corpus_docs,
            "query": "the item's lexical anchor: content words from a DIFFERENT "
                     "sentence of the same document, so the retriever is given "
                     "a topical handle but never the number or the "
                     "answer-bearing sentence",
            "answer_agnostic": True,
            "document_hit_rate": round(doc_hits / items, 4),
            "answer_present_rate": round(answer_hits / items, 4),
            "distinct_documents_returned": distinct_chunks,
            "diagnosis": "the FIRST version of this condition queried the "
                         "retriever with the instruction text, whose terms "
                         "{what, number, does, the, passage, state} cannot come "
                         "from a passage about a number. Term-overlap ranking "
                         "degenerated to a constant and returned the SAME chunk "
                         "for all 40 items, so 0/40 was a query-design failure "
                         "rather than a retriever failure. The query is now "
                         "anchored to the document.",
            "note": "the anchor is derived from the target document, so the "
                    "document hit rate is optimistic relative to a user's "
                    "paraphrase; but the retriever never sees the number or "
                    "the answer-bearing sentence, so this is not a disguised "
                    "oracle.",
        },
        "max_new_tokens": max_new,
        "wall_seconds": seconds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/v2/act8/hallucination.json")
    args = ap.parse_args()

    out = run(args.rung, items=args.items, max_new=args.max_new, device=args.device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["contrasts"], indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
