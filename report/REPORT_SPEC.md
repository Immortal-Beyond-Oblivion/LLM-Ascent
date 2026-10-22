# REPORT_SPEC.md — the source of truth for the report-authoring ticket

This file holds the ticket brief verbatim. It is the source of truth for this
ticket, not recollection of it. Re-read it after any context compaction or long
gap. Items are ticked off as they complete. Any change to the spec text is
noted in the commit message and in the report-back.

**Status of items as of this writing (see the tick list at the end):**

- A1 sequencing — DONE (working tree was clean; HEAD at start was `42407c4`).
- A2 spec persistence — DONE (this file).
- A3 stale line — SUPERSEDED, ignored.
- A4/A5 Act 8 provisional / open item — SUPERSEDED by the context update: the
  Act 8 revision landed at commit `42407c4`, so Act 8 is written as FINAL from
  the current JSON. The retrieved condition is NOT written as an open item.
- A6 Acts 9 and 10 — placeholders only.
- A7 Act 10 overlap — the tracked artefacts under `results/v2/act10/` do not
  exist yet; the prose is written from the ledger and the pending artefact is
  noted.
- A8 verification — DONE via `tools/check_tex.py` (no pdflatex on this machine).

--- BEGIN SPEC (WRITE THIS VERBATIM INTO report/REPORT_SPEC.md) ---

A1. SEQUENCING. This ticket does NOT start, measure, or interfere with the
    in-flight Act 8/9/10 ticket. If that ticket is running, let it finish and
    commit first, then begin. Do not run both against the repo at the same
    time; if the Act 9/10 ticket is MPS-bound and this one is CPU-bound
    authoring, run this one only after the other is committed. Your working
    tree must be clean when you start, and you must report what the HEAD
    commit was when you began.

A2. SPEC PERSISTENCE - DO THIS FIRST. Write the remainder of this prompt
    VERBATIM into report/REPORT_SPEC.md before you do any work. Then work
    from that file, re-reading it after any context compaction or long gap.
    The file is the source of truth for this ticket, not your recollection of
    it. Do not edit the spec except to tick off completed items, and if you
    change it, say so in the commit message.

A3. STALE LINE - OVERRIDDEN. The spec says "Do NOT start Act 9 until this
    ticket is committed". That is superseded. Act 9 and Act 10 are already
    in flight in this session. Ignore that sentence. This ticket writes the
    REPORT only.

A4. ACT 8 IS PROVISIONAL - MARK IT. The Act 8 numbers in the spec are current
    as of commit 4405f18, but a separate ticket is revising them right now:
      (i) the retrieved condition had retrieval success 0/40 and is being
          diagnosed - it may be fixed and re-measured, or retired with a
          stated reason;
      (ii) the per-item chance baseline is being corrected, because 10 of 120
          candidates were dropped for truncation so items that lost a candidate
          have chance 0.5, not 0.333 - the flat "chance = 1/3" framing in the
          spec is known to be imprecise;
      (iii) the "unstated number" construction is being clarified (present in
          the passage as a distractor, or absent entirely), which changes the
          wording of the bag-of-numbers claim.
    Therefore: write Act 8 with the current numbers, but put a visible LaTeX comment next to the Act 8 table AND a one-line note in the section header saying the retrieved condition and the chance baseline are under revision. Do not present the Act 8 claims-table row as final - mark its status "pending 8c revision". When the Act 8 revision lands, the Act 8 section is regenerated, not hand-patched.

A5. THE RETRIEVED CONDITION IS NOT YET A RESULT. If you write Act 8 before
    the diagnosis lands, write the retrieved condition as an OPEN ITEM with the
    reason it is open. Do NOT write it as a measured null, and do NOT write it
    as a finding either way. A broken retriever presented as a measured null
    would be the worst error in this project's ledger.

A6. ACTS 9 AND 10 ARE PLACEHOLDERS ONLY. Write the full section skeletons
    with the subsection structure, a visible \textbf{[PENDING - not yet
    measured]} where results go, and the subsection headings I would want
    filled. The document must compile and read as coherent WITH the gaps
    visible. I will supply the content.

A7. ACT 10 OVERLAP - DO NOT DUPLICATE. The Act 10 ticket is separately
    producing a machine-readable claims-audit table and a measurement-failure
    taxonomy as TRACKED ARTEFACTS under results/v2/act10/. In the paper, write
    the prose for Threats to Validity and the claims table, and REFERENCE those
    artefacts as the source of truth rather than re-deriving them by hand. If
    the artefacts exist when you write, read them and use their content. If
    they do not exist yet, write the prose from the ledger and note that the
    tracked artefact is pending.

A8. VERIFICATION IS NOT OPTIONAL. The owner compiles on Overleaf and there is
    no pdflatex here, so the checks in section 10 are the ONLY correctness
    gate on the .tex before the owner ever sees it. Do them and report exactly
    what you checked and what you found. A commit that introduces a broken
    \ref is worse than no commit, because the owner finds it at the worst
    moment. If you are unsure whether a construct compiles, prefer the boring
    one that certainly does.

--- 0. AUTHORITY OF NUMBERS ===

The numbers I quote below are for ORIENTATION ONLY. The JSON files under
results/ are the source of truth. Where this prompt and a JSON disagree, THE
JSON WINS and you tell me in your report-back. Never transcribe a number into
the .tex from this prompt without checking it against its source file. If a
number is missing from the JSON, leave a clearly marked placeholder and report it - do
not estimate.

--- 1. PRE-AUDIT BEFORE WRITING (do this first, report the outcome) ===

  a) Read report/final_report.tex in full. It is frozen at v1 (Oct 14) and
     covers none of Acts 0, 4, 5, 6, 7, 8. Learn and PRESERVE its conventions:
     preamble, packages, macro definitions, table style, figure macros, section
     hierarchy, caption style, bibliography style. Do not fight them. You may
     ADD to the preamble (booktabs, siunitx, amsmath, etc.) if a table needs it,
     but do not restructure what works.
  b) Enumerate report/figures/ - every .png and .pdf pair. Report the inventory
     and which act each belongs to.
  c) Read docs/implementation.md in full. It is the results ledger and it holds
     the caveats, the bug stories, and the do-not-assert list. The ledger and the
     .tex must agree. Where the ledger is more precise than this prompt, the
     ledger wins.
  d) Confirm which results live where. Note the layout: act0-ngram/, act1-rnn/,
     act2-ablations/, act3-tokenizers/, tokenizer-comparison/ are directly under
     results/, while act4-scaling/ and later are under results/v2/. Record the
     exact path for every figure and table in the MANIFEST (section 9).

--- 2. FIGURE GENERATION - GENERATE FROM JSON, NEVER BY HAND ===

Convention already established: figures come from a committed script that reads
ONLY the tracked results JSON. Follow the existing pattern in
src/eval/figures.py and src/eval/act6_figures.py. For any act that has results
but no figure, write src/eval/actN_figures.py, read only from results/, and
commit it.

Rules:
  - every figure: PNG (for Overleaf compatibility) AND vector PDF
  - deterministic: no randomness, or seeded explicitly
  - no absolute paths; resolve paths relative to the repo root
  - each figure must have a caption that states WHAT IS PLOTTED, the n, and the
    interval type
  - figures that carry a caveat must SAY IT IN THE CAPTION, not only in prose
    (Act 2 and Act 3 figures: n=1, exploratory, no error bars; Act 5a: the
    ctx-1024 microbenchmark caveat)

Figures I expect to need generating: Act 0 (smoothing x order), Act 1
(RNN/LSTM/GRU with n=3), Act 2 (ablation bars, n=1 - no error bars), Act 3
(tokenizer comparison, 2 controls, n=1), Act 7 (SFT scaling curve, PEFT matched
budget, LoRA rank sweep, ICL copy-vs-compute), Act 8 (hallucination by
condition, contrast distribution). Act 4's two figures already exist; Act 5 and
Act 6 figures exist per the ledger. Verify each existing figure is current with
its JSON - if the JSON moved on, regenerate.

--- 3. STRUCTURE ===

  \title{...}
  \begin{abstract}  <- 200-250 words. The thesis, the scale of the models, the
     headline findings including the NEGATIVE ones, and the one methodological
     lesson you want remembered.
  1  Introduction and research questions
  2  Method
       2.1 Model and training setup
       2.2 Data and tokenization
       2.3 Measurement protocol   <- seeds, n>=3, Wilson intervals, item as the
            unit of analysis, pre-registration, what makes a result exploratory
       2.4 A note on what a 6.3M model can and cannot do
  3  Act 0 - n-gram models and smoothing
  4  Act 1 - recurrent baselines
  5  Act 2 - Transformer component ablations
  6  Act 3 - tokenization
  7  Act 4 - scaling laws   <- CENTREPIECE
  8  Act 5 - inference efficiency (5a KV cache, 5b MQA/GQA, 5c speculative)
  9  Act 6 - decoding
  10 Act 7 - instruction tuning and PEFT
  11 Act 8 - grounding, hallucination, and tool use
  12 Act 9 - interpretability                [PLACEHOLDER - not yet measured]
  13 Act 10 - evaluation methodology and ethics [PLACEHOLDER]
  14 Threats to validity and known limitations
  15 Reproducibility
  References

--- 4. SECTION SHAPE - MANDATORY, EVERY ACT ---

Every act section ends with all three of:
  (i)   a TABLE with error bars (or an explicit no-error-bars note),
  (ii)  one or two figures,
  (iii) a stated conclusion that says what the measurement supports AND what it
        does not.

A negative result gets the same treatment as a positive one: table, figure,
conclusion. Never a section that reports a number and moves on.

--- 5. PER-ACT CONTENT SPEC ---

--- Act 0 (results/act0-ngram/) ---
36 configs: 3 orders x 4 smoothings x 3 seeds. Include the 80/10/10 style
table: order x smoothing, perplexity, mean +/- std over the 3 seeds. Wilson is
for accuracy; for perplexity use mean +/- std and SAY which is which. Embedding
work (SVD, word2vec CBOW/skip-gram, GloVe, fastText) belongs in a subsection -
if it is measured, report it; if only built, say "built, not measured" and do
not present it as a result. Check the ledger for which it is.

--- Act 1 (results/act1-rnn/) ---
RNN / LSTM / GRU, 3 seeds each. Report the gradient demonstration (vanishing vs
exploding) as a figure with the measurement that produced it, not an assertion.

--- Act 2 (results/act2-ablations/) ---
20 runs, SEED 42 ONLY. This is the single most caveated section in the report.
  - Every table and figure carries a visible note: "single seed (n=1);
    exploratory; no error bars; no significance testing."
  - No +/- columns. No p-values. No words like "significant" or "better" in a
    comparative sense without "suggests".
  - In the prose, state once, prominently, why: at n=1 these rank
    configurations, they do not establish effects.

--- Act 3 (results/act3-tokenizers/) ---
5 tokenizers x 2 controls, SEED 42 ONLY - same n=1 treatment as Act 2.
The equal-character-span control is what makes this comparison meaningful; show
both controls. State the result as: Act 3 structurally supersedes the v1
headline (3.4525 vs 3.0946 bits per character), which was single-seed, 500
steps, and carried four self-declared confounds - AND that Act 3 is itself
single-seed. Report the v1 numbers only as superseded context, never as current
results.

--- Act 4 (results/v2/act4-scaling/) - CENTREPIECE ---
  - Fit: L(N,D) = E + A/N^alpha + B/D^beta. Report alpha = 0.44 +/- 0.32,
    beta = 0.29 +/- 0.16, E = 1.605. These are YOUR fitted exponents.
  - LORO R^2 >= 0.985 on every held-out rung. State the leave-one-rung-out
    protocol.
  - State the pre-registered falsification condition and that it was NOT
    FALSIFIED. Include the FALSIFICATION_CONDITION.md content in the section.
  - FRAMING: report as CONSISTENCY with Chinchilla, not as an improvement or
    independent confirmation. Published alpha ~0.34.
  - MUST INCLUDE: the equal-D table (results/v2/act4-scaling/equal_D_table.csv)
    as a LaTeX table, since D=2M shows an inversion.
  - MUST CAVEAT: 6,487-token BPE at 2.27 chars/token; byte fallback would
    compress better and preserve 0% UNK, so alpha-vs-0.34 is a CAVEATED
    comparison. Interior ladder points are mid-schedule with LR not annealed -
    an unquantified bias, stated as such.
  - MUST REPORT: sensitivity bracket alpha in [0.445, 0.552], beta in
    [0.209, 0.290]; alpha range 0.314-0.477 across folds contains 0.34;
    residual RMS 0.0288 below seed spread; the D=2M inversion is a NULL
    (spread 0.186 vs gap 0.057).
  - Do not report the sweep as one point per model. The method - one run yields
    dozens of (N,D,C,loss) points from intermediate checkpoints - is what made
    the sweep affordable. Say so; it is the most transferable thing in the act.
  - figures act4_scaling_ladder and act4_loss_vs_N_fixed_D.

--- Act 5 (results/v2/act5/) ---
5a KV CACHE: byte-identical output (1.4 ULP divergence); speedup 3.71x at
    context 128 rising to 18.36x at context 1024; cache memory exactly linear
    in context; the speedup only appears once N > d (crossover ~288 for the
    9.0M model). CAVEAT IN THE FIGURE CAPTION: the 18.36x is a microbenchmark
    of the mechanism on a purpose-built context-1024 model - it is NOT a
    property of the canonical context-128 models. Do not quote it as a
    property of our trained models.
5b MQA/GQA: 8x cache reduction; -12.1% FLOPs, which CONTRADICTS Lecture 10's
    claim that these do not reduce compute - say this plainly, it is a real
    finding. Quality cost +0.0285 nats, two-sided p = 0.0008, n=6, 6/6 paired
    seeds consistent. REPORT n=5 AS PRIMARY (p = 0.0009) with n=6 as the
    sensitivity check, because seed 42's G=1 arm was re-established by inference
    from load semantics and value plausibility rather than by re-verification -
    state that provenance reasoning in the section or a footnote. The G=1/2/4/8
    curve (2.349/2.354/2.363/2.346) came from the v1 TinyShakespeare path, 500
    steps, character tokenizer, single seed: present it as CONTEXT ONLY, never
    as a result.
5c SPECULATIVE: a bounded NEGATIVE result with a mechanism. 0 of 50 cells beat
    plain decoding; best 0.622x. Acceptance was HIGH (0.857 median at K=1) and
    HIGHER at the 4.2x gap than at the 1.7x gap (0.90 vs 0.82), because the
    6.3M target is better trained. The arithmetic: a round yields
    (1 - alpha^(K+1))/(1 - alpha) tokens for 2K forwards, which is below 1.0 for
    every K > 1 even at perfect acceptance; and at 1.5M-6.3M on one MPS device
    both forwards are dispatch-latency-bound rather than compute-bound, so the
    target's batching advantage is negligible and the draft forward is pure
    cost. State the general lesson: speculative decoding pays when the target
    forward is compute-bound so that verifying K+1 positions in one pass is
    materially cheaper than K+1 sequential passes. Our models are too small.
    CAVEATS: timings are CPU best-of-3 at 0.03-0.11 s, a consistent comparison
    within one run, NOT a portable speedup figure. The distribution test is
    single-prompt, single-token on toy models plus the residual-correction
    argument; the zero-gap TV comparison is near-tautological (at zero gap
    speculative decoding IS plain sampling) and proves the code runs, not that
    the operator is correct on a real model.

  METHODOLOGY PARAGRAPH - 5c's silent double-feed bug gets its own boxed or
  clearly headed note. Speculative decoding re-fed the full sequence on top of
  a prefilled KV cache, double-counting context. It failed SILENTLY: the
  cached and the uncached equivalence tests both passed because both paths
  shared the bug. Acceptance was 15-40% at a zero draft-target gap where it
  must be 100%, and that invariant is what finally exposed it. Throughput rose
  10.7x after the fix. All 5c numbers before that commit are void. The
  generalisable lesson, stated in the text: a test that compares two
  implementations cannot detect a bug they both share. Sanity invariants catch
  what differential testing cannot.
  ALSO NOTE, in the same spirit: the 5a claim that beam search used the KV
  cache was FALSE - beam had no cache in any commit and no test covered it.
  Corrected, the conclusion held (beam-4 is 4.26x greedy for 0.061 bits, was
  4.9x). Mention it where the 5a claim is made.

--- Act 6 (results/v2/act6/6.3M_decoding.json) ---
  - The sweep table: config / BPC (ref) / rep-4 / distinct-2 / seconds, over
    beam8, beam4, beam2, greedy, temp 0.5, top-p 0.5, temp 1.0, temp 1.3.
    corr(distinct-2, BPC) = +0.87, corr(rep-4, BPC) = -0.79.
  - Finding: beam-4 beats greedy by 0.061 bits at 4.26x the time; beam-8 is
    worse than beam-4 at 2.2x the cost; greedy is the quality/cost knee -
    nothing is both cheaper and better. All 15 BPC values were bit-identical
    across repeat runs, which is a determinism check worth one sentence.
  - CAVEAT, prominent: the BPC axis is SELF-PREFERRING - the reference model is
    the model under test. It is valid for RANKING decoding configurations
    against each other under a fixed judge. It is NOT an absolute measure of
    text quality. Say so wherever BPC appears.
  - best-of-n: n=2/8/16 -> 1.8684 / 1.7459 / 1.6959 against a mean sample of
    ~1.97. +0.272 bits recovered at n=16, yet still 0.259 WORSE than greedy.
    The argument, which is the result: a self-referential reranker can only
    select text the model already finds likely, so best-of-n converges toward
    the model's own mode and never past it; greedy sits on that mode and 16
    random draws rarely reach it. Beating greedy needs an independent scorer,
    which a 6.3M model does not have.
  - self-consistency: a NEGATIVE, with the mechanism. Argmax accuracy is
    exactly chance at 0.3300 (99/300), Wilson CI [0.2792, 0.3851] containing
    0.3333 - this is structural, not bad luck: the model NEVER predicts
    "neutral" (recall 0.000), so argmax is always entailment, which on a
    balanced 3-way set scores exactly 1/3 by construction. On the 216 items it
    can ever answer, argmax accuracy is 0.4583 - BELOW the 0.5139 coin-flip
    baseline. Majority voting amplifies a biased argmax into the same bias, so
    there is nothing to amplify. Do not quote the n=1 figure of 0.3789 as
    partial competence; sampling is an unbiased draw from a biased distribution
    while argmax is dominated by the entailment over-prediction.

--- Act 7 (results/v2/act7/6.3M_sft_peft.json) ---
  - 7a zero-shot floor 0.0000, and it is ASSERTABLE: a positive control
    (oracle items through the teacher-forced, greedy, and exact-ICL scoring
    routes) scored 1.0 on all three while the real model scored 0/32 on
    identical items. Report the control.
  - 7b ICL - report as the COPY, not as a zero, and give the mechanism:
    on the 18 same-task addition items, the first generated token is the
    DEMONSTRATION's answer on 18/18 items, versus 0/18 zero-shot, where the
    model emits only punctuation. The model learned the response FORMAT of the
    demonstration and not the operation: it continues the surface pattern using
    the most salient number in context, which is the demo's number, not the
    query's. Paired A/B: 0/18 with demo vs 0/18 without, Wilson [0, 0.1759],
    0/18 items changed.
    ALSO REPORT the failure that voided the first attempt, because it is the
    same class as D11: the conventional 4-shot prompt is 328 tokens against a
    128-token context, so truncation deleted the demonstrations entirely
    (demonstrations present? False). The metric was sound; the prompt was
    unanswerable by construction. ICL is now fitted to the window with every
    selected demonstration asserted present, and re-measured.
    CAVEATS, all mandatory: the three ICL seeds are bit-identical (frozen
    model) so n=1, not n=3; shot selection is cost-driven so 110/128 items
    received the cheapest pool example, which is cross-task relative to the
    query - only the 18 addition items are a clean same-task test. Therefore do
    NOT write "no in-context learning at 6.3M". Write what was measured.
  - 7c matched 20k trainable-parameter budget: full 0.5755, adapters 0.2136,
    LoRA 0.1823, prompt tuning 0.0859, prefix tuning 0.0000. Read 7c
    TOGETHER with 7d and say why: at a matched budget, LoRA's rank is pinned low
    BY CONSTRUCTION, so the LoRA arm is compared at a point on its own curve
    that 7d shows has not flattened. Wilson is about +/-0.04 on 128 items, so
    the adapters-vs-LoRA ordering is NOT established - suggestive only.
    Prefix tuning's 0.0000 is a genuine failure to fit, not a bug: its
    final_train_loss is 3.2576. Say that explicitly - it is the evidence that
    distinguishes "did not fit" from "fit but scored zero".
  - 7d LoRA rank sweep r in {1..64}: 0.1823 rising monotonically to 0.5078,
    no plateau; r=64 reaches 0.5078 against full fine-tuning's 0.5755 at 11.5%
    of the parameters. Rank is a smooth capacity dial, not a threshold.
  - LIMA: do NOT write this as contradicting LIMA. Write it as exposing LIMA's
    PREMISE. LIMA argues capability already exists in the base model and SFT
    merely elicits it. Here zero-shot and ICL are both 0.0000, so there is no
    latent instruction-following to elicit - the 0.0703 -> 0.1667 -> 0.3255 ->
    0.5651 curve across n = 16/64/256/512 is LEARNING a skill, not aligning
    one. "Not supported at 6.3M" with that mechanism sentence is the result.
  - The n=16 row is effectively n=1 (0.0703, identical across all three seeds,
    because one batch per epoch gives the seeded shuffle nothing to shuffle).
    Annotate it in the table, the JSON, and the figure. No error bars on it.
  - Seeds vary optimisation ORDER only, not initialisation - the error bars are
    within-init order variance and between-init variance is unmeasured. State it.
  - Scope caveat: synthetic task family, not natural instructions; ICL is
    teacher-forced and so is not comparable to free-running numbers elsewhere.
  - 7d rank sweep and 7c budgets: report both in one subsection so the
    rank-constraint point is visible.

--- Act 8 (results/v2/act8/) ---
  8a SHARED PROMPT BUILDER. One formatter now serves RAG and SFT (fixes D2).
    The token-based context budget (fixes D1) is worth quantifying: the old
    character budget admitted 64 characters, about 32 tokens of a 302-token
    chunk, so 9.2% of the context survived; the token budget admits 77 tokens,
    2.4x more, with the assembled prompt at 120 <= 128 tokens. fit_context()
    binary-searches the longest context prefix whose assembled prompt fits in
    tokens while holding back an answer reserve, then re-encodes to verify.
    Report this as the cost of a real defect, measured.
  8b GROUNDING AND HALLUCINATION - FOUR CONDITIONS, ONE ITEM SET.
    THE RESULT, AND LEAD WITH IT:
      oracle chose the item's asserted number       9/40  (22.5%)
      closed-book chose the item's asserted number  11/40 (27.5%)
      McNemar oracle vs closed-book: discordant 1/3, p = 0.625 (n.s.)
      counterfactual tracks the edit: 0/40, bootstrap [0.000, 0.000]
      free generation correct: 0/40 in EVERY condition
    The model identifies which number a passage asserts at approximately the
    chance rate of the candidate set, AND removing the passage changes its
    choice on only 4 of 40 items, not in a consistent direction. Do NOT call it
    "below chance" - 9/40 versus 1/3 is not significantly below chance. The
    word is "at chance". Report the per-item chance baseline EXACTLY, because
    10 of 120 candidates were dropped for number truncation, so items that lost
    a candidate have chance 0.5 rather than 0.333 - either restrict the rate to
    items with the full candidate set or report chance-adjusted per-item rates.
    THE MECHANISM, and it is the interesting part:
      passage lift on the number the passage states:  +0.786 nats, p = 0.0017
      passage lift on a number the passage does NOT state: +0.583 nats, p = 0.0057
      context-specific discrimination (the difference): +0.204 nats,
        bootstrap [+0.044, +0.391], p = 0.030
      edit effect on the counterfactual: +0.034 nats, p = 0.308 (n.s.)
      edited number preferred under counterfactual: +0.039 nats, p = 0.839 (n.s.)
    Read: having a passage raises the likelihood of numbers appearing in it by
    roughly 0.6-0.8 nats WHETHER OR NOT THEY ARE ASSERTED. That is a
    bag-of-numbers-in-context bias, not an assertion-reading ability. The
    residual assertion-sensitivity of +0.204 nats (p=0.030) is the only trace of
    grounding, it is marginal, it is not accompanied by any behavioural change,
    and the behavioural point estimate is in the OPPOSITE direction. Report the
    two instruments disagreeing, and say the behavioural test is the one that
    matters.
    CAVEATS: the absolute hallucination rate is 1.000 in every condition, a
    CEILING - report it as a ceiling, never as a comparison between conditions.
    The retrieved condition is currently degenerate (retrieval success 0/40);
    I am diagnosing it. IF THE DIAGNOSIS HAS NOT LANDED BY THE TIME YOU WRITE
    THIS, write the retrieved condition as an open item with a stated reason,
    and NOT as a measured null. Numbers proxy for atomic claims - there is no
    fact base - so this measures GROUNDEDNESS, not truth. Denominator note: the
    unit of analysis is the ITEM, n = 40, so 0/40 gives a Wilson upper bound of
    0.0876 (not 0.031); free-generation repeats were averaged per item first
    and contrast CIs come from a percentile bootstrap that resamples items.
  8d TOOL USE. 0/40 well-formed CALC(...) under greedy, Wilson [0, 0.0876];
    0/120 under sampling with a 2.5% template-echo rate. But 19/40 greedy
    generations emit the literal "CALC(" and in all 19 the argument is exactly
    "s" - the model reproduces the tool-call FORMAT and cannot fill it. This is
    the same shape as 7b and the report should say so explicitly: format
    acquired, operation absent.

--- 6. ACTS 9 AND 10: PLACEHOLDERS ---

Write a complete section skeleton for each with the subsection structure and a
clearly marked \textbf{[PENDING - not yet measured]} where results go. The
report must compile and read as a coherent document with those gaps visible.
I will supply the content when those acts land.

--- 7. THREATS TO VALIDITY - ITS OWN SECTION ---

Consolidate, do not scatter:
  - Acts 2 and 3 are single-seed (n=1): they rank configurations, they do not
    establish effects. No error bars, no significance language.
  - Act 4's alpha comparison is caveated: 6,487-token BPE at 2.27 chars/token
    with rare Unicode; byte fallback would compress better and preserve 0% UNK.
    Interior ladder points are mid-schedule with LR not annealed - unquantified
    bias. The D=2M inversion is a null (spread 0.186 vs gap 0.057).
  - Act 5's G-curve is v1-path (500 steps, character tokenizer, single seed) and
    is context only.
  - Act 5a's 18.36x used a purpose-built context-1024 model; the canonical
    models are context-128.
  - Act 5c timings are CPU best-of-3, not a portable speedup figure; the
    distribution test is single-prompt single-token on toy models; the zero-gap
    TV comparison is near-tautological.
  - Act 5b's seed 42 G=1 arm was re-established by inference, not
    re-verification; report n=5 as primary.
  - Act 6's BPC axis is self-preferring - valid for ranking, not absolute.
  - Act 7's n=16 row is effectively n=1; seeds vary optimisation order only;
    synthetic task family; teacher-forced ICL; ICL shot selection is
    cost-driven and only 18 items are same-task.
  - Act 8's absolute hallucination rate is a ceiling; the retrieved condition
    is degenerate pending diagnosis; 10/120 candidates dropped; numbers proxy
    atomic claims.
  - The v1 headline (3.4525 vs 3.0946 BPC) is single-seed, 500 steps, four
    self-declared confounds, and is superseded by Act 3 - which is itself
    single-seed.
  - The SWA long-context curve was never measured (dropped for cost: 42.1
    min/run against a 1.8-2.4h estimate). Reported at context 128 only; no claim
    about long contexts is made.
  - The web interface has never been opened in a browser. Do not claim it works.
  - The benchmark is small: the 1.5M/2.6M/6.3M models, 128-token context,
    synthetic task families, C4-derived corpora. No claim generalises to models
    at scale.

--- 8. THE DO-NOT-ASSERT LIST GOES IN THE PAPER ---

Publish it. A table titled "Claims we do not make" or a clearly marked
subsection under Threats to Validity, listing all 14 items from the Act 7 and
Act 8 lists (I will forward the final list). This is a strength, not a
disclaimer - state that framing in one sentence. A course project that
publishes its own boundaries of validity is more credible than one that
publishes only its wins.

--- 9. DELIVERABLES AND MANIFEST ---

  1. report/final_report.tex - self-contained, compiles on Overleaf. No
     dependency on any git-ignored path. All figures referenced by relative
     path under report/figures/.
  2. report/figures/*.png AND *.pdf for every figure, each with a caption
     stating what is plotted, the n, and the interval type.
  3. src/eval/actN_figures.py - every new generator, committed.
  4. report/tables/*.csv - the LaTeX tables' source data, so a reader can
     re-derive them.
  5. report/MANIFEST.md - for EVERY figure and EVERY table: the exact path of
     the source JSON, the generating script, the row count, and the n. This is
     the audit trail and it must be complete.
  6. A CLAIMS TABLE in the paper itself (can be a table in Threats to
     Validity): each headline claim, its n, its interval, and its status
     (confirmed / n.s. / exploratory / invalid).

--- 10. VERIFICATION BEFORE YOU COMMIT ---

There is no pdflatex on this machine - the owner compiles on Overleaf. So verify
by other means and report exactly what you checked:
  - every \includegraphics path resolves to a file that exists, and the .pdf
    variant exists for each .png
  - brace and environment balance, every \begin has a matching \end
  - every \label is unique and every \ref/\cref resolves
  - no undefined macros; any macro you introduce is defined before use
  - no absolute paths, no \input of anything outside report/
  - no missing packages for anything you used
  - no orphan figures or tables
  Run chktex if it is available; if it is not, write a small checker for the
    checks above and commit it as tools/check_tex.py so the check is repeatable.
  If you find and fix a compile error, SAY SO and say what it was.

--- 11. COMMIT ---

One commit, or two if the figures and the tex are separable. WHY in the message:
what the report now claims, and what it explicitly declines to claim. Not "wrote
report" - state the argument the report makes.

--- 12. REPORT BACK ---

  - the section outline as written, with a one-line claim per section
  - the figure and table inventory, each with its source JSON path
  - the CLAIMS TABLE content: every claim, n, interval, status
  - any number where the JSON and my prompt disagreed
  - any placeholder left unfilled and why
  - the verification checks you ran and their outcomes
  - anything in the ledger you think I am misrepresenting

--- END OF SPEC ---

--- WHAT YOU WRITE ---

Acts 0-8 of report/final_report.tex, with Acts 9 and 10 as visible placeholders
per the spec. Also, per the spec: figure generator scripts, figure PNG+PDF
pairs, report/tables/*.csv, report/MANIFEST.md, tools/check_tex.py,
and report/claims.csv.

--- THE CLAIMS LEDGER (the review surface) ---

report/claims.csv, with header:
claim_id,act,claim,n,interval,status,source_json,caveat

  - One row per quantitative or qualitative claim that appears in the .tex.
  - n and interval are the MEASURED values from the source JSON, or "n/a" with
    a reason. Never infer them.
  - status is one of: confirmed | n.s. | exploratory | ceiling | invalid |
    superseded | open-item
  - source_json is the exact path of the file the number came from.
  - caveat is the qualifier that must travel WITH the claim, or "none".
  - Every claim in the .tex must be present in this CSV, and every CSV row
    must be traceable to a source file. tools/check_tex.py verifies this
    correspondence and reports any claim in the .tex with no ledger row.

--- CAVEAT DISCIPLINE (the rule that matters most) ---

  - Caveats may be STRENGTHENED, never weakened, dropped, or paraphrased into
    something more confident. If you believe a caveat is wrong or misleading,
    FLAG IT in your report-back and write your objection as a comment - do not
    silently fix it.
  - A number without its n, its interval, and its status is incomplete. If the
    spec's phrasing of a claim is stronger than the evidence supports, write
    the weaker, supported version.
  - The 14-item do-not-assert list from Acts 7 and 8 goes into the paper as a
    published table of claims NOT made. Frame it as a strength.
  - For any claim you cannot source from a JSON file, write
    \textbf{[PENDING - SOURCE NOT FOUND]} and list it in your report-back. Do
    not estimate, do not interpolate, do not carry it over from the spec text
    without checking the JSON.

--- VERIFICATION (the only correctness gate - no pdflatex on this machine) ---

Per the spec, section 10. The owner compiles on Overleaf, so anything you break
will be found by them at the worst possible moment. Run every check, report
every outcome, and fix what you find. If unsure whether a construct compiles,
prefer the boring one. Commit tools/check_tex.py so the check is repeatable.

--- REPORT BACK ---

  - the section outline as written, one line per section stating its claim
  - the figure and table inventory, each with its source JSON path
  - the full contents of report/claims.csv
  - every place the spec and a JSON disagreed
  - every caveat you think I have wrong, with your objection
  - every unfilled placeholder and why
  - the verification checks you ran and their outcomes
  - the list of files you created or modified
  - the HEAD commit you started from

--- END OF SETUP INSTRUCTIONS ---

--- BEGIN APPENDIX TO WRITE INTO report/REPORT_SPEC.md (main-session brief, NOT report content) ---

--- MoE addendum (for the main session; not a report item) ---
Top-2 routing with a noisy top-K gate, load-balancing auxiliary loss
N * sum_i(f_i * P_i) as the syllabus specifies, capacity factor fixed at 1.25
for this size. Train at 2.6M only, 3 seeds, against the existing dense 2.6M
baseline as the control - same D, same tokens, same schedule, so the only
difference is the MoE layer. Report quality delta, the load-balancing loss
trajectory, and per-expert routing entropy. Keep the gate deterministic at eval
time. Also record the parameter count honestly: MoE adds parameters, so the
comparison must be at MATCHED PARAMETERS as well as matched D, or state
plainly that the comparison is confounded by capacity and say so in the
limitations. Leave the dense baseline's pinned checkpoint untouched.

--- Framing brief (main session; supersedes parts of the spec above for Act 8) ---
1. FIX THE HEADLINE ORDERING. THE CHOICE RATES ARE THE RESULT -- the lift rows
are the mechanism and they cut against grounding.
2. ANSWER THE CANDIDATE-CONSTRUCTION QUESTION -- the "unstated number" is
ABSENT from the passage (verified 0/40), so the lift is a generic context
effect.
3. DIAGNOSE THE RETRIEVER -- done: it was a query-design failure, now fixed.
4. Act 8d: SHOW THE TOKEN IDS FOR "CALC(s" -- done, shipped in
results/v2/act8/tool_use.json under "template_completion_evidence".
5. Act 9: causal head ablation with a random-ablation null; behavioural
attention rollout hit-rate; logit lens per-layer probe.
6. Act 10: measurement-audit table, measurement-failure taxonomy, ethics
metrics at floor, reproducibility script.
7. MoE decision: minimal add-on at 2.6M (option a) or out of scope (option b).

--- END APPENDIX ---

--- TICK LIST (maintained by the author of this ticket) ---

- [x] A1 sequencing — clean tree confirmed, HEAD `42407c4` recorded. No mutating
      git command was run: this ticket does not add, commit, branch or stash.
- [x] A2 spec persistence — this file written before any other work.
- [x] A3 stale line — ignored as instructed.
- [x] A4/A5 Act 8 — SUPERSEDED by the landed revision (`42407c4`); Act 8 is
      written as FINAL from `results/v2/act8/hallucination.json` and
      `results/v2/act8/tool_use.json`. The retrieved condition is NOT an open
      item. The per-item chance baseline is 0.1838, never 1/3. The spec's
      sentence "removing the passage changes its choice on only 4 of 40 items"
      does not match the JSON (8 of 40 answers change; 1+3 = 4 are the McNemar
      discordant pairs), so the JSON figure is used and the difference is
      reported.
- [x] A6 Acts 9 and 10 — placeholders with visible `[PENDING - not yet measured]`.
- [x] A7 Act 10 overlap — `results/v2/act10/` does not exist yet; prose written
      from the ledger and the pending artefact is named in the text.
- [x] A8 verification — `tools/check_tex.py` written and run; results reported
      in the report-back.

--- WHAT WAS PRODUCED (for the commit message) ---

New files:
  report/final_report.tex        (rewritten: Acts 0-8 + 9/10 placeholders)
  report/REPORT_SPEC.md           (this file)
  report/claims.csv               (63-row claim ledger)
  report/MANIFEST.md              (audit trail for every figure and table)
  report/tables/*.csv             (24 table source files)
  report/figures/act{0,1,2,3}_*.{png,pdf}  (8 new figure files)
  src/eval/act0_figures.py, act1_figures.py, act2_figures.py,
  src/eval/act3_figures.py, act4_figures.py, report_tables.py
  tools/check_tex.py, tools/build_claims_csv.py
  results/v2/act4-scaling/ladder_points.csv, ladder_rungs.csv,
  results/v2/act4-scaling/corpus_manifest.json

Modified files:
  report/figures/act4_scaling_ladder.{png,pdf},
  report/figures/act4_loss_vs_N_fixed_D.{png,pdf}  (regenerated from tracked data)
  tests/test_report.py           (v1 assertions replaced with v2 ones)

The spec text above is unmodified. Only the tick list and this appendix were
added, and the A4/A5 entry records the supersession.

