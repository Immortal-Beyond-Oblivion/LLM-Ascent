# Falsification condition (Act 4 — scaling laws)

Pre-registered before any Stage 1 run. The claim under test: "Loss scales
predictably with N (non-embedding parameters) and D (training tokens) in the
~0.8M - 9M regime (L(N,D) = E + A/N^alpha + B/D^beta)."

Falsified if ANY of:
1. the fitted model fails to beat a constant-loss baseline on held-out points
   (R^2 <= 0 vs the rung-mean baseline, OR equivalently the fit does not
   decrease held-out loss beyond the constant);
2. residuals show systematic curvature in log N or log D (e.g. a structured
   pattern in residual-vs-log-N or residual-vs-log-D that indicates a missing
   lower- or higher-order term rather than uncorrelated noise).

Additional acceptance checks recorded now:
- every rung must produce >= 50 distinct D values (identifies beta);
- embedding share is reported per rung but does not invalidate the fit;
- a single-seed rung may not be presented with an error bar.

Anything beyond these criteria, decided after seeing the fit, is labelled
exploratory in the report.

Date written (pre-run, earliest point of this Stage 1 execution):
2026-10-15
