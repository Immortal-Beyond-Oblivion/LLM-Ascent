#!/usr/bin/env python3
"""Repeatable correctness gate for report/final_report.tex.

There is no ``pdflatex`` on the machine that runs this, and the owner compiles
on Overleaf, so this script is the only thing standing between a broken
``.tex`` and the owner. It checks the things a LaTeX compiler would catch,
plus the things only this project can catch:

  1. every ``\\includegraphics`` target exists under report/figures/ and has a
     vector PDF sibling;
  2. every file in report/figures/ is either referenced by the report or
     listed in the explicit exclusion list below, with a reason;
  3. every figure and table environment is inside report/figures/ or is
     self-contained, and is not an orphan (it has a caption, and it is not
     left over from the v1 report);
  4. brace balance and environment balance;
  5. every ``\\label`` is unique, every ``\\ref``/``\\cref``/``\\eqref``/
     ``\\autoref`` resolves, and every label is attached to a figure, a table
     or a section;
  6. no undefined macro: every macro this file defines is defined before it
     is used, and no command outside a known-package whitelist is used;
  7. no absolute path and no ``\\input``/``\\include`` outside report/;
  8. every package used is in the preamble;
  9. every claim in the text has a row in report/claims.csv, and every row in
     report/claims.csv is referenced from the text;
 10. inline math is balanced: no math span crosses a blank line, no
     math-only command lands in text mode, and the file does not end in math
     mode.  An unbalanced ``$`` is invisible without ``pdflatex`` and is not
     cosmetic when it happens: LaTeX reports ``Missing $ inserted`` and every
     later ``$`` in the document then toggles the wrong way.

Exit status is 0 when everything passes and 1 otherwise.

Usage:
    python tools/check_tex.py [--tex PATH] [--claims PATH] [--quiet]
    python tools/check_tex.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── a conservative whitelist of commands that need no extra package ──────────
BASE_MACROS = {
    "documentclass", "usepackage", "geometry", "fontenc", "lmodern",
    "graphicx", "booktabs", "array", "float", "amsmath", "longtable",
    "hypersetup", "href", "url", "newcolumntype", "graphicspath", "title",
    "thetable",
    "author", "date", "maketitle", "begin", "end", "section", "subsection",
    "subsubsection", "paragraph", "chapter", "part", "appendix", "label",
    "ref", "eqref", "pageref", "cite", "bibitem", "thebibliography",
    "bibliography", "bibliographystyle", "newblock", "item", "itemize",
    "enumerate", "description", "textbf", "textit", "emph", "texttt",
    "textsc", "textrm", "textsf", "underline", "textsuperscript",
    "textsubscript", "underline", "text", "mathrm", "mathbf", "mathit",
    "mathcal", "mathbb", "operatorname", "overline", "hat", "tilde", "dot",
    "frac", "dfrac", "tfrac", "sqrt", "left", "right", "big", "Big",
    "bigg", "Bigg", "sum", "prod", "int", "lim", "infty", "partial",
    "nabla", "cdot", "times", "div", "pm", "mp", "leq", "geq", "neq",
    "approx", "equiv", "sim", "simeq", "propto", "subset", "subseteq",
    "cup", "cap", "in", "notin", "to", "rightarrow", "leftarrow", "Rightarrow",
    "Rightarrow", "iff", "forall", "exists", "neg", "lnot", "land", "lor",
    "wedge", "vee", "dots", "ldots", "cdots", "vdots", "ddots", "angle",
    "triangle", "square", "circ", "bullet", "star", "ast", "oplus",
    "otimes", "perp", "parallel", "alpha", "beta", "gamma", "delta",
    "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu",
    "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
    "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
    "Upsilon", "Phi", "Psi", "Omega", "leq", "ge", "ne", "ldotp", "ldots",
    "toprule", "midrule", "bottomrule", "cmidrule", "cline", "multicolumn",
    "multirow", "hline", "hline", "rule", "centering", "caption",
    "includegraphics", "resizebox", "scalebox", "rotatebox", "parbox",
    "vspace", "hspace", "smallskip", "medskip", "bigskip", "noindent",
    "indent", "linebreak", "newpage", "clearpage", "pagebreak", "hfill",
    "tableofcontents", "renewcommand", "appendix", "setlength",
    "allowbreak", "arabic",
    "vfill", "hfill", "textbf", "url", "verb", "texttt", "LaTeX", "LaTeXe",
    "ae", "oe", "aa", "AA", "o", "l", "i", "j", "ss", "pounds", "copyright",
    "S", "P", "today", "appendix", "today", "figurename", "tablename",
    "contentsline", "protect", "relax", "ensuremath", "textsuperscript",
    "textsubscript", "textrm", "textsf", "textsc", "texttt", "textup",
    "textbf", "textit", "textmd", "textnormal", "emph", "fbox", "framebox",
    "makebox", "parbox", "raisebox", "rule", "strut", "phantom", "llap",
    "rlap", "clap", "hfill", "kern", "hskip", "vskip", "break", "nobreak",
    "newline", "linebreak", "centering", "raggedright", "raggedleft",
    "arraybackslash", "array", "textsuperscript", "textbackslash",
    "textbackslash", "ensuremath", "linewidth", "columnwidth",
    "textwidth", "textheight", "paperwidth", "paperheight", "baselineskip",
    "parindent", "parskip", "emergencystretch", "hbadness", "hfuzz",
    "vbadness", "vfuzz", "arraystretch", "tabcolsep", "extrarowheight",
    "arraybackslash", "bfseries", "itshape", "ttfamily", "rmfamily",
    "sffamily", "upshape", "slshape", "scshape", "normalfont", "tiny",
    "scriptsize", "footnotesize", "small", "normalsize", "large", "Large",
    "LARGE", "huge", "Huge", "bf", "it", "tt", "sf", "rm", "em", "sc",
    "selectfont", "usebox", "box", "kern", "enquote", "glqq", "grqq",
    "textquotedblleft", "textquotedblright", "textquoteleft",
    "textquoteright", "guillemotleft", "guillemotright", "textendash",
    "textemdash", "textbullet", "textperiodcentered", "textless",
    "textgreater", "textasciitilde", "textasciicircum", "textbar",
    "textbackslash", "textunderscore", "textvisiblespace", "textcomp",
    "dagger", "ddagger", "ell", "log", "ln", "min", "max", "exp", "log10",
    "sin", "cos", "tan", "arg", "deg", "gcd", "det", "dim", "ker", "hom",
    "bmod", "pmod", "Pr", "mod", "bmod", "sec", "gcd", "choose", "atop",
    "brace", "brack", "left", "right", "middle", "bigl", "bigr", "Bigl",
    "Bigr", "biggl", "biggr", "Biggl", "Biggr", "bigl", "bigr", "bigl",
    "label", "ref", "pageref", "cite", "nocite", "bibliographystyle",
    "setlength", "addtolength", "hspace", "vspace", "vskip", "hskip",
    "linewidth", "textwidth", "columnwidth", "arraystretch", "tabcolsep",
    "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge",
    "Huge", "scriptsize", "tiny", "bf", "it", "tt", "sf", "rm", "sc", "em",
    "itshape", "bfseries", "ttfamily", "rmfamily", "sffamily", "upshape",
    "slshape", "scshape", "normalfont", "selectfont", "usebox", "kern",
    "raisebox", "makebox", "fbox", "framebox", "phantom", "llap", "rlap",
    "clap", "strut", "hfill", "vfill", "centering", "raggedright",
    "raggedleft", "arraybackslash", "protect", "relax", "ensuremath",
    "today", "appendix", "contentsline", "newblock", "thebibliography",
    "toprule", "midrule", "bottomrule", "cmidrule", "cline", "hline",
    "rule", "multicolumn", "multirow", "rotatebox", "scalebox", "resizebox",
    "url", "verb", "texttt", "LaTeX", "LaTeXe", "TeX", "ss", "S", "P",
    "textquotedblleft", "textquotedblright", "textquoteleft",
    "textquoteright", "textendash", "textemdash", "textbullet", "textbar",
    "guillemotleft", "guillemotright", "textless", "textgreater",
    "textasciitilde", "textasciicircum", "textbackslash", "textunderscore",
    "input",
}

# commands that require a specific package
PACKAGE_COMMANDS = {
    "includegraphics": "graphicx",
    "graphicspath": "graphicx",
    "resizebox": "graphicx",
    "scalebox": "graphicx",
    "rotatebox": "graphicx",
    "toprule": "booktabs",
    "midrule": "booktabs",
    "bottomrule": "booktabs",
    "cmidrule": "booktabs",
    "newcolumntype": "array",
    "text": "amsmath",
    "frac": "amsmath",
    "dfrac": "amsmath",
    "tfrac": "amsmath",
    "operatorname": "amsmath",
    "mathbb": "amsmath",
    "begin": None,          # longtable is checked separately
    "H": "float",
}

CLAIM_STATUS = {
    "confirmed", "n.s.", "exploratory", "ceiling", "invalid", "superseded",
    "open-item",
}

CLAIM_HEADER = [
    "claim_id", "act", "claim", "n", "interval", "status", "source_json",
    "caveat",
]

# ── files in report/figures/ that the report does NOT reference ──────────────
#
# The check runs in BOTH directions: every referenced figure must exist, and
# every file that exists must be either referenced or listed here with a
# reason.  A figure that is silently dropped from the text is a figure whose
# caption no longer guards it; a figure that nobody can explain is clutter
# that will drift from its result.
#
# Every entry is a stem (the file exists as <stem>.png and <stem>.pdf).
FIGURE_EXCLUSIONS = {
    "loss_curves": "v1 artefact (results/v1/comparison.json, src/eval/figures.py): "
                   "the v1 loss curves, superseded by Act 3 and still referenced "
                   "by the v1 PDF report/LLM_Ascent_Project_Report v1.pdf",
    "perplexity_params": "v1 artefact (results/v1/comparison.json, "
                         "src/eval/figures.py): the v1 perplexity-against-"
                         "parameters plot, superseded by Act 3 and still "
                         "referenced by the v1 PDF",
    "latency": "v1 artefact (results/v1/comparison.json, src/eval/figures.py): "
               "the v1 latency plot, superseded by Act 5a and still referenced "
               "by the v1 PDF",
    "attention": "v1 artefact (results/v1/figures.json, src/eval/figures.py): "
                 "the v1 attention heatmap, which Act 9b shows must not be read "
                 "as evidence of grounding; kept for the v1 PDF only",
}


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, honouring escaped percent signs."""
    out = []
    for line in text.splitlines():
        buf = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_graphics(tex: str, tex_path: Path, errors: list[str],
                   notes: list[str]) -> None:
    figures_dir = tex_path.parent / "figures"
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    if not names:
        errors.append("the report references no figures at all")
    referenced_stems = set()
    for name in names:
        if name.startswith("/") or ".." in name:
            errors.append(f"\\includegraphics{{{name}}} is not a relative path "
                          f"inside report/figures/")
            continue
        referenced_stems.add(Path(name).stem)
        path = figures_dir / name
        if not path.exists():
            # an extensionless \includegraphics lets graphicx pick the vector
            # PDF; resolve it the same way graphicx would
            for suffix in (".pdf", ".png", ".jpg"):
                if path.with_suffix(suffix).exists():
                    path = path.with_suffix(suffix)
                    break
        if not path.exists():
            errors.append(f"\\includegraphics{{{name}}} has no file at "
                          f"{path}")
            continue
        vector = path if path.suffix == ".pdf" else path.with_suffix(".pdf")
        if not vector.exists():
            errors.append(f"figure {name} has no vector PDF sibling "
                          f"({vector.name} missing)")

    # every figure environment must contain an includegraphics
    for block in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex,
                            flags=re.S):
        if "\\includegraphics" not in block:
            errors.append("a figure environment contains no \\includegraphics")

    # THE OTHER DIRECTION: every figure that exists must be referenced, or be
    # on the explicit exclusion list with a reason.
    if figures_dir.is_dir():
        on_disk = sorted({p.stem for p in figures_dir.iterdir()
                          if p.suffix.lower() in (".png", ".pdf")})
        for stem in on_disk:
            if stem in referenced_stems:
                continue
            if stem in FIGURE_EXCLUSIONS:
                continue
            errors.append(f"figure {stem} is in report/figures/ but is not "
                          f"referenced by the report and is not on the "
                          f"checker's exclusion list "
                          f"(tools/check_tex.py, FIGURE_EXCLUSIONS)")
        unreferenced = [s for s in on_disk
                        if s not in referenced_stems
                        and s not in FIGURE_EXCLUSIONS]
        notes.append(f"{len(referenced_stems)} figures referenced, "
                     f"{len(on_disk)} figure stems on disk, "
                     f"{len(FIGURE_EXCLUSIONS)} exclusions, "
                     f"{len(unreferenced)} unexplained")
    else:
        errors.append(f"no figures directory at {figures_dir}")


def check_environments(tex: str, errors: list[str]) -> None:
    begins = re.findall(r"\\begin\{([^}]+)\}", tex)
    ends = re.findall(r"\\end\{([^}]+)\}", tex)
    if sorted(begins) != sorted(ends):
        from collections import Counter
        cb, ce = Counter(begins), Counter(ends)
        for name in sorted(set(cb) | set(ce)):
            if cb[name] != ce[name]:
                errors.append(f"environment {name}: {cb[name]} \\begin vs "
                              f"{ce[name]} \\end")
    # brace balance, ignoring \{ and \}
    depth = 0
    for i, ch in enumerate(tex):
        if ch == "\\":
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"unbalanced closing brace near offset {i}")
                return
    if depth != 0:
        errors.append(f"unbalanced braces: net depth {depth}")


def check_labels_and_refs(tex: str, errors: list[str]) -> None:
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    dupes = {l for l in labels if labels.count(l) > 1}
    for d in sorted(dupes):
        errors.append(f"duplicate \\label{{{d}}}")

    refs = set()
    for cmd in ("ref", "eqref", "pageref", "autoref", "cref", "Cref",
                "crefrange", "nameref"):
        refs.update(re.findall(r"\\" + cmd + r"\{([^}]+)\}", tex))
    for r in sorted(refs):
        if r not in labels:
            errors.append(f"\\ref{{{r}}} does not resolve to any \\label")

    # every label must be attached to a figure, a table, a longtable or a
    # section; an unattached label is an orphan
    attached = set()
    for env in ("figure", "table", "longtable"):
        for block in re.findall(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
                                tex, flags=re.S):
            attached.update(re.findall(r"\\label\{([^}]+)\}", block))

    # a label is attached to the nearest preceding sectioning command
    sectioning = list(re.finditer(
        r"\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{", tex))
    for m in re.finditer(r"\\label\{([^}]+)\}", tex):
        owner = None
        for s in sectioning:
            if s.start() < m.start():
                owner = s
            else:
                break
        if owner is not None:
            # the label must come before the next sectioning command
            nxt = [s.start() for s in sectioning if s.start() > owner.start()]
            limit = nxt[0] if nxt else len(tex)
            if owner.start() < m.start() < limit:
                attached.add(m.group(1))
    for l in labels:
        if l not in attached:
            errors.append(f"\\label{{{l}}} is not attached to a figure, table "
                          f"or section (orphan label)")


def check_captions(tex: str, errors: list[str]) -> None:
    for env in ("figure", "table", "longtable"):
        blocks = re.findall(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", tex,
                            flags=re.S)
        for block in blocks:
            if "\\caption" not in block:
                errors.append(f"a {env} environment has no \\caption")


def check_macros(tex: str, errors: list[str]) -> None:
    # macros defined in this file, and the position of the definition
    defs = {}
    for m in re.finditer(r"\\(?:newcommand|renewcommand|DeclareMathOperator)"
                         r"(?:\*)?\{?\\([A-Za-z@]+)", tex):
        defs.setdefault(m.group(1), m.start())

    used = {}
    for m in re.finditer(r"\\([A-Za-z@]+)", tex):
        used.setdefault(m.group(1), []).append(m.start())

    for name, pos in sorted(defs.items()):
        first_use = used.get(name, [pos])[0]
        if first_use < pos:
            errors.append(f"macro \\{name} is used before it is defined")

    # unknown commands: not in the whitelist and not defined here
    for name, positions in sorted(used.items()):
        if name in defs or name in BASE_MACROS:
            continue
        if name.startswith("end") or name.startswith("begin"):
            continue
        errors.append(f"command \\{name} is not in the known-command list and "
                      f"is not defined in this file (first use at offset "
                      f"{positions[0]})")


def check_packages(tex: str, errors: list[str]) -> None:
    preamble = tex.split(r"\begin{document}")[0]
    loaded = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", preamble))
    loaded = {p.strip() for group in loaded for p in group.split(",")}
    for cmd, package in PACKAGE_COMMANDS.items():
        if package and cmd in re.findall(r"\\([A-Za-z@]+)", tex):
            if package not in loaded:
                errors.append(f"\\{cmd} needs the '{package}' package, which "
                              f"is not in the preamble")
    if "longtable" in tex and "longtable" not in loaded:
        errors.append("the report uses longtable but does not load it")
    if "hyperref" not in loaded and "\\href" in tex:
        errors.append("the report uses \\href but does not load hyperref")


def check_paths(tex: str, tex_path: Path, errors: list[str]) -> None:
    for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", tex):
        target = m.group(1)
        if target.startswith("/") or ".." in target:
            errors.append(f"\\input{{{target}}} escapes report/")
            continue
        # a generated fragment that is read in must be present in the bundle,
        # or the report will not compile where it is compiled
        resolved = (tex_path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"\\input{{{target}}} has no file at {resolved} "
                          f"(the report bundle is incomplete)")
    for m in re.finditer(r"\\(?:includegraphics|graphicspath)\{?([^}]*)\}?", tex):
        pass
    for m in re.finditer(r"\\(?:input|include|includegraphics)\{([^}]+)\}", tex):
        target = m.group(1)
        if target.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", target):
            errors.append(f"absolute path used: {target}")


def check_claims(tex: str, claims_path: Path, errors: list[str],
                 notes: list[str]) -> None:
    if not claims_path.exists():
        errors.append(f"missing claims ledger: {claims_path}")
        return
    with claims_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        errors.append("claims ledger is empty")
        return
    header = rows[0]
    if header != CLAIM_HEADER:
        errors.append(f"claims ledger header is {header}, expected "
                      f"{CLAIM_HEADER}")
        return
    body = [r for r in rows[1:] if r]
    ids = [r[0] for r in body]
    dupes = {i for i in ids if ids.count(i) > 1}
    for d in sorted(dupes):
        errors.append(f"claims ledger: duplicate claim_id {d}")

    by_id = {}
    for r in body:
        if len(r) != len(CLAIM_HEADER):
            errors.append(f"claims ledger: row {r[0]!r} has {len(r)} fields, "
                          f"expected {len(CLAIM_HEADER)}")
            continue
        by_id[r[0]] = dict(zip(CLAIM_HEADER, r))
        if r[5] not in CLAIM_STATUS:
            errors.append(f"claims ledger: {r[0]} has status {r[5]!r}, which "
                          f"is not one of {sorted(CLAIM_STATUS)}")
        if not r[3].strip():
            errors.append(f"claims ledger: {r[0]} has an empty n")
        if not r[4].strip():
            errors.append(f"claims ledger: {r[0]} has an empty interval")
        src = r[6].strip()
        if not src:
            errors.append(f"claims ledger: {r[0]} has no source_json")
        else:
            for one in src.split(";"):
                one = one.strip()
                if not one:
                    continue
                if not (REPO_ROOT / one).exists():
                    errors.append(f"claims ledger: {r[0]} points at a source "
                                  f"file that does not exist: {one}")

    # markers in the tex:  % claims: ID1, ID2
    markers = re.findall(r"%\s*claims?:\s*(.+)$", tex, flags=re.M)
    referenced = set()
    for line in markers:
        for one in re.split(r"[,\s]+", line.strip()):
            one = one.strip().rstrip(",")
            if not one:
                continue
            referenced.add(one)
            if one not in by_id:
                errors.append(f"the text claims {one!r} but report/claims.csv "
                              f"has no such claim_id")
    for cid in sorted(set(by_id) - referenced):
        errors.append(f"claims ledger row {cid} is not referenced from the "
                      f"text (add a '% claims: {cid}' marker or drop the row)")
    notes.append(f"{len(by_id)} ledger rows, {len(referenced)} referenced "
                 f"from the text")


def check_math_balance(tex: str, errors: list[str]) -> None:
    """Inline math must be balanced, the way LaTeX sees it.

    There is no ``pdflatex`` on this machine, so an unbalanced ``$`` is
    invisible until the owner compiles on Overleaf --- where it is not a
    cosmetic problem: LaTeX reports ``Missing $ inserted``, enters math mode
    in the wrong place, and every later ``$`` in the document then toggles
    the wrong way.  This check walks the file the way TeX would and reports

      * a math span that crosses a blank line,
      * a math-only command that lands in text mode, and
      * a file that ends in math mode.

    Math environments (equation, displaymath) are skipped: they do not use
    ``$`` and their contents are math by construction.
    """
    math_only = (
        "rightarrow", "leftarrow", "pm", "times", "geq", "leq", "neq",
        "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
        "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "omega",
        "sum", "prod", "int", "infty", "cdot", "approx", "equiv", "sim",
        "partial", "nabla", "ldots", "cdots",
    )
    math_only_re = re.compile(r"\\(" + "|".join(math_only) + r")\b")
    text_ok_re = re.compile(r"\\(emph|textbf|texttt|textit|mathrm|mbox|textrm|"
                            r"textsc|text)\b")

    in_math = False
    in_math_env = False
    for lineno, line in enumerate(tex.split("\n"), 1):
        if re.search(r"\\begin\{(equation\*?|displaymath\*?|align\*?|gather\*?)\}",
                     line):
            in_math_env = True
        if re.search(r"\\end\{(equation\*?|displaymath\*?|align\*?|gather\*?)\}",
                     line):
            in_math_env = False
            continue
        if in_math_env:
            continue
        j = 0
        while j < len(line):
            ch = line[j]
            if ch == "\\" and j + 1 < len(line):
                if not in_math:
                    m = math_only_re.match(line, j)
                    if m and not text_ok_re.match(line, j):
                        errors.append(f"line {lineno}: math command "
                                      f"{m.group(0)} is in TEXT mode, which "
                                      f"means an unbalanced $ earlier")
                j += 2
                continue
            if ch == "$":
                in_math = not in_math
                j += 1
                continue
            j += 1
        if in_math and not line.strip():
            errors.append(f"line {lineno}: inline math mode crosses a blank "
                          f"line (an unbalanced $)")
    if in_math:
        errors.append("the file ends in math mode (an odd number of $)")


def check_bare_underscore(tex: str, errors: list[str]) -> None:
    """An unescaped ``_`` outside math is ``Missing $ inserted``.

    A CSV path written into a generated table as ``findings_taxonomy.csv``
    stopped the whole document compiling, on this machine and on Overleaf,
    and nothing caught it: the file existed, the references resolved, and the
    checker only looked at ``final_report.tex``.  This check walks the text
    the way TeX does --- ``_`` is legal inside ``$...$`` and inside math
    environments, and illegal everywhere else.
    """
    in_math = False
    in_math_env = False
    # Arguments of these commands are filenames or keys, not typeset text, so
    # an underscore inside them is legal.
    inert = re.compile(
        r"\\(includegraphics|input|include|label|ref|eqref|autoref|cite|url|"
        r"graphicspath|bibliography|begin|end|newcolumntype)"
        r"(\[[^\]]*\])?\{[^{}]*\}")
    for lineno, line in enumerate(tex.split("\n"), 1):
        if re.search(r"\\begin\{(equation\*?|displaymath\*?|align\*?|gather\*?)\}",
                     line):
            in_math_env = True
        if re.search(r"\\end\{(equation\*?|displaymath\*?|align\*?|gather\*?)\}",
                     line):
            in_math_env = False
            continue
        if in_math_env:
            continue
        line = inert.sub("", line)
        j = 0
        while j < len(line):
            ch = line[j]
            if ch == "\\" and j + 1 < len(line):
                j += 2
                continue
            if ch == "$":
                in_math = not in_math
                j += 1
                continue
            if ch == "_" and not in_math:
                errors.append(f"line {lineno}: unescaped _ outside math "
                              f"(TeX will report 'Missing $ inserted')")
                break
            j += 1


def collect_bundled(raw: str, tex_path: Path) -> str:
    """Return the document text plus the text of every ``\\input`` fragment.

    The report reads three generated fragments out of ``report/tables/``, and
    two of them carry a table (and its label).  A checker that only reads
    ``final_report.tex`` therefore cannot see those tables at all: it would
    not check their captions, and every ``\\ref`` to them would look
    unresolved.  So the bundle is followed: each ``\\input`` target that
    exists inside the report directory is appended to the text that is
    checked.  A target that does NOT exist is skipped here and reported by
    ``check_paths``, which is the check that owns that error.

    The recursion is bounded by a visited set, so a fragment that inputs
    itself cannot loop.
    """
    parts = [raw]
    seen = {tex_path.resolve()}
    frontier = list(re.finditer(r"\\input\{([^}]+)\}", raw))
    while frontier:
        m = frontier.pop()
        target = m.group(1)
        if target.startswith("/") or ".." in target:
            continue
        resolved = (tex_path.parent / target).resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            text = read(resolved)
        except OSError:
            continue
        parts.append(text)
        frontier.extend(re.finditer(r"\\input\{([^}]+)\}", text))
    return "\n".join(parts)


def run_checks(tex_path: Path, claims_path: Path) -> tuple[list[str], list[str]]:
    """Run every check against one report bundle.  Returns (errors, notes)."""
    raw = read(tex_path)
    raw = collect_bundled(raw, tex_path)
    tex = strip_comments(raw)

    errors: list[str] = []
    notes: list[str] = []

    check_graphics(tex, tex_path, errors, notes)
    check_environments(tex, errors)
    check_labels_and_refs(tex, errors)
    check_captions(tex, errors)
    check_macros(tex, errors)
    check_packages(tex, errors)
    check_paths(tex, tex_path, errors)
    check_math_balance(tex, errors)
    check_bare_underscore(tex, errors)
    check_claims(raw, claims_path, errors, notes)
    return errors, notes


# ── the negative self-test ───────────────────────────────────────────────────
#
# A gate that cannot fail is not a gate.  The self-test below takes the real
# report, copies it into a scratch directory, breaks it in one specific way at
# a time, and asserts that the checker catches each break.  It is deliberately
# negative: if a mutation slips through, the self-test FAILS.

SELF_TEST_CASES = [
    ("missing figure file", "file", "has no file at"),
    ("figure without a PDF sibling", "pdf", "vector PDF sibling"),
    ("unreferenced figure on disk", "extra", "not referenced by the report"),
    ("unbalanced brace", "brace", "unbalanced"),
    ("duplicate label", "label", "duplicate"),
    ("dangling reference", "ref", "does not resolve"),
    ("absolute path", "absolute", "absolute path"),
    ("claim marker with no ledger row", "claim", "has no such claim_id"),
    ("ledger row not referenced from the text", "orphan",
     "is not referenced from the text"),
    ("missing generated fragment", "input",
     "the report bundle is incomplete"),
    ("unbalanced inline math", "math", "unbalanced $"),
    ("unescaped underscore", "underscore", "unescaped _ outside math"),
]


def self_test(tex_path: Path, claims_path: Path) -> int:
    """Break the report on purpose and assert that the checker catches it."""
    import shutil
    import tempfile

    base_tex = read(tex_path)
    base_claims = claims_path.read_text(encoding="utf-8")
    failures: list[str] = []
    checked = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        figures_src = tex_path.parent / "figures"
        tables_src = tex_path.parent / "tables"

        def no_mutation(tex_text, claims_text, figures_dst):
            return tex_text, claims_text

        def run(tex_text: str, claims_text: str,
                mutate=no_mutation) -> list[str]:
            # A fresh copy of the figure directory for every case, so that one
            # case cannot mask another.  The mutation is applied AFTER the
            # copy, because it usually deletes or adds a figure file.
            figures_dst = tmpdir / "figures"
            if figures_dst.exists():
                shutil.rmtree(figures_dst)
            shutil.copytree(figures_src, figures_dst)
            # the report reads two generated fragments out of report/tables/,
            # so those travel with the copy as well
            tables_dst = tmpdir / "tables"
            if tables_src.is_dir():
                if tables_dst.exists():
                    shutil.rmtree(tables_dst)
                shutil.copytree(tables_src, tables_dst)
            tex_text, claims_text = mutate(tex_text, claims_text, figures_dst)
            t = tmpdir / "final_report.tex"
            t.write_text(tex_text, encoding="utf-8")
            c = tmpdir / "claims.csv"
            c.write_text(claims_text, encoding="utf-8")
            errors, _ = run_checks(t, c)
            return errors

        # The unmodified copy must pass, otherwise the self-test proves
        # nothing.
        errors = run(base_tex, base_claims)
        if errors:
            failures.append("the unmodified copy already fails: "
                             + "; ".join(errors[:3]))
        else:
            print("note: self-test baseline (unmodified copy) passes")

        def delete(name):
            def mutate(tex_text, claims_text, figures_dst):
                victim = figures_dst / name
                if victim.exists():
                    victim.unlink()
                return tex_text, claims_text
            return mutate

        def add_orphan(tex_text, claims_text, figures_dst):
            (figures_dst / "orphan_figure.png").write_bytes(b"")
            (figures_dst / "orphan_figure.pdf").write_bytes(b"")
            return tex_text, claims_text

        def brace(tex_text, claims_text, figures_dst):
            return tex_text.replace("\\end{document}", "}", 1), claims_text

        def label(tex_text, claims_text, figures_dst):
            return (tex_text.replace("\\label{sec:act4}\n",
                                     "\\label{sec:act4}\n"
                                     "\\label{sec:act4}\n", 1), claims_text)

        def ref(tex_text, claims_text, figures_dst):
            return (tex_text.replace("\\ref{sec:act4}",
                                     "\\ref{sec:no-such-label}", 1), claims_text)

        def absolute(tex_text, claims_text, figures_dst):
            return (tex_text.replace(
                "\\includegraphics[width=\\linewidth]"
                "{act4_scaling_ladder.png}",
                "\\includegraphics{/etc/passwd}", 1), claims_text)

        def claim(tex_text, claims_text, figures_dst):
            return (tex_text.replace("% claims: A4-07",
                                     "% claims: A4-07, A9-99", 1), claims_text)

        def orphan(tex_text, claims_text, figures_dst):
            return tex_text, claims_text + (
                "Z9-99,9,An orphan row,1,none,confirmed,"
                "results/v2/act9/interpretability.json,none\n")

        def missing_input(tex_text, claims_text, figures_dst):
            victim = tmpdir / "tables" / "act10_headline_claims.tex"
            if victim.exists():
                victim.unlink()
            return tex_text, claims_text

        def unbalanced_math(tex_text, claims_text, figures_dst):
            # remove the closing $ of one inline-math span, so that the rest
            # of the document is read in the wrong mode -- exactly the defect
            # that LaTeX reports as "Missing $ inserted".
            return tex_text.replace("$0.5295$ at mean confidence $0.5255$",
                                    "$0.5295 at mean confidence $0.5255$",
                                    1), claims_text

        def bare_underscore(tex_text, claims_text, figures_dst):
            return tex_text.replace("findings\\_taxonomy.json",
                                    "findings_taxonomy.json", 1), claims_text

        MUTATIONS = {
            "file": delete("act4_scaling_ladder.png"),
            "pdf": delete("act4_scaling_ladder.pdf"),
            "extra": add_orphan,
            "brace": brace,
            "label": label,
            "ref": ref,
            "absolute": absolute,
            "claim": claim,
            "orphan": orphan,
            "input": missing_input,
            "math": unbalanced_math,
            "underscore": bare_underscore,
        }

        for name, mutation, expect in SELF_TEST_CASES:
            errors = run(base_tex, base_claims, MUTATIONS[mutation])
            checked += 1
            if not any(expect in e for e in errors):
                failures.append(f"{name}: the checker did NOT report "
                                f"{expect!r}")
            else:
                print(f"note: self-test case {name!r} correctly caught")

    if failures:
        print(f"FAIL: the self-test found {len(failures)} problem(s)",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"OK: the self-test caught all {checked} deliberate break(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path,
                    default=REPO_ROOT / "report" / "final_report.tex")
    ap.add_argument("--claims", type=Path,
                    default=REPO_ROOT / "report" / "claims.csv")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="break the report on purpose and assert that this "
                         "checker catches each break")
    args = ap.parse_args()

    if not args.tex.exists():
        print(f"FAIL: {args.tex} does not exist", file=sys.stderr)
        return 1

    if args.self_test:
        return self_test(args.tex, args.claims)

    errors, notes = run_checks(args.tex, args.claims)

    if not args.quiet:
        for note in notes:
            print(f"note: {note}")
    if errors:
        print(f"FAIL: {len(errors)} problem(s) found", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("OK: report/final_report.tex passes every check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
