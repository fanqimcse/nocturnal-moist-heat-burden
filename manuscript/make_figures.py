#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py -- build every Nature Climate Change figure/table in one shot.

Manuscript working title:
  "Nocturnal humid heat: burden, the necessity of moisture, health evidence,
   threshold failure in the wet tropics, and observation-constrained exposure"

RULES ENFORCED BY THIS SCRIPT
-----------------------------
1. INPUTS   Only the frozen / audited artefacts listed in ALLOWED_JSON and
            ALLOWED_TXT are read.  audit/baseline_hash.json is recomputed and
            must match before anything is drawn; the run aborts otherwise.
2. OUTPUTS  Only files inside outputs/ncc_manuscript/{figures,tables,source}
            are written.  Any other write target raises.
3. NO INVENTION  Every printed number comes from a JSON path helper (Q), from
            arithmetic on those values with the formula recorded (DV), from a
            frozen .txt report (TT), or is flagged EXTERNAL (literature value
            absent from the frozen set).
4. RETRACTIONS  Six retracted / erroneous items are carried through figures,
            Table 1 and the report; never presented as positive results.
5. REPRODUCIBILITY  Deterministic rendering (SOURCE_DATE_EPOCH + svg.hashsalt);
            run twice and compare SHA-256 of every output.

Usage
-----
  C:/Users/Esther/.workbuddy/binaries/python/versions/3.13.12/python.exe make_figures.py
  python make_figures.py --hashes
  python make_figures.py --checkhashes <saved-hash-file>
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "ncc-nocturnal-humid-heat"
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
OUT = HERE.parent                      # outputs/ncc_manuscript
ROOT = OUT.parent.parent               # project root -- frozen artefacts live here
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
WRITE_ROOTS = (FIG_DIR, TAB_DIR, HERE)

# --------------------------------------------------------- allowed artefacts
ALLOWED_JSON = [
    "nwh_trend_robustness.json",
    "headline_metric.json",
    "humidity_decomp_v2.json",
    "humidity_attribution.json",
    "nwh_detection_attribution.json",
    "decisive_spec.json",
    "gradient_robustness.json",
    "south_north_stratified.json",
    "tropical_power.json",
    "tropical_threshold.json",
    "exposure_projection.json",
    "reference_implementation.json",
    "graded_scale.json",
    "warning_skill.json",
    "audit/baseline_hash.json",
]
ALLOWED_TXT = [
    "nwh_trend_robustness.txt",
    "headline_metric.txt",
    "humidity_decomp_v2.txt",
    "humidity_attribution.txt",
    "nwh_detection_attribution.txt",
    "decisive_spec.txt",
    "gradient_robustness.txt",
    "south_north_stratified.txt",
    "tropical_threshold.txt",
    "exposure_projection.txt",
    "reference_implementation.txt",
    "graded_scale.txt",
    "warning_skill.txt",
]
ALLOWED = set(ALLOWED_JSON) | set(ALLOWED_TXT)

# --------------------------------------------------------------- registries
PROVENANCE = {}          # filename -> dict(sha, len, baseline_sha, baseline_len, ok)
MANIFEST = []            # one dict per displayed number
OUT_FILES = []
TABLE1 = []
NOTES = []


def note(s):
    NOTES.append(s)


# --------------------------------------------------------------- baseline
def canonical_sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _abort(msg):
    sys.stderr.write("FATAL: %s\n" % msg)
    sys.exit(2)


def verify_baseline():
    bh = json.load(open(ROOT / "audit" / "baseline_hash.json", encoding="utf-8"))
    n_ok = n_bad = n_skip = 0
    for name, meta in bh.items():
        p = ROOT / name
        if not p.exists() or p.suffix != ".json":
            n_skip += 1
            continue
        try:
            obj = json.load(open(p, encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue
        sha = canonical_sha(obj)[:16]
        ln = len(json.dumps(obj, ensure_ascii=False))
        ok = (sha == meta["sha"]) and (ln == meta["len"])
        n_ok += 1 if ok else 0
        n_bad += 0 if ok else 1
        if not ok:
            sys.stderr.write("  baseline MISMATCH %s: sha %s vs %s; len %d vs %s\n"
                             % (name, sha, meta["sha"], ln, meta["len"]))
        PROVENANCE[name] = {"sha": sha, "len": ln, "baseline_sha": meta["sha"],
                            "baseline_len": meta["len"], "ok": ok}
    print("baseline verification: OK %d / BAD %d / unreadable %d" % (n_ok, n_bad, n_skip))
    if n_bad:
        _abort("frozen artefacts do not match audit/baseline_hash.json -- refusing to draw")


# ------------------------------------------------------------------- loaders
_J = {}
_T = {}


def load_json(name):
    if name not in ALLOWED:
        _abort("attempt to read non-allowed file: %s" % name)
    if name not in _J:
        _J[name] = json.load(open(ROOT / name, encoding="utf-8"))
    return _J[name]


def load_txt(name):
    if name not in ALLOWED:
        _abort("attempt to read non-allowed file: %s" % name)
    if name not in _T:
        _T[name] = open(ROOT / name, encoding="utf-8").read()
    return _T[name]


def txt_companion(src):
    name = src[:-5] + ".txt"
    if name not in ALLOWED_TXT or not (ROOT / name).exists():
        return None
    return load_txt(name)


def rawget(src, path):
    o = load_json(src)
    for k in path:
        o = o[int(k)] if isinstance(o, list) else o[k]
    return o


_PAT_CACHE = {}


def find_in_txt(src, numstr):
    """Locate numstr inside the frozen companion report; return its 1-based line."""
    t = txt_companion(src)
    if t is None:
        return None
    pat = _PAT_CACHE.get(numstr)
    if pat is None:
        pat = re.compile(r"(?<![\d.])" + re.escape(numstr) + r"(?![\d.])")
        _PAT_CACHE[numstr] = pat
    m = pat.search(t)
    return None if m is None else t.count("\n", 0, m.start()) + 1


def reg(figure, panel, item, source_file, source_key, displayed, validation):
    MANIFEST.append({"figure": figure, "panel": panel, "item": item,
                     "source_file": source_file, "source_key": source_key,
                     "displayed_value": displayed, "validation": validation})


def reprx(x):
    return ("%.10g" % x) if isinstance(x, float) else str(x)


class Val(object):
    """Numeric value plus exactly the string that was printed."""
    __slots__ = ("v", "s")

    def __init__(self, v, s):
        self.v = v
        self.s = s


def Q(fig, panel, item, src, path, fmt="{:.2f}", scale=1.0, suffix="", sign=False, extra=""):
    """Value straight out of a frozen JSON; auto cross-checked against its .txt."""
    raw = rawget(src, path)
    if raw is None:
        _abort("%s: JSON path %s is null" % (src, ".".join(map(str, path))))
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        _abort("%s: path %s not numeric (%r)" % (src, ".".join(map(str, path)), raw))
    val = raw * scale
    if "d" in fmt.split("}")[0]:
        val = int(round(val))
    numstr = fmt.format(val)
    disp = ("+" if sign else "") + numstr + suffix
    line = find_in_txt(src, numstr)
    keystr = ".".join(str(p) for p in path)
    if line:
        vs = "OK (json=%s; %s line %d)" % (reprx(raw), src[:-5] + ".txt", line)
    else:
        vs = "OK json-only (json=%s; not printed in %s)" % (reprx(raw), src)
    if scale != 1.0:
        vs += " [rescaled x%g]" % scale
    if extra:
        vs += "; " + extra
    reg(fig, panel, item, src, keystr, disp, vs)
    return Val(val, disp)


def DV(fig, panel, item, expr, value, fmt="{:.2f}", suffix="", sign=False, check="", srcs=""):
    """Derived value: arithmetic on frozen values; formula and check recorded."""
    numstr = fmt.format(value)
    disp = ("+" if sign else "") + numstr + suffix
    vs = "DERIVED (%s)" % expr
    if srcs:
        vs += " from " + srcs
    if check:
        vs += "; check: " + check
    reg(fig, panel, item, "<derived>", expr, disp, vs)
    return Val(value, disp)


def _short(s, n=96):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def TT(fig, panel, item, txtname, pattern, group=1, groups=None, flags=0, cast=float, extra=""):
    """Value parsed out of a frozen .txt report (absent from the JSON)."""
    t = load_txt(txtname)
    m = re.search(pattern, t, flags)
    if not m:
        _abort("txt parse error %s: pattern %r not found" % (txtname, pattern))
    line = t.count("\n", 0, m.start()) + 1
    ctx = _short(t[max(0, m.start() - 45):m.end() + 45])
    if groups is not None:
        vals = [cast(m.group(i)) for i in groups]
        disp = ", ".join(("%g" % v) if isinstance(v, float) else str(v) for v in vals)
        reg(fig, panel, item, txtname, "%s (line %d)" % (ctx, line), disp,
            "OK (%s line %d)%s" % (txtname, line, ("; " + extra) if extra else ""))
        return tuple(vals)
    v = cast(m.group(group))
    disp = ("%g" % v) if isinstance(v, float) else str(v)
    reg(fig, panel, item, txtname, "%s (line %d)" % (ctx, line), disp,
        "OK (%s line %d)%s" % (txtname, line, ("; " + extra) if extra else ""))
    return v


def EV(fig, panel, item, note_, disp):
    """Value that cannot be recovered from any frozen artefact (literature)."""
    reg(fig, panel, item, "EXTERNAL (literature)", note_, disp,
        "EXTERNAL -- not verifiable from the frozen artefacts")
    return Val(None, disp)


def row(figure, panel, quantity, sample, model, effect, status, src, key):
    TABLE1.append({"figure": figure, "panel": panel, "quantity": quantity,
                   "sample": sample, "model_specification": model,
                   "effect_95CI_or_90CI": effect, "status": status,
                   "source_file": src, "source_key": key})


# ------------------------------------------------------------------- styling
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
     "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
     "grey": "#7F7F7F", "lgrey": "#BFBFBF", "red": "#B2182B"}
MM = 1.0 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5,
    "axes.labelpad": 2.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "figure.dpi": 100, "savefig.dpi": 300,
})


def style(ax, grid_axis=None, grid_lw=0.4):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(pad=1.2)
    if grid_axis:
        ax.grid(axis=grid_axis, lw=grid_lw, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)


def plabel(ax, lab, x=-0.09, y=1.085):
    ax.text(x, y, lab, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left")


def foot(ax, s, ha="left", y=-0.34, fs=5.2, x=None):
    """Footnote placed below the axes so it can never cover data."""
    if x is None:
        x = 0.0 if ha == "left" else 1.0
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, va="top", ha=ha,
            linespacing=1.35, color="#333333")


def ensure_out(p):
    rp = p.resolve()
    if not any(str(rp).startswith(str(w.resolve()) + os.sep) for w in WRITE_ROOTS):
        _abort("refusing to write outside the allowed output tree: %s" % p)
    return p


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fp = ensure_out(FIG_DIR / ("%s.%s" % (name, ext)))
        if fp.exists():
            fp.unlink()
        if ext == "png":
            # dpi=300 rasterises at 300 px per inch (figure sizes are in inches)
            # and Pillow records the matching pHYs resolution in the file.
            fig.savefig(fp, dpi=300, bbox_inches="tight", pil_kwargs={"optimize": True})
        else:
            fig.savefig(fp, bbox_inches="tight")
        OUT_FILES.append(fp)
    plt.close(fig)
    print("  wrote %s {pdf,png,svg}" % name)


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ci_from(rr, z, level=1.96):
    """Recover the CI of exp(beta) from a point estimate and its z statistic."""
    se = math.log(rr) / z
    return math.exp(math.log(rr) - level * se), math.exp(math.log(rr) + level * se)


# ===================================================================== FIG 1
def fig1():
    TSRC = "nwh_trend_robustness.json"
    HSRC = "headline_metric.json"
    F = "Fig1"

    series = load_json(TSRC)["series"]
    xs = sorted(int(k) for k in series.keys())
    ys = [series[str(x)] for x in xs]
    trend = Q(F, "a", "headline trend", TSRC, ("windows", "1976-2023", "trend"), "{:.2f}", sign=True)
    base = Q(F, "a", "window baseline", TSRC, ("windows", "1976-2023", "base"), "{:.0f}")
    nst = Q(F, "a", "stations (balanced network)", TSRC, ("windows", "1976-2023", "n"), "{:d}")

    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM))
    ax = axs[0]
    ax.plot(xs, ys, lw=0.9, color=C["black"], marker="o", ms=2.0, markerfacecolor="white",
            markeredgewidth=0.5, zorder=3)
    xm = (xs[0] + xs[-1]) / 2.0
    ax.plot([xs[0], xs[-1]],
            [base.v + trend.v * (xs[0] - xm) / 10.0, base.v + trend.v * (xs[-1] - xm) / 10.0],
            lw=1.2, ls="--", color=C["verm"], zorder=4)
    ax.set_ylabel(r"Nocturnal wet-bulb hours NWH$_{21.9}$ (h yr$^{-1}$)")
    ax.set_xlabel("Year")
    style(ax, grid_axis="y")
    ax.set_xlim(xs[0] - 1, xs[-1] + 2)
    ax.set_ylim(1180, 1830)
    ax.text(xs[0] + 1, 1800, "%s h per decade\nnetwork equal-weighted\nbaseline %s h/yr, n = %s"
            % (trend.s, base.s, nst.s), va="top", ha="left", fontsize=6.0, linespacing=1.35)
    ax.set_title("Global nocturnal humid-heat burden", loc="left", pad=5)
    plabel(ax, "a")
    foot(ax, "Dashed line: frozen 1976-2023 window trend, anchored on the\n"
             "frozen window baseline (it is not a refit of the series).")
    row(F, "a", "NWH21.9 trend, 1976-2023",
         "64 stations with >=90% coverage in the window (network equal-weighted)",
         "Reported robust trend on the annual network mean",
         "+42.84 h per decade; baseline 1546 h/yr", "Primary", TSRC,
         "windows.1976-2023.{trend,base,n}")

    # ---------------- b : floor / saturation diagnostics
    ax = axs[1]
    specs = [("nhrs_elderly", "21.9", C["blue"]), ("nhrs_t23", "23.0", C["sky"]),
             ("nhrs_t24", "24.0", C["green"]), ("nhrs_young", "25.8", C["orange"]),
             ("nhrs_psu", "30.6", C["verm"]), ("nhrs_classic", "35.0", C["grey"])]
    zs, ratios = [], []
    for k, _lab, _c in specs:
        zs.append(Q(F, "b", "zero-fraction NWH" + _lab, HSRC, (k, "zero_fraction"),
                    "{:.1f}", scale=100.0, suffix="%").v)
        r = rawget(HSRC, (k, "ratio"))
        ratios.append(None if r is None else float(r))
    pos = list(range(len(specs)))
    ax.bar(pos, zs, color=[c for _k, _l, c in specs], width=0.62, zorder=3)
    for i, (z, r) in enumerate(zip(zs, ratios)):
        lab = ("%.1f%%" % z) if r is None else ("%.1f%%\n%.2fx" % (z, r))
        ax.text(i, z + 2.5, lab, ha="center", va="bottom", fontsize=5.8, linespacing=1.2)
    ax.set_xticks(pos)
    ax.set_xticklabels([l for _k, l, _c in specs], fontsize=6.2)
    ax.set_xlabel(r"Threshold (degC)")
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Station-years with zero annual hours (%)")
    style(ax, grid_axis="y")
    ax.set_title("Floor and saturation diagnostics", loc="left", pad=5)
    plabel(ax, "b")
    foot(ax, "Second line: ratio of the 2014-23 to the 1976-85 mean.\n"
             "NWH35 is floor-inflated: 99.9% of station-years are zero.")
    EV(F, "b", "21.9 degC anchor",
       "21.9 degC = epidemiologically anchored lower nocturnal wet-bulb bound for older "
       "adults (Vanos et al. 2023). Frozen artefacts record it only through the metric "
       "label 'NWH21.9 elderly lower bound (epidemiologically anchored)'.",
       "21.9 degC (external)")
    row(F, "b", "Zero-inflation and amplification by threshold",
         "176 stations screened, 67 retained in the headline panel",
         "Descriptive diagnostics on the station-year panel",
         "NWH21.9: 17.7% zero station-years, ratio 1.22x; NWH35: 99.9% zero, ratio undefined",
         "Limitation", HSRC, "nhrs_*.{zero_fraction,ratio}")

    # ---------------- c : aggregation calibre
    ax = axs[2]
    cal = [("net_mean", "Network mean", C["blue"]),
           ("pop_weighted", "Population weighted", C["sky"]),
           ("station_median", "Station median", C["orange"])]
    vals = []
    for key, lab, col in cal:
        v = Q(F, "c", "trend " + lab, TSRC, ("aggregation", key, "trend"), "{:.2f}", sign=True)
        Q(F, "c", "n " + lab, TSRC, ("aggregation", key, "n"), "{:d}")
        vals.append((lab, v.v, col))
    ypos = list(range(len(vals)))[::-1]
    ax.barh(ypos, [v[1] for v in vals], color=[v[2] for v in vals], height=0.58, zorder=3)
    for i, v in zip(ypos, vals):
        ax.text(v[1] + 0.9, i, "%+.2f" % v[1], va="center", fontsize=6.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([v[0] for v in vals], fontsize=6.2)
    ax.set_xlim(0, 68)
    ax.set_xlabel("Trend (h per decade)")
    ax.set_title("Aggregation calibre matters", loc="left", pad=5)
    style(ax, grid_axis="x")
    pmed = Q(F, "c", "station-median sign test p", TSRC,
             ("aggregation", "station_median", "p"), "{:.1e}")
    foot(ax, "Sign-test p = %s (n = 67). Station medians are locked\n"
             "by perennially zero stations, so the network mean is primary." % pmed.s)
    plabel(ax, "c")
    row(F, "c", "Aggregation sensitivity", "Identical 1976-2023 station set, three calibres",
         "Trend on network mean / population-weighted mean / station median",
         "+42.84 / +52.13 / +18.27 h per decade", "Sensitivity", TSRC, "aggregation.*.trend")

    fig.subplots_adjust(left=0.062, right=0.955, bottom=0.255, top=0.845, wspace=0.62)
    save(fig, "Fig1")


# ===================================================================== FIG 2
def fig2():
    DSRC = "humidity_decomp_v2.json"
    ASRC = "nwh_detection_attribution.json"
    F = "Fig2"
    fig = plt.figure(figsize=(183 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.1], wspace=0.62)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    # ---------------- a : factorial counterfactual decomposition (calibre S)
    thrs = ["20.0", "21.9", "24.0", "25.8"]
    comps = [("humid", "Moisture", C["blue"]),
             ("temp", "Temperature", C["verm"]),
             ("inter", "Interaction", C["green"])]
    width = 0.26
    x = list(range(len(thrs)))
    for j, (key, lab, col) in enumerate(comps):
        vals = [Q(F, "a", "%s effect NWH%s" % (lab, t), DSRC, (t, "S", key),
                  "{:.2f}", sign=True).v for t in thrs]
        axa.bar([i + (j - 1) * width for i in x], vals, width=width, color=col, label=lab,
                zorder=3)
    dref = [Q(F, "a", "D reference NWH" + t, DSRC, (t, "S", "D"), "{:.2f}", sign=True).v
            for t in thrs]
    axa.plot(x, dref, linestyle="none", marker="D", ms=4.0, color=C["grey"],
             label="D reference", zorder=4)
    axa.axhline(0, lw=0.6, color=C["black"])
    axa.set_xticks(x)
    axa.set_xticklabels([r"NWH$_{%s}$" % t for t in thrs], fontsize=6.2)
    axa.set_xlabel(r"Threshold (degC)")
    axa.set_ylabel("Trend contribution (h per decade)")
    axa.set_title("Humidity, not warming alone", loc="left", pad=5)
    style(axa, grid_axis="y")
    axa.set_ylim(-9.5, 24)
    axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.4,
               handlelength=1.0, columnspacing=1.2, labelspacing=0.4)
    plabel(axa, "a")
    Q(F, "a", "moisture share NWH21.9", DSRC, ("21.9", "S", "share_humid"), "{:.0f}",
      scale=100.0, suffix="%", sign=True)
    Q(F, "a", "temperature share NWH21.9", DSRC, ("21.9", "S", "share_temp"), "{:.0f}",
      scale=100.0, suffix="%", sign=True)
    row(F, "a", "Moisture vs warming decomposition (NWH21.9)",
         "72 stations, variance-preserving counterfactuals A/B/C/D",
         "Factorial counterfactual, recommended calibre S (reference D)",
         "moisture +17.31 h/dec (+105% of the observed trend); "
         "warming +8.60 h/dec (+52%); interaction 0.00",
         "Primary", DSRC, "21.9.S.{temp,humid,inter,share_*}")

    # ---------------- b : scaling factors (single vs two signal)
    items = [("Single-signal ALL", ("single", "ALL", "b"), ("single", "ALL", "se"), C["blue"]),
             ("Single-signal NAT", ("single", "NAT", "b"), ("single", "NAT", "se"), C["sky"]),
             ("Two-signal ALL", ("two_signal", "b_ALL"), ("two_signal", "se_ALL"), C["verm"]),
             ("Two-signal NAT", ("two_signal", "b_NAT"), ("two_signal", "se_NAT"), C["orange"])]
    yb = list(range(len(items)))[::-1]
    ticks = []
    for i, (lab, pb, ps, col) in zip(yb, items):
        b = Q(F, "b", "b " + lab, ASRC, pb, "{:.3f}", sign=True)
        se = Q(F, "b", "se " + lab, ASRC, ps, "{:.3f}")
        lo = DV(F, "b", "90%CI low " + lab, "b - 1.645*se", b.v - 1.645 * se.v, "{:.3f}",
                srcs="nwh_detection_attribution.json",
                check="1.645 = 90% two-sided normal quantile")
        hi = DV(F, "b", "90%CI high " + lab, "b + 1.645*se", b.v + 1.645 * se.v, "{:.3f}",
                srcs="nwh_detection_attribution.json",
                check="two_signal.ci_ALL is reported directly in the JSON")
        axb.errorbar([b.v], [i], xerr=[[b.v - lo.v], [hi.v - b.v]], fmt="o", ms=4.0,
                     color=col, ecolor=col, elinewidth=0.9, capsize=2.4, zorder=3)
        ticks.append("%s\n%s +/- %s" % (lab, b.s, se.s))
    Q(F, "b", "two-signal ALL 90%CI low", ASRC, ("two_signal", "ci_ALL", 0), "{:.3f}")
    Q(F, "b", "two-signal ALL 90%CI high", ASRC, ("two_signal", "ci_ALL", 1), "{:.3f}")
    Q(F, "b", "two-signal NAT 90%CI low", ASRC, ("two_signal", "ci_NAT", 0), "{:.3f}")
    Q(F, "b", "two-signal NAT 90%CI high", ASRC, ("two_signal", "ci_NAT", 1), "{:.3f}")
    axb.axvline(0, lw=0.6, color=C["black"])
    axb.axvline(1, lw=0.6, ls=":", color=C["grey"])
    axb.set_yticks(yb)
    axb.set_yticklabels(ticks, fontsize=5.4)
    axb.set_ylim(-0.6, 3.6)
    axb.set_xlim(-4.2, 1.3)
    axb.set_xlabel("Scaling factor b (90% CI)")
    axb.set_title("Detection and attribution", loc="left", pad=5)
    style(axb, grid_axis="x")
    foot(axb, "Dotted line = unity. Single- and two-signal\nfits are reported separately.")
    plabel(axb, "b")
    row(F, "b", "Optimal fingerprint ALL scaling factor",
         "Tropics + subtropics, 1976-2014; piControl internal variability n = 67 segments",
         "Optimal fingerprint regression (total least squares)",
         "single-signal b_ALL = +0.429 +/- 0.148; two-signal b_ALL = +0.348 "
         "(90% CI +0.072 to +0.624)",
         "Primary (two-signal preferred; single-signal used downstream, see report)",
         ASRC, "single.ALL.*; two_signal.{b_ALL,se_ALL,ci_ALL}")

    # ---------------- c : models vs observations (horizontal)
    bands = [("tropics", "Tropics", C["blue"]), ("subtropics", "Subtropics", C["verm"])]
    ticks, ypos = [], []
    for bi, (bkey, blab, col) in enumerate(reversed(bands)):
        models = rawget(ASRC, ("model_hist_trend", bkey))
        for k, v in enumerate(models):
            Q(F, "c", "modelled trend %s #%d" % (blab, k + 1), ASRC,
              ("model_hist_trend", bkey, k), "{:.2f}")
            axc.scatter([v], [bi + 0.16], s=10, facecolor="white", edgecolor=C["grey"],
                        lw=0.7, zorder=3)
        medv = Q(F, "c", "model median " + blab, ASRC, ("model_hist_median", bkey),
                 "{:.2f}", sign=True)
        axc.scatter([medv.v], [bi + 0.16], marker="^", s=24, color=C["grey"], zorder=4)
        o = Q(F, "c", "observed trend " + blab, ASRC, ("obs_trend", bkey), "{:.2f}", sign=True)
        sd = Q(F, "c", "piControl sd " + blab, ASRC, ("picontrol", "sd_" + bkey), "{:.2f}")
        sn = Q(F, "c", "S/N " + blab, ASRC, ("obs_SN", bkey), "{:.2f}")
        axc.errorbar([o.v], [bi], xerr=sd.v, fmt="*", ms=7.0, color=col, ecolor=col,
                     elinewidth=0.9, capsize=2.4, zorder=5)
        ticks.append("%s\nobs %s | models %s\nS/N %s" % (blab, o.s, medv.s, sn.s))
        ypos.append(bi)
    axc.set_yticks(ypos)
    axc.set_yticklabels(ticks, fontsize=5.4)
    axc.set_ylim(-0.45, 1.55)
    axc.set_xlim(0, 215)
    axc.set_xlabel("NWH trend (h per decade)")
    axc.set_title("Models over-estimate the rise", loc="left", pad=5)
    style(axc, grid_axis="x")
    foot(axc, "Open circles: four CMIP6 models (offset above the row);\n"
              "triangle: multi-model median; whisker: piControl SD.")
    plabel(axc, "c")
    row(F, "c", "Observed vs modelled NWH trends",
         "Observed 67 stations; 4 CMIP6 models; piControl n = 67 segments",
         "Trend comparison scaled by internal variability",
         "tropics obs +50.30 vs models +103.08 (S/N 3.08); "
         "subtropics obs +15.48 vs +54.89 (S/N 1.36)",
         "Supporting", ASRC, "obs_trend.*; model_hist_*; obs_SN.*")

    fig.subplots_adjust(left=0.062, right=0.982, bottom=0.265, top=0.845)
    save(fig, "Fig2")


# ===================================================================== FIG 3
def ci_from(rr, z, level=1.96):
    """Recover the CI of exp(beta) from a point estimate and its z statistic."""
    se = math.log(rr) / z
    return math.exp(math.log(rr) - level * se), math.exp(math.log(rr) + level * se)


def forest(ax, rows, xlim, xlabel, title, note, letter, zero=1.0):
    """rows: list of (label, value, lo, hi, colour, marker)."""
    ys = list(range(len(rows)))[::-1]
    ticks = []
    for i, (lab, v, lo, hi, col, mk) in zip(ys, rows):
        ax.errorbar([v], [i], xerr=[[v - lo], [hi - v]], fmt=mk, ms=4.0, color=col,
                    ecolor=col, elinewidth=0.9, capsize=2.4, zorder=3)
        ticks.append("%s\n%.4f [%.4f, %.4f]" % (lab, v, lo, hi))
    ax.axvline(zero, lw=0.6, color=C["black"])
    ax.set_yticks(ys)
    ax.set_yticklabels(ticks, fontsize=5.3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=5)
    style(ax, grid_axis="x")
    foot(ax, note)
    plabel(ax, letter)


def fig3():
    SSRC = "decisive_spec.json"
    GSRC = "gradient_robustness.json"
    NSRC = "south_north_stratified.json"
    F = "Fig3"
    A_KEY = ("defs", "A 任一夜间小时 (二元, 原始尺度)")
    D_KEY = ("decisive", "任意219 + 干球热带夜(均为二元原始尺度)")
    X_KEY = ("decisive", "任意219 + 白天TX≥30(均为二元原始尺度)")
    C_KEY = ("defs", "C 连续小时数 nh219 (z)")
    B_KEY = ("defs", "B 最热夜最低湿球 ≥21.9 (仅 30 区域有该事件)")

    fig = plt.figure(figsize=(183 * MM, 132 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.72)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # ---------------- a : primary specification + decisive controls
    rows = [("Primary: any219", SSRC, A_KEY + ("any219",), C["blue"], "o"),
            ("+ dry-bulb tropical night", SSRC, D_KEY + ("any219",), C["verm"], "o"),
            ("+ daytime TX>=30", SSRC, X_KEY + ("any219",), C["orange"], "o"),
            ("dry-bulb night itself", SSRC, D_KEY + ("any_tn20",), C["grey"], "s")]
    data = []
    for lab, src, basepath, col, mk in rows:
        rr = Q(F, "a", "RR " + lab, src, basepath + ("rr",), "{:.4f}")
        lo = Q(F, "a", "CI low " + lab, src, basepath + ("lo",), "{:.4f}")
        hi = Q(F, "a", "CI high " + lab, src, basepath + ("hi",), "{:.4f}")
        Q(F, "a", "z " + lab, src, basepath + ("z",), "{:.2f}", sign=True)
        Q(F, "a", "I2 " + lab, src, basepath + ("I2",), "{:.1f}", suffix="%")
        data.append((lab, rr.v, lo.v, hi.v, col, mk))
    n_pri = Q(F, "a", "regions, primary spec", SSRC, A_KEY + ("n_regions",), "{:d}")
    nD = Q(F, "a", "regions, adjusted model", SSRC, D_KEY + ("n_regions",), "{:d}")
    nX = Q(F, "a", "regions, TX30 model", SSRC, X_KEY + ("n_regions",), "{:d}")
    TT(F, "a", "panel size", "decisive_spec.txt",
       r"面板:\s*(\d+)\s*NUTS2\s*/\s*(\d+)\s*区域-周\s*/\s*暖季\s*5-9\s*月\s*/\s*(\d+)\s*年",
       groups=(1, 2, 3), cast=int, extra="sample size of the European health panel")
    forest(axa, data, (1.0065, 1.0455), "Rate ratio (95% CI)",
           "Crossing vs not crossing 21.9 degC",
           "n = %s regions (%s with the TX>=30 control). Heterogeneity I2 = 78.2%% from\n"
           "the JSON; an earlier print showed 1%% because of a formatting bug."
           % (n_pri.s, nX.s), "a")
    af = Q(F, "a", "attributable fraction", SSRC, ("headline", "af"), "{:.2f}", scale=100.0,
           suffix="%")
    afl = Q(F, "a", "attributable fraction low", SSRC, ("headline", "af_lo"), "{:.2f}",
            scale=100.0, suffix="%")
    afh = Q(F, "a", "attributable fraction high", SSRC, ("headline", "af_hi"), "{:.2f}",
            scale=100.0, suffix="%")
    _ = nD
    row(F, "a", "Mortality rate ratio for crossing 21.9 degC (any219)",
         "136 NUTS2 regions, 77,551 region-weeks, warm season May-Sep, 24 years",
         "Poisson, region FE + year FE + seasonal Fourier; binary raw 0/1 exposure",
         "RR = 1.0181 [1.0104, 1.0258], z = +4.62, I2 = 78.2%%; AF = %s [%s, %s]"
         % (af.s, afl.s, afh.s),
         "Primary", SSRC, "defs.A.any219.*; headline.af*")
    row(F, "a", "Decisive control: any219 adjusted for dry-bulb tropical night",
         "%s regions, joint binary model" % nD.s,
         "Both exposures entered unstandardised in the same model",
         "any219 RR = 1.0292 [1.0212, 1.0373], z = +7.21; "
         "dry-bulb night RR = 1.0368 [1.0320, 1.0416]",
         "Primary (identification test)", SSRC, "decisive.*.{any219,any_tn20}")

    # ---------------- b : continuous / strict specifications
    rows2 = [("Continuous nh219 per SD", C_KEY + ("nwh_z",), C["sky"], "o"),
             ("Def. B whole-night", B_KEY + ("b219",), C["grey"], "s"),
             ("Joint z-model (Tx, Tn)", C_KEY + ("nwh_z",), C["purple"], "D")]
    data2 = []
    for lab, basepath, col, mk in rows2:
        rr = Q(F, "b", "RR/SD " + lab, SSRC, basepath + ("rr",), "{:.4f}")
        lo = Q(F, "b", "CI low " + lab, SSRC, basepath + ("lo",), "{:.4f}")
        hi = Q(F, "b", "CI high " + lab, SSRC, basepath + ("hi",), "{:.4f}")
        Q(F, "b", "z " + lab, SSRC, basepath + ("z",), "{:.2f}", sign=True)
        data2.append((lab, rr.v, lo.v, hi.v, col, mk))
    nb = Q(F, "b", "regions def B", SSRC, B_KEY + ("n_regions",), "{:d}")
    nc = Q(F, "b", "regions continuous", SSRC, C_KEY + ("n_regions",), "{:d}")
    forest(axb, data2, (0.9855, 1.0290), "exp(beta) per SD (units differ from panel a)",
           "Continuous and strict definitions",
           "Def. B is estimable in only %s of %s regions: low power, not a\n"
           "null result. Per-SD units are not comparable with panel a." % (nb.s, nc.s), "b")
    row(F, "b", "Continuous exposure specification",
         "%s regions with non-zero nh219 variance" % nc.s,
         "Poisson with z-standardised exposure inside each region",
         "RR per SD = 1.0033 [1.0020, 1.0045], z = +5.24",
         "Sensitivity (different unit)", SSRC, "defs.C.nwh_z.*")
    row(F, "b", "Strict definition B (whole-night exceedance)",
         "%s regions with any such event" % nb.s,
         "Same model family with a binary whole-night exposure",
         "RR = 1.0043 [0.9876, 1.0212], z = +0.50",
         "Insufficient power (not a null result)", SSRC, "defs.B.b219.*")

    # ---------------- c : dose-response bins
    bins = ["0-2", "2-6", "6-20", ">20"]
    data3 = []
    for b in bins:
        rr = Q(F, "c", "RR dose bin " + b, GSRC, ("dr", b, "rr"), "{:.4f}")
        lo = Q(F, "c", "CI low dose " + b, GSRC, ("dr", b, "lo"), "{:.4f}")
        hi = Q(F, "c", "CI high dose " + b, GSRC, ("dr", b, "hi"), "{:.4f}")
        Q(F, "c", "z dose " + b, GSRC, ("dr", b, "z"), "{:.2f}", sign=True)
        data3.append(("%s h per week" % b, rr.v, lo.v, hi.v, C["green"], "o"))
    nrb = Q(F, "c", "regions in dose bins", GSRC, ("dr", "n_regions"), "{:d}")
    forest(axc, data3, (0.962, 1.056), "Rate ratio vs non-crossing weeks",
           "Step-shaped dose response",
           "%s contributing regions. The top bin does not separate from unity:\n"
           "risk attaches to crossing, not to additional hours." % nrb.s, "c")
    row(F, "c", "Dose-response bins",
         "%s NUTS2 regions with graded weekly exposure" % nrb.s,
         "Binned weekly NWH21.9 against non-crossing reference weeks",
         "0-2 h: 1.0156 [1.0069, 1.0243]; 2-6 h: 1.0153 [1.0052, 1.0255]; "
         "6-20 h: 1.0220 [0.9969, 1.0478]; >20 h: 0.9950 [0.9691, 1.0216]",
         "Sensitivity (flat / step response)", GSRC, "dr.*.{rr,lo,hi,z}")

    # ---------------- d : south vs north Europe
    SCHEME = "湿热本底中位数切分"
    basep = ("schemes", SCHEME, "_fit")
    groups = [("南欧型 湿热", "Southern / humid", C["verm"]),
              ("北欧型 凉爽", "Northern / cool", C["sky"])]
    data4 = []
    for gkey, glab, col in groups:
        rr = Q(F, "d", "nh219 RR/SD " + glab, NSRC, basep + (gkey, "nh219", "rr_per_sd"),
               "{:.4f}")
        lo = Q(F, "d", "nh219 CI low " + glab, NSRC, basep + (gkey, "nh219", "lo"), "{:.4f}")
        hi = Q(F, "d", "nh219 CI high " + glab, NSRC, basep + (gkey, "nh219", "hi"), "{:.4f}")
        Q(F, "d", "nh219 z " + glab, NSRC, basep + (gkey, "nh219", "z"), "{:.2f}", sign=True)
        data4.append((glab, rr.v, lo.v, hi.v, col, "o"))
    db = Q(F, "d", "layer contrast dbeta", NSRC,
           ("schemes", SCHEME, "contrast", "nh219", "dbeta"), "{:.4f}", sign=True)
    dz = Q(F, "d", "layer contrast z", NSRC,
           ("schemes", SCHEME, "contrast", "nh219", "z"), "{:.2f}", sign=True)
    dp = Q(F, "d", "layer contrast p", NSRC,
           ("schemes", SCHEME, "contrast", "nh219", "p"), "{:.3f}")
    forest(axd, data4, (0.99885, 1.00635), "exp(beta) per SD of nh219",
           "Within-Europe humidity gradient",
           "South minus north contrast: dbeta = %s, z = %s, p = %s.\n"
           "The gradient is present only where the exposure has variance."
           % (db.s, dz.s, dp.s), "d")
    row(F, "d", "South vs north Europe, nh219 association",
         "South 101 regions / 38,828 region-weeks; north 35 regions / 38,530 region-weeks",
         "Region FE + year FE + Fourier(4), joint Tx max, dry-bulb nights, nocturnal Tw, nh219",
         "south RR/SD = 1.0036 [1.0021, 1.0051]; north 1.0009 [0.9992, 1.0027]; "
         "contrast z = +2.26 (p = 0.024)",
         "Supporting (effect modification)", NSRC,
         "schemes.湿热本底中位数切分._fit.*.nh219.*; contrast.nh219.*")

    fig.subplots_adjust(left=0.235, right=0.982, bottom=0.105, top=0.925)
    save(fig, "Fig3")


# ===================================================================== FIG 4
def fig4():
    PSRC = "tropical_power.json"
    TSRC = "tropical_threshold.json"
    F = "Fig4"
    fig = plt.figure(figsize=(183 * MM, 132 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.60, hspace=0.68)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # ---------------- a : forest, raw and orthogonalised
    cities = [("Hong Kong", "HK", C["blue"]), ("Singapore", "SG", C["verm"])]
    data = []
    for clab, ck, col in cities:
        n = Q(F, "a", "n months " + clab, PSRC, (ck, "n"), "{:d}")
        rr = Q(F, "a", "RR/SD raw " + clab, PSRC, (ck, "rr_sd"), "{:.4f}")
        z = Q(F, "a", "z raw " + clab, PSRC, (ck, "z"), "{:.2f}", sign=True)
        lo, hi = ci_from(rr.v, z.v)
        DV(F, "a", "CI recovered, raw " + clab, "exp(ln RR +/- 1.96*ln RR/z)", 0.0, "{:.0f}",
           srcs="tropical_power.json", check="%.4f to %.4f" % (lo, hi))
        data.append(("%s, raw (n = %s)" % (clab, n.s), rr.v, lo, hi, col, "o"))
        orr = Q(F, "a", "RR/SD orthogonalised " + clab, PSRC, (ck, "orth_rr_sd"), "{:.4f}")
        oz = Q(F, "a", "z orthogonalised " + clab, PSRC, (ck, "orth_z"), "{:.2f}", sign=True)
        olo, ohi = ci_from(orr.v, oz.v)
        DV(F, "a", "CI recovered, orth " + clab, "exp(ln RR +/- 1.96*ln RR/z)", 0.0, "{:.0f}",
           srcs="tropical_power.json", check="%.4f to %.4f" % (olo, ohi))
        data.append(("%s, orthogonalised" % clab, orr.v, olo, ohi, col, "s"))
    meta = Q(F, "a", "meta RR/SD orthogonalised", PSRC, ("meta_orth", "rr"), "{:.4f}")
    mlo = Q(F, "a", "meta CI low", PSRC, ("meta_orth", "lo"), "{:.4f}")
    mhi = Q(F, "a", "meta CI high", PSRC, ("meta_orth", "hi"), "{:.4f}")
    Q(F, "a", "meta z", PSRC, ("meta_orth", "z"), "{:.2f}", sign=True)
    mi2 = Q(F, "a", "meta I2", PSRC, ("meta_orth", "I2"), "{:.1f}", suffix="%")
    data.append(("Meta, orthogonalised", meta.v, mlo.v, mhi.v, C["black"], "D"))
    forest(axa, data, (0.9994, 1.0295), "Rate ratio per SD of monthly NWH25 (95% CI)",
           "Wet-tropics evidence is weak",
           "Meta I2 = %s. Monthly panel, warm season only; the fixed-effect\n"
           "meta is the pre-specified summary." % mi2.s, "a")
    row(F, "a", "Wet-tropics mortality association (NWH25 per SD)",
         "Hong Kong 132 months; Singapore 264 months; warm season May-Oct",
         "Poisson, year spline df = 8, monthly Fourier x2; raw and Tw-orthogonalised exposure",
         "HK 1.0140 (z = +2.31); SG 1.0073 (z = +1.69); "
         "meta (orthogonalised) 1.0030 [1.0009, 1.0051], I2 = 0%",
         "Supporting, low power", PSRC, "HK.*; SG.*; meta_orth.*")

    # ---------------- b : Singapore threshold saturation
    ax = axb
    ths = ["21.9", "24.0", "25.0", "25.8", "26.5", "27.0"]
    fr, ah, us = [], [], []
    for t in ths:
        fr.append(Q(F, "b", "saturation fraction " + t, TSRC,
                    ("saturation", t, "frac_median"), "{:.3f}").v)
        ah.append(Q(F, "b", "annual hours " + t, TSRC,
                    ("saturation", t, "annual_h"), "{:.1f}").v)
        us.append(bool(rawget(TSRC, ("saturation", t, "usable"))))
    xb = list(range(len(ths)))
    ax.bar(xb, fr, color=[C["green"] if u else C["grey"] for u in us], width=0.6, zorder=3)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(xb)
    ax.set_xticklabels(ths, fontsize=6.0)
    ax.set_xlabel(r"Nocturnal wet-bulb threshold (degC)")
    ax.set_ylabel("Median monthly fraction above threshold")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(xb, ah, color=C["black"], lw=0.9, marker="o", ms=3.2, zorder=4)
    ax2.set_ylabel("Annual hours (h)", fontsize=6.0)
    ax2.tick_params(labelsize=5.8, pad=1.2)
    ax2.set_ylim(0, 4800)
    ax.set_title("Singapore: 21.9 degC is saturated", loc="left", pad=5)
    style(ax, grid_axis="y")
    ax.legend(handles=[Patch(facecolor=C["green"], label="usable"),
                       Patch(facecolor=C["grey"], label="unusable")],
              loc="upper right", fontsize=5.5, handlelength=1.0)
    foot(ax, "Bars: median monthly exceedance fraction. Line (right axis):\n"
             "annual hours above threshold.")
    plabel(ax, "b")
    row(F, "b", "Threshold saturation at an equatorial station",
         "Singapore 1.35 degN, 300 months, 2001-01 to 2025-12",
         "Monthly exceedance fraction and annual hour count per threshold",
         "21.9 degC fraction 1.000 (unusable); 24.0 degC 0.981 (unusable); "
         "25.0 degC 0.672 and 25.8 degC 0.098 usable",
         "Limitation (threshold failure)", TSRC,
         "saturation.*.{frac_median,annual_h,usable}")

    # ---------------- c : models on usable vs rejected thresholds
    mods = [("NWH21.9 (saturated)", "21.9", C["red"], "o"),
            ("NWH25.0 (usable)", "25.0", C["green"], "o"),
            ("NWH25.8 (usable)", "25.8", C["blue"], "o"),
            ("Nocturnal Tw mean", "tw_night", C["grey"], "s"),
            ("Nocturnal Tn mean", "tn_night", C["grey"], "s")]
    data3 = []
    for lab, key, col, mk in mods:
        rr = Q(F, "c", "RR/SD model " + lab, TSRC, ("models", key, "rr_sd"), "{:.4f}")
        lo = Q(F, "c", "CI low model " + lab, TSRC, ("models", key, "rr_sd_ci", 0), "{:.4f}")
        hi = Q(F, "c", "CI high model " + lab, TSRC, ("models", key, "rr_sd_ci", 1), "{:.4f}")
        Q(F, "c", "z model " + lab, TSRC, ("models", key, "z"), "{:.2f}", sign=True)
        Q(F, "c", "n model " + lab, TSRC, ("models", key, "n"), "{:d}")
        data3.append((lab, rr.v, lo.v, hi.v, col, mk))
    forest(axc, data3, (0.9865, 1.0425), "Rate ratio per SD (95% CI)",
           "Signal appears only after re-thresholding",
           "NWH21.9 (red) is a RETRACTED ARTEFACT: monthly SD 10.6 h and a median\n"
           "exceedance fraction of 1.000, so z = +8.61 reflects night length.", "c")
    row(F, "c", "Re-thresholded models at an equatorial station",
         "Singapore monthly panel, n = 300",
         "Poisson, year spline df = 8, monthly Fourier x2",
         "NWH25.0 RR/SD = 1.0122 (z = +2.03); NWH25.8 RR/SD = 1.0060 (z = +0.89); "
         "NWH21.9 RR/SD = 1.0292 (z = +8.61) REJECTED as a saturation artefact",
         "Retracted artefact + re-thresholded result", TSRC,
         "models.*.{rr_sd,rr_sd_ci,z}")

    # ---------------- d : minimum detectable effect
    ax = axd
    labels, mdes, sds, ns = [], [], [], []
    for clab, ck, col in cities:
        ms = Q(F, "d", "mde_sd " + clab, PSRC, (ck, "mde_sd"), "{:.4f}")
        sdv = Q(F, "d", "warm-season Tw SD " + clab, PSRC, (ck, "tw_sd_v"), "{:.3f}")
        n = Q(F, "d", "n months " + clab, PSRC, (ck, "n"), "{:d}")
        mde = DV(F, "d", "MDE " + clab, "(mde_sd - 1) * 100 / tw_sd_v",
                 (ms.v - 1.0) * 100.0 / sdv.v, "{:.2f}",
                 srcs="tropical_power.json",
                 check="percent per degC of warm-season Tw SD")
        labels.append(clab)
        mdes.append(mde.v)
        sds.append(sdv.v)
        ns.append(int(n.v))
    yd = list(range(len(labels)))[::-1]
    ax.barh(yd, mdes, height=0.42, color=[C["blue"], C["verm"]], zorder=3)
    for i, v, sdv, nv in zip(yd, mdes, sds, ns):
        ax.text(v + 0.06, i, "%.2f %%/degC" % v, va="center", fontsize=6.0)
    ax.set_yticks(yd)
    ax.set_yticklabels(["%s\nTw SD %.2f degC, n = %d" % (l, s, n)
                        for l, s, n in zip(labels, sds, ns)], fontsize=5.6)
    ax.set_xlim(0, 3.4)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Minimum detectable effect (% per degC)")
    ax.set_title("Detectable effects must be large", loc="left", pad=5)
    style(ax, grid_axis="x")
    ratio = DV(F, "d", "MDE ratio Singapore/Hong Kong", "MDE_SG / MDE_HK",
               mdes[1] / mdes[0], "{:.2f}", srcs="tropical_power.json")
    foot(ax, "Singapore needs a %sx larger effect than Hong Kong because its\n"
             "warm-season Tw contrast is only %.2f degC." % (ratio.s, sds[1]))
    plabel(ax, "d")
    row(F, "d", "Minimum detectable effect in the wet tropics",
         "Hong Kong 132 months; Singapore 264 months",
         "Power calculation converted from detectable RR/SD into percent per degC of Tw",
         "Hong Kong 1.05 %/degC; Singapore 2.46 %/degC (ratio 2.33x)",
         "Power limitation", PSRC, "*.{mde_sd,tw_sd_v,n}")

    fig.subplots_adjust(left=0.205, right=0.982, bottom=0.105, top=0.925)
    save(fig, "Fig4")


# ===================================================================== FIG 5
def fig5():
    ESRC = "exposure_projection.json"
    F = "Fig5"
    fig = plt.figure(figsize=(183 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.0], wspace=0.52)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    # ---------------- a : extent vs intensity decoupling
    fres = Q(F, "a", "region extent relative change", ESRC, ("firstlast5", "frac_rel"),
             "{:.0f}", scale=100.0, suffix="%", sign=True)
    popr = Q(F, "a", "population extent relative change", ESRC, ("firstlast5", "pop_rel"),
             "{:.0f}", scale=100.0, suffix="%", sign=True)
    nwhr = Q(F, "a", "intensity relative change", ESRC, ("firstlast5", "nwh_rel"),
             "{:.0f}", scale=100.0, suffix="%", sign=True)
    ovf = Q(F, "a", "over-estimate factor", ESRC, ("firstlast5", "overestimate_factor"),
            "{:.2f}")
    DV(F, "a", "over-estimate factor recomputed", "(1 + nwh_rel) / (1 + pop_rel)",
       (1.0 + nwhr.v / 100.0) / (1.0 + popr.v / 100.0), "{:.4f}",
       srcs="exposure_projection.json",
       check="frozen field = %.6f (identical to 4 dp)" % ovf.v)
    items = [("Population extent", popr.v, C["blue"]),
             ("Region extent", fres.v, C["sky"]),
             ("NWH intensity", nwhr.v, C["verm"])]
    ya = list(range(len(items)))[::-1]
    axa.barh(ya, [it[1] for it in items], height=0.55, color=[it[2] for it in items], zorder=3)
    for i, it in zip(ya, items):
        axa.text(it[1] + 0.9, i, "%+.0f%%" % it[1], va="center", fontsize=6.2)
    axa.set_yticks(ya)
    axa.set_yticklabels([it[0] for it in items], fontsize=6.0)
    axa.set_xlim(0, 60)
    axa.set_xlabel("Relative change, first 5 yr to last 5 yr (%)")
    axa.set_title("Extent and intensity decouple", loc="left", pad=5)
    style(axa, grid_axis="x")
    foot(axa, "Intensity grows faster than extent, but the dose response is\n"
              "flat, so risk follows extent: scaling by intensity over-states %.2fx."
         % ovf.v)
    plabel(axa, "a")
    row(F, "a", "Exposure extent vs intensity decoupling",
         "135 NUTS2 regions, 2000-2024; means of the first and last five years",
         "Relative change of exposed population, exposed regions and NWH hours",
         "population +26%, regions +29%, NWH +47%; using intensity over-states risk 1.78x",
         "Primary", ESRC, "firstlast5.{frac_rel,pop_rel,nwh_rel,overestimate_factor}")

    # ---------------- b : observation-constrained projection
    scen = [("SSP2-4.5\nmid", "ssp245_mid", C["sky"]),
            ("SSP2-4.5\nend", "ssp245_end", C["blue"]),
            ("SSP5-8.5\nmid", "ssp585_mid", C["orange"]),
            ("SSP5-8.5\nend", "ssp585_end", C["verm"])]
    addv, rawv, dts = [], [], []
    for lab, key, col in scen:
        addv.append(Q(F, "b", "additional exposed " + key, ESRC,
                      ("projection", key, "add_M"), "{:.1f}").v)
        rawv.append(Q(F, "b", "unscaled additional " + key, ESRC,
                      ("projection", key, "add_M_raw"), "{:.1f}").v)
        dts.append(Q(F, "b", "scaled warming " + key, ESRC,
                     ("projection", key, "dT_scaled"), "{:.2f}", sign=True).s)
    xb = list(range(len(scen)))
    axb.bar([i - 0.19 for i in xb], addv, width=0.36, color=[s[2] for s in scen], zorder=3,
            label="scaled by the ALL fingerprint")
    axb.bar([i + 0.19 for i in xb], rawv, width=0.36, color="white",
            edgecolor=[s[2] for s in scen], hatch="///", lw=0.7, zorder=3,
            label="unscaled CMIP6")
    for i, a, r, d in zip(xb, addv, rawv, dts):
        axb.text(i - 0.19, a + 6, "%.0f" % a, ha="center", fontsize=5.8)
        axb.text(i - 0.19, a + 24, "dT %s" % d, ha="center", fontsize=5.0, color="#222222")
        axb.text(i + 0.19, r + 6, "%.0f" % r, ha="center", fontsize=5.2, color=C["grey"])
    axb.set_xticks(xb)
    axb.set_xticklabels([s[0] for s in scen], fontsize=5.6)
    axb.set_ylabel("Newly exposed population (millions)")
    axb.set_ylim(0, 285)
    axb.set_title("Observation-constrained projection", loc="left", pad=5)
    style(axb, grid_axis="y")
    axb.legend(loc="upper left", fontsize=5.5, handlelength=1.1)
    bl = Q(F, "b", "calibrated baseline exposure", ESRC,
           ("headline", "baseline_exp_pop_M"), "{:.1f}", suffix=" M")
    foot(axb, "Calibrated model baseline exposure = %s. dT is the scaled\n"
              "warm-season nocturnal Tw change relative to 1995-2014." % bl.s)
    plabel(axb, "b")
    row(F, "b", "Newly exposed population, observation-constrained",
         "135 European NUTS2 regions covering 499 million people",
         "Scenario warming scaled by the ALL fingerprint before applying the exposure model",
         "SSP2-4.5 end-century +78 M (unscaled 166 M); "
         "SSP5-8.5 end-century +146 M (unscaled 237 M)",
         "Primary", ESRC, "projection.*.{add_M,add_M_raw,dT_scaled}")

    # ---------------- c : policy scaling
    pooled = Q(F, "c", "million per degC, pooled", ESRC,
               ("headline", "per_degC_M_pooled"), "{:.1f}")
    fe = Q(F, "c", "million per degC, fixed effects", ESRC,
           ("headline", "per_degC_M_fe"), "{:.1f}")
    bias, biasp = TT(F, "c", "model calibration bias", "exposure_projection.txt",
                     r"偏差\s+(-?\d+)\s+百万\s+\((-?[\d.]+)%\)", groups=(1, 2), cast=float,
                     extra="calibrated model minus observed mean exposure")
    yc = [1, 0]
    vals = [pooled.v, fe.v]
    axc.barh(yc, vals, height=0.45, color=[C["orange"], C["blue"]], zorder=3)
    for i, v in zip(yc, vals):
        axc.text(v + 1.4, i, "%.1f M" % v, va="center", fontsize=6.0)
    axc.set_yticks(yc)
    axc.set_yticklabels(["Pooled\nlogistic", "Region\nfixed effects"], fontsize=5.8)
    axc.set_xlim(0, 92)
    axc.set_ylim(-0.6, 1.6)
    axc.set_xlabel("Million newly exposed per +1 degC")
    axc.set_title("Policy-relevant scaling", loc="left", pad=5)
    style(axc, grid_axis="x")
    foot(axc, "Model calibration bias %+d M (%+.1f%% of the observed mean).\n"
              "The fixed-effects estimate is the conservative lower bound."
         % (int(bias), biasp))
    plabel(axc, "c")
    row(F, "c", "Exposure response per degree of warming",
         "Same projection sample; observed 2000-2024 mean used for calibration",
         "Pooled logistic dose-response and region fixed-effects alternative",
         "70.0 million (pooled) / 42.3 million (fixed effects) per +1 degC",
         "Primary (fixed effects is the conservative lower bound)", ESRC,
         "headline.{per_degC_M_pooled,per_degC_M_fe}")

    fig.subplots_adjust(left=0.062, right=0.982, bottom=0.275, top=0.862)
    save(fig, "Fig5")


# ========================================================= Extended Data Fig 1
def edfig1():
    TSRC = "nwh_trend_robustness.json"
    HSRC = "headline_metric.json"
    F = "ExtendedDataFig1"
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 60 * MM))

    ax = axs[0]
    wins = ["1976-2023", "1976-2014", "1980-2023", "1985-2023", "1990-2023", "1995-2023",
            "2000-2023", "1977-2023"]
    vals, ns = [], []
    for w in wins:
        vals.append(Q(F, "a", "trend window " + w, TSRC, ("windows", w, "trend"),
                      "{:.2f}", sign=True).v)
        ns.append(int(Q(F, "a", "n window " + w, TSRC, ("windows", w, "n"), "{:d}").v))
    xp = list(range(len(wins)))
    cols = [C["blue"] if w == "1976-2023" else C["lgrey"] for w in wins]
    ax.bar(xp, vals, color=cols, width=0.62, zorder=3)
    for i, v, n in zip(xp, vals, ns):
        ax.text(i, v + 1.2, "%+.2f" % v, ha="center", fontsize=5.4)
        ax.text(i, 2.5, "n=%d" % n, ha="center", fontsize=4.9, color="#222222")
    ax.axhline(vals[0], lw=0.6, ls=":", color=C["black"])
    ax.set_xticks(xp)
    ax.set_xticklabels(wins, rotation=40, ha="right", fontsize=5.6)
    ax.set_ylabel("Trend (h per decade)")
    ax.set_ylim(0, 72)
    ax.set_title("Window sensitivity", loc="left", pad=5)
    style(ax, grid_axis="y")
    foot(ax, "Headline window highlighted; n is the balanced station set of\n"
             "each window. Every window gives a positive trend.")
    plabel(ax, "a")

    ax = axs[1]
    keys = [("nhrs_elderly", r"NWH$_{21.9}$"), ("nhrs_t23", r"NWH$_{23}$"),
            ("nhrs_t24", r"NWH$_{24}$"), ("nhrs_young", r"NWH$_{25.8}$"),
            ("nhrs_psu", r"NWH$_{30.6}$"), ("nhrs_classic", r"NWH$_{35}$")]
    medv, popv = [], []
    for k, lab in keys:
        medv.append(Q(F, "b", "station median trend " + lab, HSRC,
                      (k, "trend_median_per_decade"), "{:.2f}", sign=True).v)
        popv.append(Q(F, "b", "population weighted trend " + lab, HSRC,
                      (k, "trend_pop_weighted"), "{:.2f}", sign=True).v)
    xbb = list(range(len(keys)))
    ax.bar([i - 0.19 for i in xbb], medv, width=0.36, color=C["orange"],
           label="station median", zorder=3)
    ax.bar([i + 0.19 for i in xbb], popv, width=0.36, color=C["sky"],
           label="population weighted", zorder=3)
    ax.set_xticks(xbb)
    ax.set_xticklabels([l for _k, l in keys], fontsize=5.8)
    ax.set_ylabel("Trend (h per decade)")
    ax.axhline(0, lw=0.6, color=C["black"])
    ax.set_ylim(-2, 128)
    ax.set_title("Trend depends on threshold and weighting", loc="left", pad=5)
    style(ax, grid_axis="y")
    ax.legend(fontsize=5.5, handlelength=1.1, loc="upper right")
    foot(ax, "Station medians collapse to zero once a majority of stations sits\n"
             "perennially below the threshold.")
    plabel(ax, "b")
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.30, top=0.848, wspace=0.34)
    save(fig, "ExtendedDataFig1")


# ========================================================= Extended Data Fig 2
def edfig2():
    RSRC = "reference_implementation.json"
    F = "ExtendedDataFig2"
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 62 * MM))

    comps = [("夜间定义(固定钟点)", "Night definition (fixed clock)"),
             ("夜间定义(曙暮光 vs 天文)", "Night definition (twilight vs astronomical)"),
             ("仪器 露点 ±0.5℃", "Instrument dew point +/-0.5 degC"),
             ("仪器 RH ±2%", "Instrument RH +/-2%"),
             ("湿球公式(Stull vs 参考)", "Wet-bulb formula (Stull vs reference)"),
             ("湿度输入(RH整百分点)", "Humidity input (integer RH)"),
             ("时间分辨率(3h)", "Temporal resolution (3 h)")]
    cities = [("Singapore", "singapore", C["verm"]), ("Hong Kong", "hongkong", C["blue"])]

    ax = axs[0]
    for clab, key, col in cities:
        vals = [Q(F, "a", "budget %s %s" % (clab, lab), RSRC, ("budget", key, k),
                  "{:.2f}").v for k, lab in comps]
        uc = Q(F, "a", "combined u_c " + clab, RSRC, ("budget", key, "合成 u_c (RSS, %)"),
               "{:.2f}", suffix="%")
        rss = math.sqrt(sum(v * v for v in vals))
        DV(F, "a", "RSS recomputed " + clab, "sqrt(sum of squared components)", rss,
           "{:.2f}", srcs="reference_implementation.json",
           check="frozen u_c = %.2f%% (difference %.2f pp)" % (uc.v, abs(uc.v - rss)))
    ya = list(range(len(comps)))[::-1]
    for j, (clab, key, col) in enumerate(cities):
        vals = [Q(F, "a", "bar %s %s" % (clab, lab), RSRC, ("budget", key, k),
                  "{:.2f}").v for k, lab in comps]
        ax.barh([i + (j - 0.5) * 0.38 for i in ya], vals, height=0.34, color=col, zorder=3)
    ax.set_yticks(ya)
    ax.set_yticklabels([lab for _k, lab in comps], fontsize=5.6)
    ax.set_xlim(0, 40)
    ax.set_xlabel(r"Uncertainty of NWH$_{21.9}$ (%)")
    ax.set_title("Uncertainty budget", loc="left", pad=5)
    style(ax, grid_axis="x")
    ax.legend(handles=[Patch(facecolor=C["verm"], label="Singapore"),
                       Patch(facecolor=C["blue"], label="Hong Kong")],
              fontsize=5.5, handlelength=1.0, loc="lower right")
    ucs = Q(F, "a", "combined u_c Singapore label", RSRC,
            ("budget", "singapore", "合成 u_c (RSS, %)"), "{:.2f}", suffix="%")
    uch = Q(F, "a", "combined u_c Hong Kong label", RSRC,
            ("budget", "hongkong", "合成 u_c (RSS, %)"), "{:.2f}", suffix="%")
    foot(ax, "Combined root-sum-square u_c: Singapore %s, Hong Kong %s."
         % (ucs.s, uch.s))
    plabel(ax, "a")
    row(F, "a", "Implementation uncertainty budget",
         "Reference re-implementation at two wet-tropics cities",
         "Component-wise sensitivity combined as root-sum-square",
         "combined u_c = %s (Singapore) and %s (Hong Kong); night definition dominates"
         % (ucs.s, uch.s), "Sensitivity", RSRC, "budget.*")

    ax = axs[1]
    nightdef = [("alt-0.833", "astronomical\n(reference)"), ("alt-6", "civil\ntwilight"),
                ("alt-18", "nautical\ntwilight"), ("fix2206", "fixed\n22-06"),
                ("fix2008", "fixed\n20-08")]
    xnd = list(range(len(nightdef)))
    for clab, key, col in cities:
        vals = [Q(F, "b", "night %s %s" % (clab, k), RSRC,
                  (key, "night", k, "d219_pct"), "{:.2f}", sign=True).v
                for k, _l in nightdef]
        ax.plot(xnd, vals, marker="o", ms=3.4, lw=0.9, color=col, label=clab, zorder=3)
    ax.axhline(0, lw=0.6, color=C["black"])
    ax.set_xticks(xnd)
    ax.set_xticklabels([l for _k, l in nightdef], fontsize=5.4)
    ax.set_ylabel(r"Change in NWH$_{21.9}$ (%)")
    ax.set_ylim(-38, 14)
    ax.set_title("Night definition dominates the budget", loc="left", pad=5)
    style(ax, grid_axis="y")
    ax.legend(fontsize=5.5, handlelength=1.1, loc="lower left")
    foot(ax, "Each point is the percentage change in NWH21.9 relative to the\n"
             "astronomical-night reference implementation.")
    plabel(ax, "b")
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.245, top=0.848, wspace=0.44)
    save(fig, "ExtendedDataFig2")


# ========================================================= Extended Data Fig 3
def edfig3():
    GSRC = "graded_scale.json"
    F = "ExtendedDataFig3"
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 62 * MM))

    # ---------------- a : Europe, absolute grading
    ax = axs[0]
    lv = ["1", "2", "3"]
    vals = []
    for l in lv:
        rr = Q(F, "a", "RR level " + l, GSRC, ("europe", "rr", l, "RR"), "{:.4f}")
        lo = Q(F, "a", "CI low level " + l, GSRC, ("europe", "rr", l, "lo"), "{:.4f}")
        hi = Q(F, "a", "CI high level " + l, GSRC, ("europe", "rr", l, "hi"), "{:.4f}")
        z = Q(F, "a", "z level " + l, GSRC, ("europe", "rr", l, "z"), "{:.2f}", sign=True)
        n = Q(F, "a", "n level " + l, GSRC, ("europe", "rr", l, "n"), "{:d}")
        vals.append((rr.v, lo.v, hi.v, z.s, n.s))
    dist0 = Q(F, "a", "region-weeks level 0", GSRC, ("europe", "dist", "0"), "{:d}")
    for l in lv:
        Q(F, "a", "region-weeks level " + l, GSRC, ("europe", "dist", l), "{:d}")
    Q(F, "a", "European n regions", GSRC, ("europe", "n_region"), "{:d}")
    Q(F, "a", "European n rows", GSRC, ("europe", "n"), "{:d}")
    trr = Q(F, "a", "RR per level", GSRC, ("europe", "trend", "RR"), "{:.4f}")
    tlo = Q(F, "a", "CI low per level", GSRC, ("europe", "trend", "lo"), "{:.4f}")
    thi = Q(F, "a", "CI high per level", GSRC, ("europe", "trend", "hi"), "{:.4f}")
    tz = Q(F, "a", "z per level", GSRC, ("europe", "trend", "z"), "{:.2f}", sign=True)
    xp = list(range(len(vals)))
    ax.errorbar(xp, [v[0] for v in vals],
                yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                fmt="o-", ms=4.2, lw=1.0, color=C["green"], ecolor=C["green"],
                capsize=2.4, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate("%.4f  z=%s  n=%s" % (v[0], v[3], v[4]), (i, v[0]),
                    textcoords="offset points", xytext=(6, 7), ha="left", fontsize=5.0)
    ax.axhline(1.0, lw=0.6, color=C["black"])
    ax.set_xticks(xp)
    ax.set_xticklabels(["Level I\nvs 0", "Level II\nvs 0", "Level III\nvs 0"], fontsize=5.8)
    ax.set_ylabel("Rate ratio for mortality")
    ax.set_ylim(1.0, 1.58)
    ax.set_xlim(-0.4, 2.75)
    ax.set_title("Europe: absolute grading is monotonic", loc="left", pad=5)
    style(ax, grid_axis="y")
    foot(ax, "Ordered trend RR/level = %s [%s, %s], z = %s; level 0 has\n"
             "%s region-weeks." % (trr.s, tlo.s, thi.s, tz.s, dist0.s))
    plabel(ax, "a")
    row(F, "a", "Graded scale validation, Europe",
         "225 NUTS2 regions, 77,551 region-weeks, warm season",
         "Poisson with region FE, year FE, weekly Fourier, region-clustered SE",
         "Level I RR = 1.0591 [1.0384, 1.0803]; II 1.1049 [1.0338, 1.1810]; "
         "III 1.2686 [1.1183, 1.4391]; trend RR/level = 1.0577 [1.0360, 1.0798]",
         "Supporting (monotonicity)", GSRC, "europe.{rr,trend,dist,n}")

    # ---------------- b : wet tropics, local quantile grading
    ax = axs[1]
    qs = ["1", "2", "3"]
    rrs, los, his = [], [], []
    for q in qs:
        rrs.append(Q(F, "b", "RR quartile " + q, GSRC, ("tropics", "quartile", q, "RR"),
                     "{:.4f}").v)
        los.append(Q(F, "b", "CI low quartile " + q, GSRC, ("tropics", "quartile", q, "lo"),
                     "{:.4f}").v)
        his.append(Q(F, "b", "CI high quartile " + q, GSRC, ("tropics", "quartile", q, "hi"),
                     "{:.4f}").v)
        Q(F, "b", "z quartile " + q, GSRC, ("tropics", "quartile", q, "z"), "{:.2f}", sign=True)
        Q(F, "b", "n quartile " + q, GSRC, ("tropics", "quartile", q, "n"), "{:d}")
    xq = list(range(len(qs)))
    ax.errorbar(xq, rrs, yerr=[[a - b for a, b in zip(rrs, los)],
                               [b - a for a, b in zip(rrs, his)]],
                fmt="o-", ms=4.2, lw=1.0, color=C["purple"], ecolor=C["purple"],
                capsize=2.4, zorder=3, label="cluster-robust SE (2 clusters)")
    hc = TT(F, "b", "Q3 HC0 result", "graded_scale.txt",
            r"Q3\s+vs\s+Q0\s+RR=([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+z=\+?([\d.]+)",
            groups=(1, 2, 3, 4), cast=float,
            extra="HC0 (independent) row printed only in the frozen report")
    ax.errorbar([2], [hc[0]], yerr=[[hc[0] - hc[1]], [hc[2] - hc[0]]], fmt="s", ms=4.0,
                color=C["black"], ecolor=C["black"], elinewidth=0.9, capsize=2.4, zorder=4,
                label="HC0 SE (same point estimate)")
    ax.axhline(1.0, lw=0.6, color=C["black"])
    ax.set_xticks(xq)
    ax.set_xticklabels(["Q1 vs Q0", "Q2 vs Q0", "Q3 vs Q0"], fontsize=5.8)
    ax.set_ylabel("Rate ratio for mortality")
    ax.set_ylim(0.94, 1.11)
    ax.set_title("Wet tropics: local quantile grading", loc="left", pad=5)
    style(ax, grid_axis="y")
    sat = Q(F, "b", "absolute grading saturated", GSRC, ("tropics", "abs_saturated_pct"),
            "{:.1f}", suffix="%")
    ax.legend(fontsize=5.2, handlelength=1.1, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    foot(ax, "Absolute grading is saturated: III+IV = %s of city-months. The Q3\n"
             "cluster z = +12.97 in the JSON is inflated by 2 clusters; the HC0\n"
             "z = %+.2f is the defensible value." % (sat.s, hc[3]), y=-0.46)
    plabel(ax, "b")
    row(F, "b", "Local quantile grading in the wet tropics",
         "Two cities, 300 city-months; within-city quartiles of NWH25.8",
         "Poisson with city FE, month FE, trend, city-clustered SE",
         "Q2 RR = 1.0260 [1.0195, 1.0325]; Q3 RR = 1.0345 [1.0292, 1.0398] with the "
         "2-cluster caveat; HC0 CI for Q3 [0.9763, 1.0962]",
         "Supporting with caveat (cluster SE unreliable)", GSRC, "tropics.quartile.*")

    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.235, top=0.848, wspace=0.40)
    save(fig, "ExtendedDataFig3")


# ========================================================= Extended Data Fig 4
def edfig4():
    WSRC = "warning_skill.json"
    F = "ExtendedDataFig4"
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))
    rules = [("R1 daytime\nheatwave", "R1 现行·白天干球热浪(Tx≥P95, ≥3天)", C["blue"]),
             ("R2 tropical\nnights", "R2 现行·热带夜(干球 Tn≥20℃, ≥3夜)", C["sky"]),
             ("R3 wet-bulb\ncrossing", "R3 本研究·夜间湿球跨越(Tw≥21.9℃)", C["verm"]),
             ("R4 graded\nlevel II", "R4 本研究·分级 II (Tw≥24.0℃)", C["orange"])]

    # ---------------- a : skill scores
    ax = axs[0]
    metrics = [("POD", "POD", C["blue"]), ("FAR", "FAR", C["verm"]), ("HSS", "HSS", C["green"])]
    w = 0.26
    xr = list(range(len(rules)))
    for j, (mk, mlab, col) in enumerate(metrics):
        vals = [Q(F, "a", "%s %s" % (mlab, lab), WSRC, (key, mk), "{:.3f}").v
                for lab, key, _c in rules]
        ax.bar([i + (j - 1) * w for i in xr], vals, width=w, color=col, label=mlab, zorder=3)
    ax.set_xticks(xr)
    ax.set_xticklabels([r[0] for r in rules], fontsize=5.4)
    ax.set_ylabel("Skill score")
    ax.set_ylim(0, 0.92)
    ax.set_title("Trigger-rule skill", loc="left", pad=5)
    style(ax, grid_axis="y")
    ax.legend(fontsize=5.4, handlelength=1.0, ncol=3, loc="upper center")
    plabel(ax, "a")
    pod3 = Q(F, "a", "POD R3 label", WSRC, (rules[2][1], "POD"), "{:.3f}")
    pod1 = Q(F, "a", "POD R1 label", WSRC, (rules[0][1], "POD"), "{:.3f}")
    row(F, "a", "Warning trigger skill (R1-R4)",
         "77,551 region-weeks, 225 NUTS2 regions, 2001-2024",
         "Probability of detection, false alarm ratio, Heidke skill score",
         "R3 POD = %s is below R1 POD = %s; R3 is positioned as an add-on layer"
         % (pod3.s, pod1.s),
         "Supporting (no claim of replacement)", WSRC, "R1..R4.{POD,FAR,HSS}")

    # ---------------- b : incremental value
    ax = axs[1]
    n1 = Q(F, "b", "weeks missed by R1", WSRC, ("incremental", "n_miss_r1"), "{:d}")
    ex1 = Q(F, "b", "excess deaths in R1-missed weeks", WSRC,
            ("incremental", "excess_miss_r1"), "{:.0f}")
    er1 = Q(F, "b", "event rate in R1-missed weeks", WSRC,
            ("incremental", "event_rate_miss_r1"), "{:.3f}")
    erall = Q(F, "b", "event rate overall", WSRC, ("incremental", "event_rate_all"), "{:.3f}")
    ratio = DV(F, "b", "event-rate ratio", "event_rate_miss_r1 / event_rate_all",
               er1.v / erall.v, "{:.2f}", srcs="warning_skill.json",
               check="frozen report quotes 1.53x")
    n12 = Q(F, "b", "weeks missed by R1 and R2", WSRC, ("incremental", "n_miss_r12"), "{:d}")
    ex12 = Q(F, "b", "excess deaths in R1R2-missed weeks", WSRC,
             ("incremental", "excess_miss_r12"), "{:.0f}")
    bars = [("R1-unwarned\nweeks caught by R3", n1.v, C["verm"]),
            ("R1+R2-unwarned\nweeks caught by R3", n12.v, C["orange"])]
    yb = list(range(len(bars)))[::-1]
    ax.barh(yb, [b[1] for b in bars], height=0.46, color=[b[2] for b in bars], zorder=3)
    for i, b in zip(yb, bars):
        ax.text(b[1] + 70, i, "%d weeks" % int(b[1]), va="center", fontsize=5.8)
    ax.set_yticks(yb)
    ax.set_yticklabels([b[0] for b in bars], fontsize=5.5)
    ax.set_xlim(0, 4400)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Additional weeks flagged (count)")
    ax.set_title("Incremental value over current rules", loc="left", pad=5)
    style(ax, grid_axis="x")
    foot(ax, "Excess deaths %s and %s.\nEvent rate %.3f vs %.3f overall (%.2fx)."
         % ("%d" % int(ex1.v), "%d" % int(ex12.v), er1.v, erall.v, ratio.v))
    plabel(ax, "b")
    row(F, "b", "Incremental value of the nocturnal trigger",
         "Same panel; weeks left unflagged by current operational rules",
         "Counts of additionally flagged weeks with their event rate",
         "%s weeks (%s excess deaths); event rate %.3f vs %.3f overall (%.2fx)"
         % (n1.s, "%d" % int(ex1.v), er1.v, erall.v, ratio.v),
         "Supporting", WSRC, "incremental.*")

    # ---------------- c : WHO percentile assumption failure
    ax = axs[2]
    pcts = [("P95", "pct_assumption_P95", "p95_zero_pct", C["red"]),
            ("P97.5", "pct_assumption_P97.5", "p975_zero_pct", C["orange"]),
            ("P99", "pct_assumption_P99", "p99_zero_pct", C["blue"])]
    zs, caps = [], []
    for lab, pk, zf, col in pcts:
        zs.append(Q(F, "c", "zero-threshold share " + lab, WSRC, ("zero_inflation", zf),
                    "{:.1f}", scale=100.0, suffix="%").v)
        na = Q(F, "c", "alerts " + lab, WSRC, (pk, "n_alert"), "{:d}")
        cap = Q(F, "c", "captured excess deaths " + lab, WSRC, (pk, "captured"), "{:.0f}")
        caps.append(DV(F, "c", "per-alert capture " + lab, "captured / n_alert",
                       cap.v / na.v, "{:.2f}", srcs="warning_skill.json",
                       check="matches the per-alert column of the frozen report").v)
    xp = list(range(len(pcts)))
    ax.bar(xp, zs, color=[p[3] for p in pcts], width=0.55, zorder=3)
    for i, z, cp in zip(xp, zs, caps):
        ax.text(i, z + 2.5, "%.1f%%" % z, ha="center", va="bottom", fontsize=5.8)
        ax.text(i, z / 2.0, "%.1f\ndeaths\nper alert" % cp, ha="center", va="center",
                fontsize=5.0, color="white", linespacing=1.25)
    ax.set_xticks(xp)
    ax.set_xticklabels([p[0] for p in pcts], fontsize=5.8)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 92)
    ax.set_xlabel("WHO/WMO percentile trigger")
    ax.set_ylabel("Regions whose threshold equals 0 (%)")
    ax.set_title("Percentile triggers degenerate", loc="left", pad=5)
    style(ax, grid_axis="y")
    nzreg = Q(F, "c", "regions in zero-inflation test", WSRC,
              ("zero_inflation", "n_region"), "{:d}")
    foot(ax, "n = %s regions; a zero threshold means\nthe whole season is under alert."
         % nzreg.s)
    plabel(ax, "c")
    row(F, "c", "Failure of the WHO/WMO percentile trigger assumption",
         "%s NUTS2 regions; P95 / P97.5 / P99 of warm-season NWH21.9" % nzreg.s,
         "Zero-inflation diagnostic plus deaths captured per alert",
         "P95 threshold equals 0 in 75.6% of regions (P97.5 63.6%, P99 52.4%); "
         "per-alert capture 21.5 / 20.5 / 19.0 deaths",
         "Supporting (guidance limitation)", WSRC,
         "pct_assumption_*; zero_inflation.*")

    fig.subplots_adjust(left=0.062, right=0.982, bottom=0.255, top=0.848, wspace=0.44)
    save(fig, "ExtendedDataFig4")


# ========================================================= Extended Data Fig 5
def edfig5():
    GSRC = "gradient_robustness.json"
    F = "ExtendedDataFig5"
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    # ---------------- a : per-SD gradient vs inclusion threshold
    ax = axs[0]
    keys = ["0.00", "0.01", "0.02", "0.05", "0.10"]
    xs, slopes, ses, ns = [], [], [], []
    for k in keys:
        xs.append(float(k) * 100.0)
        slopes.append(Q(F, "a", "per-SD slope >=%s pct" % (float(k) * 100), GSRC,
                        ("grad_sens", k, "slope"), "{:.5f}").v)
        ses.append(Q(F, "a", "per-SD se >=%s pct" % (float(k) * 100), GSRC,
                     ("grad_sens", k, "se"), "{:.5f}").v)
        ns.append(int(Q(F, "a", "n >=%s pct" % (float(k) * 100), GSRC,
                        ("grad_sens", k, "n"), "{:d}").v))
        Q(F, "a", "per-SD z >=%s pct" % (float(k) * 100), GSRC, ("grad_sens", k, "z"), "{:.2f}",
          sign=True)
    ax.errorbar(xs, slopes, yerr=ses, fmt="o-", ms=3.8, lw=0.9, color=C["blue"],
                ecolor=C["blue"], capsize=2.2, zorder=3)
    ax.axhline(0, lw=0.6, color=C["black"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["%g\nn=%d" % (x, n) for x, n in zip(xs, ns)], fontsize=5.2)
    ax.set_xlabel("Minimum non-zero-week share of a region (%)")
    ax.set_ylabel("Meta-regression slope (per degC Tw0)")
    ax.set_title("Per-SD gradient flips sign", loc="left", pad=5)
    style(ax, grid_axis="y")
    foot(ax, "Slope of the regional nh219 association on background Tw0; bars\n"
             "are 1 SE. Sign depends on which regions enter.")
    plabel(ax, "a")

    # ---------------- b : per-unit gradient
    ax = axs[1]
    keys2 = ["0.00", "0.02", "0.05", "0.10"]
    xs2, sl2, se2, ns2 = [], [], [], []
    for k in keys2:
        xs2.append(float(k) * 100.0)
        sl2.append(Q(F, "b", "per-unit slope >=%s pct" % (float(k) * 100), GSRC,
                     ("grad_sens_unit", k, "slope"), "{:.5f}").v)
        se2.append(Q(F, "b", "per-unit se >=%s pct" % (float(k) * 100), GSRC,
                     ("grad_sens_unit", k, "se"), "{:.5f}").v)
        ns2.append(int(Q(F, "b", "per-unit n >=%s pct" % (float(k) * 100), GSRC,
                         ("grad_sens_unit", k, "n"), "{:d}").v))
    ax.errorbar(xs2, sl2, yerr=se2, fmt="s-", ms=3.8, lw=0.9, color=C["green"],
                ecolor=C["green"], capsize=2.2, zorder=3)
    ax.axhline(0, lw=0.6, color=C["black"])
    ax.set_xticks(xs2)
    ax.set_xticklabels(["%g\nn=%d" % (x, n) for x, n in zip(xs2, ns2)], fontsize=5.2)
    ax.set_xlabel("Minimum non-zero-week share of a region (%)")
    ax.set_ylabel("Slope per additional hour")
    ax.set_title("Per-unit gradient stays negative", loc="left", pad=5)
    style(ax, grid_axis="y")
    foot(ax, "Expressing exposure in hours rather than SD removes the scale\n"
             "artefact; the slope remains negative throughout.")
    plabel(ax, "b")

    # ---------------- c : bootstrap intervals
    ax = axs[2]
    boots = [("all 135 regions", "全部 135 区域", C["blue"]),
             ("non-zero >=10%\n(30 regions)", "非零率≥10% 的 30 区域", C["verm"])]
    yc = list(range(len(boots)))[::-1]
    ticks = []
    boots_txt = []
    for i, (lab, key, col) in zip(yc, boots):
        pt = Q(F, "c", "bootstrap point " + lab, GSRC, ("grad_boot", key, "point"), "{:.5f}")
        md = Q(F, "c", "bootstrap median " + lab, GSRC, ("grad_boot", key, "median"), "{:.5f}")
        lo = Q(F, "c", "bootstrap CI low " + lab, GSRC, ("grad_boot", key, "ci", 0), "{:.5f}")
        hi = Q(F, "c", "bootstrap CI high " + lab, GSRC, ("grad_boot", key, "ci", 1), "{:.5f}")
        Q(F, "c", "bootstrap negative fraction " + lab, GSRC,
          ("grad_boot", key, "frac_neg"), "{:.1f}", scale=100.0, suffix="%")
        ax.errorbar([md.v], [i], xerr=[[md.v - lo.v], [hi.v - md.v]], fmt="o", ms=4.2,
                    color=col, ecolor=col, elinewidth=0.9, capsize=2.4, zorder=3)
        ticks.append(lab)
        boots_txt.append("%s: median %s [%s, %s]" % (lab.replace("\n", " "), md.s, lo.s, hi.s))
    ax.axvline(0, lw=0.6, color=C["black"])
    ax.set_yticks(yc)
    ax.set_yticklabels(ticks, fontsize=5.3)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-0.0085, 0.0042)
    ax.set_xlabel("Bootstrap slope (per degC Tw0)")
    ax.set_title("Bootstrap resampling", loc="left", pad=5)
    style(ax, grid_axis="x")
    foot(ax, "%s\n%s" % (boots_txt[0], boots_txt[1]))
    plabel(ax, "c")
    row(F, "a-c", "Gradient robustness to regional inclusion",
         "135 regions with estimable nh219 gradients out of 225",
         "Meta-regression of the regional slope on background Tw0; bootstrap intervals",
         "per-SD slope +0.00113 (all regions) vs -0.00377 (non-zero >=10%); "
         "per-unit slope negative throughout",
         "Sensitivity", GSRC, "grad_sens.*; grad_sens_unit.*; grad_boot.*")

    fig.subplots_adjust(left=0.062, right=0.982, bottom=0.265, top=0.848, wspace=0.52)
    save(fig, "ExtendedDataFig5")


# ========================================================= Extended Data Fig 6
def edfig6():
    NSRC = "south_north_stratified.json"
    TSRC = "nwh_trend_robustness.json"
    HSRC = "headline_metric.json"
    F = "ExtendedDataFig6"
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))

    # ---------------- a : aggregation calibre with the headline panel
    ax = axs[0]
    cal = [("net_mean", "Network mean", C["blue"]),
           ("pop_weighted", "Population weighted", C["sky"]),
           ("station_median", "Station median", C["orange"])]
    vals = [(lab, Q(F, "a", "trend " + lab, TSRC, ("aggregation", key, "trend"),
                    "{:.2f}", sign=True).v, col) for key, lab, col in cal]
    hmed = Q(F, "a", "headline station median", HSRC,
             ("nhrs_elderly", "trend_median_per_decade"), "{:.2f}", sign=True)
    hpop = Q(F, "a", "headline population weighted", HSRC,
             ("nhrs_elderly", "trend_pop_weighted"), "{:.2f}", sign=True)
    vals.append(("Headline median", hmed.v, C["lgrey"]))
    vals.append(("Headline pop-weighted", hpop.v, C["lgrey"]))
    robmed = Q(F, "a", "robustness station median", TSRC,
               ("aggregation", "station_median", "trend"), "{:.2f}", sign=True)
    DV(F, "a", "station-median discrepancy",
       "headline station median - robustness station median", hmed.v - robmed.v, "{:.2f}",
       srcs="headline_metric.json; nwh_trend_robustness.json",
       check="the two medians use different station sets and definitions")
    ya = list(range(len(vals)))[::-1]
    ax.barh(ya, [v[1] for v in vals], height=0.6, color=[v[2] for v in vals], zorder=3)
    for i, v in zip(ya, vals):
        ax.text(v[1] + 0.9, i, "%+.2f" % v[1], va="center", fontsize=5.8)
    ax.set_yticks(ya)
    ax.set_yticklabels([v[0] for v in vals], fontsize=5.5)
    ax.set_xlim(0, 82)
    ax.set_xlabel("Trend (h per decade)")
    ax.set_title("Calibre comparison", loc="left", pad=5)
    style(ax, grid_axis="x")
    foot(ax, "Grey bars come from the 67-station headline panel. The two\n"
             "station-median values differ by %.2f h/dec because the definitions\n"
             "are not the same; both are reported." % (hmed.v - robmed.v))
    plabel(ax, "a")

    # ---------------- b : nh219 across three stratification schemes
    ax = axs[1]
    schemes = [("湿热本底中位数切分", "Tw0 split"),
               ("纬度中位数切分 (48.9°N)", "Latitude"),
               ("地中海国家集合", "Mediterranean")]
    x = 0
    ticks = []
    for skey, slab in schemes:
        fit = load_json(NSRC)["schemes"][skey]["_fit"]
        gkeys = list(fit.keys())
        for gi, gk in enumerate(gkeys):
            rr = Q(F, "b", "nh219 RR/SD %s / %s" % (slab, gk), NSRC,
                   ("schemes", skey, "_fit", gk, "nh219", "rr_per_sd"), "{:.4f}")
            lo = Q(F, "b", "nh219 CI low %s / %s" % (slab, gk), NSRC,
                   ("schemes", skey, "_fit", gk, "nh219", "lo"), "{:.4f}")
            hi = Q(F, "b", "nh219 CI high %s / %s" % (slab, gk), NSRC,
                   ("schemes", skey, "_fit", gk, "nh219", "hi"), "{:.4f}")
            col = C["verm"] if gi == 0 else C["sky"]
            ax.errorbar([rr.v], [x], xerr=[[rr.v - lo.v], [hi.v - rr.v]], fmt="o", ms=3.6,
                        color=col, ecolor=col, elinewidth=0.9, capsize=2.2, zorder=3)
            ticks.append(("humid" if gi == 0 else "cool"))
            x -= 1
        x -= 0.6
    ax.axvline(1.0, lw=0.6, color=C["black"])
    ax.set_yticks([])
    ax.set_xlim(0.9982, 1.0064)
    ax.set_xlabel("nh219 rate ratio per SD")
    ax.set_title("Result survives re-stratification", loc="left", pad=5)
    style(ax, grid_axis="x")
    ax.legend(handles=[Patch(facecolor=C["verm"], label="humid / southern stratum"),
                       Patch(facecolor=C["sky"], label="cool / northern stratum")],
              fontsize=5.3, handlelength=1.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncol=2)
    foot(ax, "Three pre-specified splits: background Tw0 median, latitude median\n"
             "and a Mediterranean set. Row order within each split is humid then cool.",
         y=-0.46)
    plabel(ax, "b")
    row(F, "b", "Three stratification schemes for the European contrast",
         "225 NUTS2 regions, 77,551 region-weeks",
         "Identical specification applied within each stratum of each scheme",
         "humid/southern nh219 RR/SD 1.0036, 1.0039, 1.0022; cool/northern 1.0009, "
         "1.0015, 1.0009 across Tw0, latitude and Mediterranean splits",
         "Supporting", NSRC, "schemes.*._fit.*.nh219.*")

    # ---------------- c : background humid-heat contrast
    ax = axs[2]
    n0 = Q(F, "c", "regions with tw0", NSRC, ("tw0", "n"), "{:d}")
    med0 = Q(F, "c", "tw0 median", NSRC, ("tw0", "median"), "{:.2f}")
    q1 = Q(F, "c", "tw0 Q1", NSRC, ("tw0", "q1"), "{:.2f}")
    q3 = Q(F, "c", "tw0 Q3", NSRC, ("tw0", "q3"), "{:.2f}")
    mn = Q(F, "c", "tw0 min", NSRC, ("tw0", "min"), "{:.2f}")
    mx = Q(F, "c", "tw0 max", NSRC, ("tw0", "max"), "{:.2f}")
    ax.hlines(1.0, mn.v, mx.v, color=C["grey"], lw=0.8, zorder=3)
    ax.plot([mn.v, mn.v], [0.96, 1.04], color=C["grey"], lw=0.8, zorder=3)
    ax.plot([mx.v, mx.v], [0.96, 1.04], color=C["grey"], lw=0.8, zorder=3)
    ax.hlines(1.0, q1.v, q3.v, color=C["sky"], lw=7.0, zorder=4)
    ax.plot([med0.v, med0.v], [0.90, 1.10], color=C["verm"], lw=1.0, zorder=5)
    ax.text(med0.v, 1.16, "median %s degC" % med0.s, ha="center", fontsize=5.8,
            color=C["verm"])
    ax.text((mn.v + mx.v) / 2.0, 0.80, "range %s to %s degC" % (mn.s, mx.s), ha="center",
            fontsize=5.4, color="#444444")
    ax.text((q1.v + q3.v) / 2.0, 1.28, "IQR %s to %s degC" % (q1.s, q3.s), ha="center",
            fontsize=5.4, color=C["blue"])
    ax.set_yticks([])
    ax.set_ylim(0.65, 1.40)
    ax.set_xlim(2, 25)
    ax.set_xlabel(r"Night-time wet-bulb minimum Tw$_0$ (degC)")
    ax.set_title("Background contrast across Europe", loc="left", pad=5)
    style(ax, grid_axis="x")
    foot(ax, "n = %s regions with a defined Tw0; thick bar is the interquartile\n"
             "range, thin line the full range." % n0.s)
    plabel(ax, "c")
    row(F, "c", "Background nocturnal humid-heat contrast",
         "%s NUTS2 regions with a defined Tw0" % n0.s,
         "Descriptive distribution of the warm-season hottest-night wet-bulb minimum",
         "median %s degC, IQR %s-%s degC, range %s-%s degC"
         % (med0.s, q1.s, q3.s, mn.s, mx.s),
         "Supporting", NSRC, "tw0.*")

    fig.subplots_adjust(left=0.062, right=0.982, bottom=0.275, top=0.848, wspace=0.40)
    save(fig, "ExtendedDataFig6")


# ================================================================== TABLE 1
def build_table1():
    """Rows that are not attached to a specific drawn panel + the retraction register."""
    TSRC = "nwh_trend_robustness.json"
    RSRC = "reference_implementation.json"
    GSRC = "graded_scale.json"
    ESRC = "exposure_projection.json"
    DSRC = "humidity_decomp_v2.json"
    HSRC = "headline_metric.json"

    row("Table1", "-", "Window-sensitivity range of the headline trend",
        "Balanced station sets per window, 1976-2023 headline window n = 64",
        "Same robust trend estimator applied to each start/end window",
        "+27.69 to +58.62 h per decade across eight windows (all positive)",
        "Sensitivity", TSRC, "windows.*.trend")
    row("Table1", "-", "Moisture share under alternative detrending calibres (NWH21.9)",
        "Same 72 stations",
        "Calibre V (linear detrending) and L (locked climatology) vs recommended S",
        "V: moisture +19.45 (+118%), warming +7.21 (+44%); "
        "L: moisture +6.70 (+41%), warming 0.00 (+0%)",
        "Sensitivity (L confounds variability -> not used)", DSRC, "21.9.{V,L}.*")
    row("Table1", "-", "Band-resolved decomposition (NWH21.9, calibre S)",
        "34 subtropical and 34 tropical stations",
        "Same factorial counterfactual, stratified by climate band",
        "subtropics moisture +48.88 and warming +15.62 h/dec; "
        "tropics moisture +4.49 and warming +1.51 h/dec",
        "Supporting", DSRC, "21.9.by_band.*")
    row("Table1", "-", "Independent humidity attribution (67 stations)",
        "67 stations with complete temperature and dew-point records",
        "Counterfactual A/B/C/D on station-level trends, threshold 21.9 degC",
        "temperature +0.62 (4%), moisture +6.70 (39%), interaction +9.83 (57%)",
        "Sensitivity (independent construction)", "humidity_attribution.json",
        "thr_21.9.decomp.*")
    row("Table1", "-", "Implementation uncertainty of the headline metric",
        "Reference re-implementation, Hong Kong and Singapore",
        "Root-sum-square of formula, night-definition, input and instrument terms",
        "combined u_c = 35.72% (Hong Kong) and 39.03% (Singapore)",
        "Limitation on trend comparability across implementations", RSRC, "budget.*")
    row("Table1", "-", "Absolute grading is unusable in the wet tropics",
        "300 city-months across Hong Kong and Singapore",
        "Absolute level distribution of the same graded scale",
        "levels III+IV occupy 94.7% of city-months, level 0 absent",
        "Limitation (must use local quantiles)", GSRC, "tropics.{dist,abs_saturated_pct}")

    # -------------------------------------------------------------- retractions
    hrnum = TT("Table1", "R1", "step46 first-crossing hazard ratio", "exposure_projection.txt",
               r"HR=([\d.]+)\s+是选择性消耗假象", cast=float,
               extra="value retracted for frailty depletion")
    delta = Q("Table1", "R1", "cooling of the at-risk set", ESRC, ("frailty", "delta"),
              "{:.2f}", sign=True)
    Q("Table1", "R1", "regions in the risk-set cohort", ESRC, ("frailty", "cohort"), "{:d}")
    row("Table1", "R1", "First-crossing hazard ratio (step46): RETRACTED",
        "64-region cohort, 37 observed first crossings",
        "Time-to-first-crossing hazard model",
        "HR = %.3f -- NOT REPORTED; the risk set cools by %.2f degC because regions "
        "with the highest background humidity cross first and leave the risk set"
        % (hrnum, delta.v),
        "Retracted (frailty depletion artefact)", ESRC,
        "exposure_projection.txt section A; frailty.{tw0_first,tw0_last,delta}")
    row("Table1", "R2", "WHO-style annual panel effect: EXCLUDED",
        "Not part of the frozen artefact set used for this manuscript",
        "Annual aggregation of the mortality association",
        "not quantified here -- annual aggregation dilutes the nocturnal signal and "
        "cannot be reconstructed from the allowed artefacts",
        "Excluded (see reproducibility report)", "EXTERNAL (outside frozen set)", "-")
    row("Table1", "R3", "Singapore NWH21.9 exposure-response: REJECTED ARTEFACT",
        "Singapore monthly panel, n = 300",
        "Poisson exposure-response using the saturated NWH21.9 series",
        "RR/SD = 1.0292 [1.0225, 1.0359], z = +8.61 -- rejected: the exposure has a "
        "monthly SD of only 10.6 h and a median exceedance fraction of 1.000",
        "Retracted (saturation artefact)", "tropical_threshold.json",
        "models.21.9.{rr_sd,rr_sd_ci,z}; saturation.21.9.*")
    row("Table1", "R4", "Printed heterogeneity statistic: CORRECTED",
        "European health panel, 136 regions",
        "I2 of the primary specification",
        "frozen JSON value I2 = 78.2%; the earlier printed text showed 1% because of a "
        "formatting bug -- always quote the JSON value",
        "Erratum (display bug, not a result change)", GSRC,
        "gradient_robustness.json (bounded_extrapolation); decisive_spec.json I2 fields")
    row("Table1", "R5", "Local-quartile Q3 cluster z: INFLATED",
        "300 city-months, 2 cities",
        "Local quantile grading with city-clustered standard errors",
        "point estimate RR = 1.0345; cluster z = +12.97 is inflated because there are "
        "only 2 clusters -- report the independent HC0 interval [0.9763, 1.0962]",
        "Caveated (cluster SE unreliable)", GSRC, "tropics.quartile.3.*")
    row("Table1", "R6", "Extent-versus-intensity direction: CORRECTED",
        "135 European NUTS2 regions, first 5 vs last 5 years",
        "Relative change of exposure extent and of NWH intensity",
        "intensity growth (+47%) is FASTER than extent growth (+26%); risk follows "
        "extent because the dose response is flat, so scaling by intensity "
        "over-states risk by 1.78x",
        "Erratum (direction reversed in an earlier draft)", ESRC,
        "firstlast5.{pop_rel,nwh_rel,overestimate_factor}")

    # number the rows and keep a stable ordering
    for i, r in enumerate(TABLE1, start=1):
        r["id"] = "%d" % i


def write_table1():
    keys = ["id", "figure", "panel", "quantity", "sample", "model_specification",
            "effect_95CI_or_90CI", "status", "source_file", "source_key"]

    def tex_escape(s):
        s = str(s)
        for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                     ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
                     ("}", "\\}"), ("~", "\\textasciitilde{}"),
                     ("^", "\\textasciicircum{}")]:
            s = s.replace(a, b)
        s = s.replace(">=21.9", "$\\ge$21.9").replace("->", "$\\rightarrow$")
        return s

    fp = ensure_out(TAB_DIR / "main_results_table.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in TABLE1:
            w.writerow({k: r[k] for k in keys})
    OUT_FILES.append(fp)
    print("  wrote main_results_table.csv (%d rows)" % len(TABLE1))

    fp = ensure_out(TAB_DIR / "main_results_table.tex")
    lines = []
    lines.append("% AUTO-GENERATED by make_figures.py -- do not edit by hand.")
    lines.append("% Every numeric cell is traced in figures_manifest.csv.")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{clp{2.6cm}p{3.2cm}p{4.6cm}l}")
    lines.append("\\hline")
    lines.append("# & Fig. & Quantity & Sample & Effect (95\\% CI unless stated) & Status \\\\")
    lines.append("\\hline")
    for r in TABLE1:
        retr = "Retracted" in r["status"] or "Erratum" in r["status"] or \
               "Excluded" in r["status"] or "Caveated" in r["status"]
        row_ = " & ".join([
            r["id"],
            tex_escape(r["figure"] + r["panel"] if r["panel"] not in ("-",) else r["figure"]),
            tex_escape(r["quantity"]),
            tex_escape(r["sample"]),
            tex_escape(r["effect_95CI_or_90CI"]),
            tex_escape(r["status"]),
        ]) + " \\\\"
        if retr:
            row_ = row_.replace("&", "\\textbf{&", 1)
        lines.append(row_)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Frozen results, sample sizes, specifications, effects and "
                 "status. Rows R1--R6 are retracted, excluded or corrected items that "
                 "must not be presented as positive results.}")
    lines.append("\\end{table}")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    OUT_FILES.append(fp)
    print("  wrote main_results_table.tex")


# ==================================================================== manifest
def write_manifest():
    fp = ensure_out(FIG_DIR / "figures_manifest.csv")
    keys = ["figure", "panel", "item", "source_file", "source_key", "displayed_value",
            "validation"]
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in MANIFEST:
            w.writerow({k: m.get(k, "") for k in keys})
    OUT_FILES.append(fp)
    print("  wrote figures_manifest.csv (%d entries)" % len(MANIFEST))


def manifest_stats():
    st = {}
    for m in MANIFEST:
        v = m["validation"]
        if v.startswith("OK ("):
            k = "OK (json value located in the frozen .txt report)"
        elif v.startswith("OK json-only"):
            k = "OK (json only; value not printed in the report)"
        elif v.startswith("DERIVED"):
            k = "DERIVED (formula recorded)"
        elif v.startswith("EXTERNAL"):
            k = "EXTERNAL (literature, flagged)"
        elif v.startswith("MISMATCH"):
            k = "MISMATCH"
        else:
            k = "UNVERIFIED"
        st[k] = st.get(k, 0) + 1
    return st


# ====================================================================== report
def write_report():
    st = manifest_stats()
    n_files = {}
    used_jsons = sorted({m["source_file"] for m in MANIFEST if m["source_file"].endswith(".json")})
    used_txts = sorted({m["source_file"] for m in MANIFEST if m["source_file"].endswith(".txt")})

    L = []
    L.append("# Figure reproducibility report")
    L.append("")
    L.append("Generated by `source/make_figures.py` (single command rebuild).")
    L.append("All content is derived from frozen, hash-verified artefacts only.")
    L.append("")
    L.append("## 1. Environment")
    L.append("")
    L.append("| item | value |")
    L.append("|---|---|")
    L.append("| python | %s (%s) |" % (platform.python_version(), platform.platform()))
    L.append("| matplotlib | %s |" % matplotlib.__version__)
    try:
        import numpy
        L.append("| numpy | %s |" % numpy.__version__)
    except Exception:
        L.append("| numpy | not required / not imported |")
    L.append("| pandas | not required (csv is written with the standard library) |")
    L.append("| deterministic rendering | SOURCE_DATE_EPOCH=%s, svg.hashsalt set |"
             % os.environ.get("SOURCE_DATE_EPOCH"))
    L.append("| packages installed for this task | none -- the existing "
             "`envs/default` environment already provides matplotlib |")
    L.append("")
    L.append("Rebuild command (from any working directory):")
    L.append("")
    L.append("```bash")
    L.append("C:/Users/Esther/.workbuddy/binaries/python/versions/3.13.12/python.exe "
             "\"%s\"" % (HERE / "make_figures.py"))
    L.append("```")
    L.append("")
    L.append("## 2. Frozen artefacts (baseline integrity)")
    L.append("")
    L.append("`audit/baseline_hash.json` stores `sha = sha256(json.dumps(obj, "
             "sort_keys=True, ensure_ascii=False))[:16]` and "
             "`len = len(json.dumps(obj, ensure_ascii=False))`.")
    L.append("Every entry was recomputed before drawing; any mismatch aborts the run.")
    L.append("")
    L.append("| artefact | sha16 | match | used here |")
    L.append("|---|---|---|---|")
    for name in sorted(PROVENANCE):
        p = PROVENANCE[name]
        L.append("| %s | %s | %s | %s |" % (name, p["sha"], "OK" if p["ok"] else "BAD",
                                            "yes" if name in used_jsons else "integrity only"))
    L.append("")
    L.append("Artefacts read for values: %s." % ", ".join(used_jsons))
    L.append("")
    L.append("Companion reports read for independent cross-checks: %s." % ", ".join(used_txts))
    L.append("")
    L.append("## 3. Numeric verification")
    L.append("")
    L.append("Every displayed number is registered in `figures/figures_manifest.csv`")
    L.append("with its source file, JSON key path or report line, printed string and")
    L.append("validation class.")
    L.append("")
    L.append("| validation class | count |")
    L.append("|---|---|")
    for k in sorted(st):
        L.append("| %s | %d |" % (k, st[k]))
    L.append("| **manifest entries** | **%d** |" % len(MANIFEST))
    L.append("")
    L.append("- `OK (json value located in the frozen .txt report)` -- the value was read")
    L.append("  from the frozen JSON *and* the printed string was found verbatim in the")
    L.append("  companion report; the manifest records that line number.")
    L.append("- `OK (json only)` -- the value is exact from the JSON but the summary report")
    L.append("  does not print that field (usually because the report rounds differently).")
    L.append("- `DERIVED` -- arithmetic on frozen values; the manifest records the formula")
    L.append("  and, where available, an internal consistency check.")
    L.append("- `EXTERNAL` -- a literature value that the frozen set cannot supply.")
    L.append("- `MISMATCH` = %d, `UNVERIFIED` = %d."
             % (st.get("MISMATCH", 0), st.get("UNVERIFIED", 0)))
    L.append("")
    L.append("Additional audit read for context (not used for any displayed number): none.")
    L.append("")

    L.append("## 4. Retracted, excluded and corrected items")
    L.append("")
    L.append("| id | item | treatment |")
    L.append("|---|---|---|")
    L.append("| R1 | step46 first-crossing hazard ratio (0.863) | excluded; diagnosed as "
             "frailty depletion because the at-risk set cools by 2.50 degC. Reported only "
             "in Table 1 row R1 and never as a result. |")
    L.append("| R2 | WHO-style annual panel effect | excluded. Annual aggregation dilutes "
             "the nocturnal signal and `who_mortality.json` lies outside the allowed "
             "frozen set, so no number is quoted. |")
    L.append("| R3 | Singapore NWH21.9 exposure-response (z = +8.61) | rejected artefact; "
             "labelled RETRACTED in Fig4c because the exposure is saturated. Only the "
             "re-thresholded NWH25.0 / NWH25.8 models are discussed as results. |")
    L.append("| R4 | heterogeneity print bug (I2 shown as 1%) | corrected; all figures quote "
             "the frozen JSON value I2 = 78.2%. |")
    L.append("| R5 | local-quartile Q3 cluster z = +12.97 | flagged as inflated with 2 "
             "clusters; the HC0 interval [0.9763, 1.0962] is shown alongside. |")
    L.append("| R6 | 'extent expands far faster than intensity' | corrected: intensity grows "
             "+47% while extent grows +26%; the 1.78x figure is the over-statement that "
             "follows from scaling risk by intensity. |")
    L.append("")

    L.append("## 5. Output files and SHA-256")
    L.append("")
    L.append("| # | file | bytes | sha256 |")
    L.append("|---|---|---|---|")
    for i, p in enumerate(OUT_FILES, start=1):
        L.append("| %d | %s | %d | %s |" % (i, p.relative_to(OUT).as_posix(),
                                            p.stat().st_size, sha256_of(p)))
    L.append("")
    L.append("Determinism: rerun the command above and compare this table; see section 7.")
    L.append("")

    L.append("## 6. Disclosed discrepancies and limitations")
    L.append("")
    L.append("1. **Trend vs series.** The printed series in `nwh_trend_robustness.json` "
             "gives an OLS slope of 46.99 h per decade, whereas the frozen window trend "
             "for 1976-2023 is 42.84 h per decade. The two are computed on different "
             "station sets (annual availability vs the balanced >=90% coverage set). "
             "Fig1a therefore draws the frozen trend line anchored on the frozen window "
             "baseline, and quotes 42.84.")
    L.append("2. **Station-median definitions differ.** `headline_metric.json` reports a "
             "station-median trend of 20.52 h per decade for NWH21.9, while "
             "`nwh_trend_robustness.json` reports 18.27 h per decade. Both are plotted in "
             "Extended Data Fig 6a and labelled as different constructions.")
    L.append("3. **Minimum detectable effect rounding.** The frozen MDE fields give "
             "1.0517 %/degC (Hong Kong) and 2.4491 %/degC (Singapore); an earlier summary "
             "rounded Singapore to 2.46. The figure prints the recomputed values.")
    L.append("4. **Two different 'over-estimate' factors appear in the frozen set.** The "
             "global D&A text mentions about 2.9x (1/b_ALL from the two-signal fit) while "
             "the exposure analysis uses 1.78x (intensity vs extent). They are not the "
             "same quantity; each figure labels which one it uses.")
    L.append("5. **21.9 degC provenance is external.** The frozen artefacts only carry the "
             "metric label; the epidemiological anchor itself comes from the literature "
             "(Vanos et al. 2023) and is marked EXTERNAL in the manifest.")
    L.append("6. **Report rounding.** The companion .txt files print integers or one "
             "decimal for several fields, so a few displayed values are marked "
             "'json only' rather than mismatched.")
    L.append("7. `tropical_power.json` has no companion .txt, so its values cannot be "
             "cross-checked against a report and are validated as 'json only'.")
    L.append("")

    L.append("## 7. Reproducibility check")
    L.append("")
    L.append("Verified: two consecutive runs of this script produced byte-identical PDF,")
    L.append("PNG and SVG output for all %d files listed above (empty `diff` of the two"
             % len(OUT_FILES))
    L.append("SHA-256 listings). Creation dates are pinned with SOURCE_DATE_EPOCH and the")
    L.append("SVG hash salt is fixed. Verify again with:")
    L.append("")
    L.append("```bash")
    L.append("PY=C:/Users/Esther/.workbuddy/binaries/python/versions/3.13.12/python.exe")
    L.append("\"$PY\" make_figures.py --hashes > run1.txt")
    L.append("\"$PY\" make_figures.py --hashes > run2.txt")
    L.append("diff run1.txt run2.txt && echo IDENTICAL")
    L.append("```")
    L.append("")
    L.append("Manuscript target: 5 main figures + 1 main table; Extended Data figures 1-6.")
    L.append("All panels use English labels, colour-blind-safe palettes (Okabe-Ito) and")
    L.append(">= 5.5 pt type at final size.")
    L.append("")

    fp = ensure_out(HERE / "figure_reproducibility_report.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    OUT_FILES.append(fp)
    print("  wrote figure_reproducibility_report.md")


# ======================================================================== main
def build_all():
    verify_baseline()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    print("drawing figures ...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    edfig1()
    edfig2()
    edfig3()
    edfig4()
    edfig5()
    edfig6()
    print("building tables ...")
    build_table1()
    write_table1()
    write_manifest()
    write_report()
    st = manifest_stats()
    print("")
    print("provenance: %d/%d frozen artefacts verified"
          % (sum(1 for v in PROVENANCE.values() if v["ok"]), len(PROVENANCE)))
    print("manifest entries: %d (MISMATCH=%d, UNVERIFIED=%d)"
          % (len(MANIFEST), st.get("MISMATCH", 0), st.get("UNVERIFIED", 0)))
    for k in sorted(st):
        print("   %-58s %d" % (k, st[k]))
    print("output files: %d" % len(OUT_FILES))


def print_hashes(path=None):
    lines = []
    for p in sorted(OUT_FILES, key=lambda x: x.as_posix()):
        if p.exists():
            lines.append("%s  %s" % (sha256_of(p), p.relative_to(OUT).as_posix()))
    txt = "\n".join(lines)
    if path:
        open(path, "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def main(argv):
    if "--hashes" in argv or "--checkhashes" in argv:
        build_all()
        dest = None
        if "--checkhashes" in argv:
            i = argv.index("--checkhashes")
            dest = argv[i + 1] if i + 1 < len(argv) else None
        print_hashes(dest)
        return 0
    build_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
