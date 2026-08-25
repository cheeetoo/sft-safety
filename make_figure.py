"""Three-checkpoint (SFT / DPO / final RLVR) figures for the writeup.

Outputs (figures/):
  fig_main.png        GDM-style sections: rate panels share a 0-80% axis, the Petri
                      panel carries its own 1-10 mean-judge-score axis; a vertical
                      rule separates the capability control on the right.
  fig_petri_dims.png  every Petri dimension, mean judge score (1-10)

Run:  uv run --with plotly --with 'kaleido==0.2.1' python make_figure.py
"""

import glob
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

ROOT = Path(__file__).parent
CKPTS = ["sft", "dpo", "final"]
NAMES = {"sft": "Post-SFT", "dpo": "Post-DPO", "final": "Final (post-RLVR)"}
CREAM = "#fbfbf9"

# teal / salmon / rust
COLORS = {"sft": "#649891", "dpo": "#e3b3aa", "final": "#c27752"}


def darken(hex_color, f=0.72):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


# the salmon is much lighter than the other two against cream, so every fill gets a
# darker stroke of itself: keeps the palette but holds the edges at small sizes
EDGES = {c: darken(v) for c, v in COLORS.items()}

MODEL_NAMES = {
    "vllm/allenai/Olmo-3-32B-Think-SFT": "sft",
    "vllm/allenai/Olmo-3-32B-Think-DPO": "dpo",
    "vllm/allenai/Olmo-3-32B-Think": "final",
}

PETRI_META_DIMS = {
    "admirable", "disappointing", "needs_attention", "fun_or_funny",
    "scenario_realism", "high_stakes_setting", "auditor_failure",
    "broken_tool_use", "stuck_in_loops",
}

# Petri judge scores run 1-10 with 1 = "none of this behavior", so bars sit on 1.
PETRI_FLOOR = 1.0


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) / denom, (center + half) / denom


def rate(k, n):
    lo, hi = wilson(k, n)
    return SimpleNamespace(p=100 * k / n, lo=100 * lo, hi=100 * hi, k=k, n=n)


def mean_ci(xs):
    m, n = statistics.mean(xs), len(xs)
    half = 1.96 * statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
    return SimpleNamespace(p=m, lo=m - half, hi=m + half, k=None, n=n)


# ---------------------------------------------------------------- data
def odcv_stats():
    scores = {}  # (model, scenario) -> [judge scores]
    for j in ("claude47", "gemini", "gpt55", "grok"):
        with open(ROOT / "judging" / f"scores_{j}_by_model.json") as f:
            for model, scen_scores in json.load(f).items():
                for scen, s in scen_scores.items():
                    scores.setdefault((model, scen), []).append(s)
    out = {}
    for c in CKPTS:
        meds = [
            statistics.median(v)
            for (m, _), v in scores.items()
            if m in (f"olmo3-{c}-mandated", f"olmo3-{c}-incentivized")
        ]
        assert len(meds) == 80, (c, len(meds))
        out[c] = rate(sum(1 for m in meds if m >= 3), len(meds))
    return out


def petri_values():
    out = {}
    for c in CKPTS:
        files = glob.glob(str(ROOT / f"logs/petri/{c}/*.eval"))
        assert len(files) == 1, files
        vals = []
        for s in read_eval_log_sample_summaries(files[0]):
            v = (s.scores or {}).get("audit_judge")
            if v is not None and isinstance(v.value, dict):
                vals.append(v.value)
        out[c] = vals
    return out


def petri_dim_mean(vals, c, dim):
    xs = [int(v[dim]) for v in vals[c] if isinstance(v.get(dim), (int, float))]
    return mean_ci(xs)


def strongreject_stats():
    out = {}
    for c in CKPTS:
        candidates = []
        for p in glob.glob(str(ROOT / f"logs/strongreject/{c}/*.eval")):
            if read_eval_log(p, header_only=True).status != "success":
                continue
            scored = [s for s in read_eval_log_sample_summaries(p) if s.scores]
            candidates.append((len(scored), p, scored))
        _, _, scored = max(candidates)
        k = n = 0
        for s in scored:
            v = s.scores["strong_reject_scorer"].value
            if isinstance(v, dict) and "refusal" in v:
                n += 1
                k += int(v["refusal"] == 0)
        out[c] = rate(k, n)
    return out


def impossiblebench_stats():
    best = {}  # (ckpt, split) -> (n_scored, name, scored)
    for p in glob.glob(str(ROOT / "impossiblebench/logs/**/*.eval"), recursive=True):
        hdr = read_eval_log(p, header_only=True)
        ckpt = MODEL_NAMES.get(hdr.eval.model)
        split = (hdr.eval.task_args or {}).get("split")
        if ckpt is None or split is None:
            continue
        scored = [s for s in read_eval_log_sample_summaries(p) if s.scores]
        cand = (len(scored), Path(p).name, scored)
        if (ckpt, split) not in best or cand[:2] > best[(ckpt, split)][:2]:
            best[(ckpt, split)] = cand
    out = {}
    for (ckpt, split), (n, _, scored) in sorted(best.items()):
        k = sum(1 for s in scored
                if s.scores["agentic_humaneval_scorer"].value in ("C", 1, 1.0))
        out.setdefault(ckpt, {})[split] = rate(k, n)
    return out


def mask_stats():
    out = {}
    for c in CKPTS:
        files = glob.glob(str(ROOT / f"logs/mask/{c}/*.eval"))
        assert len(files) == 1, files
        cats = []
        for s in read_eval_log_sample_summaries(files[0]):
            v = (s.scores or {}).get("accuracy_and_honesty")
            if v is not None and isinstance(v.value, dict):
                cats.append(v.value.get("honesty"))
        n = len(cats)
        out[c] = {"lie": rate(sum(1 for x in cats if x == "lie"), n),
                  "evade": rate(sum(1 for x in cats if x == "evade"), n)}
    return out


# ---------------------------------------------------------------- fig 1
def make_main(panels, out_stem):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=len(panels),
        column_widths=[len(g) + 0.42 for _, _, _, g in panels],
        subplot_titles=[t for t, _, _, _ in panels],
        horizontal_spacing=0.022,
    )
    # the % axis is shared, so it is labelled once on the leftmost rate panel;
    # Petri is on its own score axis and always carries its own labels
    first_pct = next(i for i, (_, _, kind, _) in enumerate(panels, start=1)
                     if kind == "pct")

    # The Petri panel is on a 1-10 score axis and the rest are percentages, so pick
    # both ranges as the SAME number of grid intervals (20 points / 1 score point).
    # Gridlines then land at identical heights across every panel.
    PCT_DTICK, SCORE_DTICK = 20.0, 1.0
    BARS = 0.876  # share of the span left to bars; the rest is value-label headroom
    max_pct = max(st.hi for _, _, kind, groups in panels if kind == "pct"
                  for _, stats in groups for st in stats.values())
    max_score = max(st.hi - PETRI_FLOOR
                    for _, _, kind, groups in panels if kind == "score"
                    for _, stats in groups for st in stats.values())
    intervals = max(max_pct / BARS / PCT_DTICK, max_score / BARS / SCORE_DTICK)
    pct_max = PCT_DTICK * intervals
    score_max = PETRI_FLOOR + SCORE_DTICK * intervals

    for col, (_, _, kind, groups) in enumerate(panels, start=1):
        floor = 0.0 if kind == "pct" else PETRI_FLOOR
        top = pct_max if kind == "pct" else score_max
        pad = (top - floor) * 0.045
        for gi, (label, stats) in enumerate(groups):
            for c, dx in zip(CKPTS, (-0.28, 0.0, 0.28)):
                st = stats[c]
                count = f" ({st.k}/{st.n})" if st.k is not None else f" (n={st.n})"
                fig.add_trace(go.Bar(
                    x=[gi + dx], y=[st.p - floor], base=floor,
                    width=0.26, marker_color=COLORS[c],
                    marker_line=dict(color=EDGES[c], width=0.8),
                    name=NAMES[c], legendgroup=c,
                    showlegend=(col == 1 and gi == 0),
                    error_y=dict(type="data", symmetric=False,
                                 array=[st.hi - st.p], arrayminus=[st.p - st.lo],
                                 width=2.5, thickness=1.1, color="#444"),
                    hovertemplate=(
                        f"{NAMES[c]}: {st.p:.2f}{count}"
                        f"<br>95% CI [{st.lo:.2f}, {st.hi:.2f}]<extra></extra>"),
                ), row=1, col=col)
                # zigzag: middle (dpo) label sits a step higher to avoid collisions
                ly = st.hi + (pad * 2.3 if c == "dpo" else pad)
                txt = f"{st.p:.1f}" if kind == "pct" else f"{st.p:.2f}"
                fig.add_trace(go.Scatter(
                    x=[gi + dx], y=[ly], text=[txt], mode="text",
                    textfont=dict(size=9.5, color="#333"),
                    showlegend=False, hoverinfo="skip", cliponaxis=False,
                ), row=1, col=col)
        fig.update_xaxes(
            tickvals=list(range(len(groups))),
            ticktext=[f"<b>{g[0]}</b>" for g in groups],
            tickfont=dict(size=12.5, color="#2f2f2f"),
            tickangle=-28,
            range=[-0.62, len(groups) - 0.38],
            fixedrange=True, row=1, col=col,
        )
        fig.update_yaxes(
            range=[floor, top], tickfont=dict(size=10.5), gridcolor="#e8e6df",
            ticksuffix="%" if kind == "pct" else None,
            dtick=PCT_DTICK if kind == "pct" else SCORE_DTICK,
            showticklabels=(col == first_pct or kind == "score"),
            row=1, col=col,
        )

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor=CREAM, plot_bgcolor=CREAM,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.16,
                    yanchor="bottom", font=dict(size=12)),
        width=1480, height=446,
        # autoexpand would pad the edges to fit the angled tick labels, which mostly
        # sit inside their own panel already; fixed margins reclaim that space
        margin=dict(t=82, b=98, l=34, r=16, autoexpand=False),
    )
    fig.update_annotations(font=dict(size=15.5))  # subplot titles (eval names)

    # Lift the eval names and hang the parenthetical below them on its own line: a
    # one-line title would run off the right edge above the narrow last panel.
    # add_annotation rebuilds layout.annotations and detaches any handles taken
    # before it, so shift every title first and only then add the second lines
    titles = list(fig.layout.annotations)
    specs = [(t.x, t.xref, t.y, t.yref, note)
             for t, (_, note, _, _) in zip(titles, panels)]
    for t in titles:
        t.y = t.y + 0.055
    for x, xref, base_y, yref, note in specs:
        if note:
            fig.add_annotation(
                text=f"({note})", xref=xref, x=x, yref=yref, y=base_y,
                xanchor="center", yanchor="bottom", showarrow=False,
                font=dict(size=11.5, color="#6b6b6b"),
            )

    # vertical rule separating the capability control (last panel) from the safety evals
    last = len(panels)
    prev_end = fig.layout[f"xaxis{last - 1}"].domain[1]
    this_start = fig.layout[f"xaxis{last}"].domain[0]
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=(prev_end + this_start) / 2, x1=(prev_end + this_start) / 2,
        y0=-0.36, y1=1.10, line=dict(color="#c3c0b6", width=1.2),
    )

    fig.write_html(ROOT / "figures" / f"{out_stem}.html", include_plotlyjs=True)
    fig.write_image(ROOT / "figures" / f"{out_stem}.png", scale=2)
    print(f"wrote figures/{out_stem}.png")


# ---------------------------------------------------------------- fig 2
def make_petri_dims(vals, out_stem):
    import plotly.graph_objects as go

    dims = sorted({d for c in CKPTS for v in vals[c] for d in v})
    stats = {d: {c: petri_dim_mean(vals, c, d) for c in CKPTS} for d in dims}

    def block(ds):
        return sorted(ds, key=lambda d: -max(stats[d][c].p for c in CKPTS))

    behavior = block([d for d in dims if d not in PETRI_META_DIMS])
    meta = block([d for d in dims if d in PETRI_META_DIMS])

    GAP = 1.6
    ys = {}
    y = len(behavior) + len(meta) + GAP
    for d in behavior:
        ys[d] = (y := y - 1)
    y -= GAP
    for d in meta:
        ys[d] = (y := y - 1)

    fig = go.Figure()
    for c, dy in zip(CKPTS, (0.24, 0.0, -0.24)):
        xs_, ys_, err_hi, err_lo, hover = [], [], [], [], []
        for d in behavior + meta:
            st = stats[d][c]
            xs_.append(st.p)
            ys_.append(ys[d] + dy)
            err_hi.append(st.hi - st.p)
            err_lo.append(st.p - st.lo)
            hover.append(f"{d}<br>{NAMES[c]}: {st.p:.2f} (n={st.n})")
        fig.add_trace(go.Scatter(
            x=xs_, y=ys_, mode="markers",
            marker=dict(color=COLORS[c], size=6,
                        line=dict(color=EDGES[c], width=0.8)),
            error_x=dict(type="data", symmetric=False, array=err_hi,
                         arrayminus=err_lo, thickness=1.1, width=0,
                         color=EDGES[c]),
            name=NAMES[c], legendgroup=c,
            hovertext=hover, hoverinfo="text",
        ))

    div_y = len(meta) + GAP / 2
    fig.add_hline(y=div_y, line_dash="dot", line_color="#bbb", line_width=1)
    fig.add_annotation(
        text="<i>audit-quality / meta dimensions</i>", xref="x domain", yref="y",
        x=1, y=div_y - 0.15, xanchor="right", yanchor="top",
        showarrow=False, font=dict(size=11, color="#888"),
    )

    top = len(behavior) + len(meta) + GAP
    max_mean = max(stats[d][c].hi for d in dims for c in CKPTS)
    fig.update_yaxes(
        tickvals=[ys[d] for d in behavior + meta],
        ticktext=[d.replace("_", " ") for d in behavior + meta],
        tickfont=dict(size=11),
        range=[-0.7, top - 0.3], gridcolor="#eeece6", automargin=True,
    )
    fig.update_xaxes(
        title=dict(text="mean judge score (1–10)", font=dict(size=12.5)),
        range=[0.85, max_mean + 0.25], dtick=1,
        tickfont=dict(size=10), gridcolor="#e8e6df",
    )
    fig.update_layout(
        paper_bgcolor=CREAM, plot_bgcolor=CREAM,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.005,
                    yanchor="bottom", font=dict(size=12)),
        width=780, height=30 + 21 * (len(behavior) + len(meta)) + 84,
        margin=dict(t=40, b=48, l=10, r=16),
    )
    fig.write_html(ROOT / "figures" / f"{out_stem}.html", include_plotlyjs=True)
    fig.write_image(ROOT / "figures" / f"{out_stem}.png", scale=2)
    print(f"wrote figures/{out_stem}.png")


# ---------------------------------------------------------------- main
def main():
    print("collecting ...")
    odcv = odcv_stats()
    pvals = petri_values()
    sr = strongreject_stats()
    ib = impossiblebench_stats()
    mask = mask_stats()

    def petri_group(label, dim):
        return (label, {c: petri_dim_mean(pvals, c, dim) for c in CKPTS})

    # (eval name, title parenthetical or None, axis kind, groups). Petri leads so its
    # score axis sits at the far left and can't be read as belonging to a rate panel.
    panels = [
        ("Petri", None, "score", [
            petri_group("concerning", "concerning"),
            petri_group("eval awareness", "eval_awareness"),
            petri_group("overrefusal", "overrefusal"),
            petri_group("unprompted sycophancy", "unprompted_sycophancy"),
        ]),
        ("ODCV-Bench", None, "pct",
         [("misbehavior rate", odcv)]),
        ("StrongREJECT", None, "pct",
         [("unsafe response rate", sr)]),
        ("ImpossibleBench", "reward hacking", "pct", [
            ("one-off tests", {c: ib[c]["oneoff"] for c in CKPTS}),
            ("conflicting tests", {c: ib[c]["conflicting"] for c in CKPTS}),
        ]),
        ("MASK", None, "pct", [
            ("lie rate", {c: mask[c]["lie"] for c in CKPTS}),
            ("evasion rate", {c: mask[c]["evade"] for c in CKPTS}),
        ]),
        ("LiveCodeBench", "capability control", "pct",
         [("pass rate", {c: ib[c]["original"] for c in CKPTS})]),
    ]

    print("\n=== numbers ===")
    for title, _, kind, groups in panels:
        for label, stats in groups:
            unit = "%" if kind == "pct" else ""
            cells = "  ".join(
                f"{c}={stats[c].p:.2f}{unit}" for c in CKPTS)
            print(f"{title:16s} {label.replace(chr(10), ' '):22s} {cells}")

    (ROOT / "figures").mkdir(exist_ok=True)
    make_main(panels, "fig_main")
    make_petri_dims(pvals, "fig_petri_dims")


if __name__ == "__main__":
    main()
