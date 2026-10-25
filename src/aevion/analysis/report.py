"""One-call analysis report for a run: figures, metrics tables, written insights, manifest.

``build_report(archive, outdir)`` writes research-grade figures (PDF + PNG), metric tables (CSV, LaTeX,
Markdown), an ``insights.md`` of automatically derived findings, and a self-contained ``report.html``.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import metrics as MT
from . import figures as F
from . import style as S


# --------------------------------------------------------------------------- insights
def insights(archive, vehicle: int = 0) -> List[dict]:
    """Derive plain-language findings from a run. Each item: ``{level, title, text}`` (level: good/info/warn/bad)."""
    out: List[dict] = []
    add = lambda level, title, text: out.append(dict(level=level, title=title, text=text))  # noqa: E731
    v = vehicle
    t = archive.t - archive.t[0]
    pos, vel = archive["pos"][:, v], archive["vel"][:, v]
    m = MT.flight_metrics(archive)

    def g(k: str, default: float = float("nan")) -> float:
        return float(m[k][v]) if k in m else default

    spec = archive.spec

    if not np.isfinite(pos).all():
        add("bad", "Numerical failure", "Non-finite values appear in the trajectory: the simulation diverged.")
        return out

    # ---- tracking
    if "ref_pos" in archive:
        e = np.linalg.norm(pos - archive["ref_pos"][:, v], axis=-1)
        rmse, p95, emax = g("rmse_pos"), float(np.quantile(e, 0.95)), float(e.max())
        lvl = "good" if rmse < 0.05 else "info" if rmse < 0.25 else "warn" if rmse < 1.0 else "bad"
        add(
            lvl,
            "Tracking accuracy",
            f"3-D RMSE {rmse * 100:.1f} cm (horizontal {g('rmse_xy') * 100:.1f} cm, vertical {g('rmse_z') * 100:.1f} cm); "
            f"95th percentile {p95 * 100:.1f} cm; worst {emax * 100:.1f} cm at t = {t[e.argmax()]:.2f} s.",
        )
        ss = g("rmse_steady")
        if np.isfinite(ss) and rmse > 0 and ss / rmse < 0.7:
            add(
                "info",
                "Transient vs. steady state",
                f"Error shrinks from {rmse * 100:.1f} cm overall to {ss * 100:.1f} cm in the second half of the run, so a start-up transient dominates the overall figure.",
            )
        if len(t) > 16:
            from scipy.signal import welch

            fs = 1 / np.median(np.diff(t))
            err = pos - archive["ref_pos"][:, v]
            f, P = welch(err[:, 0] - err[:, 0].mean(), fs=fs, nperseg=min(len(t), max(64, len(t) // 4)))
            if len(f) > 2 and P[1:].sum() > 0:
                fd = float(f[1:][P[1:].argmax()])
                add(
                    "info",
                    "Dominant error frequency",
                    f"The horizontal tracking error is concentrated near {fd:.2f} Hz"
                    + (f" (trajectory period ≈ {1 / fd:.1f} s)." if fd > 0 else "."),
                )
        sp = np.linalg.norm(vel, axis=-1)
        if len(sp) > 8 and np.std(sp) > 1e-6 and np.std(e) > 1e-9:
            c = float(np.corrcoef(sp, e)[0, 1])
            if abs(c) > 0.5:
                add(
                    "info",
                    "Speed dependence",
                    f"Tracking error is {'strongly ' if abs(c) > 0.75 else ''}{'positively' if c > 0 else 'negatively'} correlated with speed (r = {c:+.2f}).",
                )

    # ---- actuators
    if spec is not None:
        rot = archive["rotor"][:, v]
        frac = g("rotor_saturation_frac", 0.0)
        peak = float(rot.max() / spec.w_max)
        if frac > 0.02:
            add(
                "warn",
                "Actuator saturation",
                f"At least one rotor sits above 98 % of its speed limit for {frac * 100:.1f} % of the flight; demanded manoeuvres may exceed vehicle authority.",
            )
        else:
            add(
                "good",
                "Actuator headroom",
                f"Peak rotor speed is {peak * 100:.0f} % of maximum; thrust margin {g('thrust_margin', 0.0) * 100:.0f} %.",
            )
        spread = float((rot.max(1) - rot.min(1)).mean() / max(rot.mean(), 1e-9))
        if spread > 0.15:
            add(
                "info",
                "Rotor imbalance",
                f"Rotor speeds differ by {spread * 100:.0f} % on average, consistent with aggressive attitude control, wind, or a centre-of-mass offset.",
            )

    # ---- attitude
    tilt = g("max_tilt_deg")
    add(
        "good" if tilt < 30 else "info" if tilt < 50 else "warn",
        "Attitude excursions",
        f"Maximum roll/pitch {tilt:.1f}°, peak body rate {g('max_rate_deg_s'):.0f} °/s.",
    )

    # ---- energy
    mp, wh = g("mean_power_w"), g("energy_wh")
    txt = f"{wh:.3f} Wh consumed at {mp:.0f} W average; battery state of charge ends at {g('soc_end') * 100:.1f} %."
    if spec is not None and mp > 0:
        hp = spec.hover_power()
        txt += f" Average power is {(mp / hp - 1) * 100:+.0f} % relative to ideal hover ({hp:.0f} W)."
        rem = spec.battery.capacity_wh * g("soc_end") / mp * 60
        txt += f" At this rate the remaining charge lasts about {rem:.0f} min."
    add("info", "Energy", txt)

    # ---- wind
    w = np.linalg.norm(archive["wind"][:, v], axis=-1)
    if w.max() > 0.2:
        txt = f"Wind averages {w.mean():.1f} m/s (max {w.max():.1f} m/s)."
        if "ref_pos" in archive and np.std(w) > 1e-6:
            e = np.linalg.norm(pos - archive["ref_pos"][:, v], axis=-1)
            c = float(np.corrcoef(w, e)[0, 1])
            txt += f" Correlation between wind speed and tracking error: r = {c:+.2f}."
        add("info", "Wind", txt)

    # ---- safety
    ev = archive.meta.get("events") or []
    cols = [e for e in ev if e["type"] == "collision" and e.get("vehicle", 0) == v]
    if cols:
        add("bad", "Collisions", f"{len(cols)} collision step(s); first at t = {cols[0]['t'] - archive.t[0]:.2f} s.")
    elif "min_clearance" in m:
        c = g("min_clearance")
        add("good" if c > 1.0 else "warn", "Obstacle clearance", f"Closest approach to any obstacle surface was {c:.2f} m.")
    if g("crashed", 0.0):
        add("bad", "Ground contact", "The vehicle touched the ground after take-off (crash or landing).")
    t0 = float(archive.t[0])
    for e in ev:
        if e.get("name") == "fault.start" or e["type"] == "rotor_failure":
            detail = ", ".join(f"{k} = {v}" for k, v in e.items() if k not in ("type", "t", "name", "fault", "vehicle"))
            who = f" on vehicle {e['vehicle']}" if e.get("vehicle") is not None else ""
            add(
                "warn",
                "Injected fault",
                f"{e.get('fault', e['type']).replace('_', ' ')}{who} began at t = {e['t'] - t0:.2f} s" + (f" ({detail})." if detail else "."),
            )
    for e in ev:
        if e["type"] == "crash":
            add("bad", "Crash", f"Hard ground impact ({e.get('speed', 0):.1f} m/s) at t = {e['t'] - t0:.2f} s.")
            break
    for e in ev:
        if e["type"] == "battery.low":
            add("warn", "Battery low", f"State of charge fell below 20 % at t = {e['t'] - t0:.2f} s.")
            break
    if archive.meta.get("status") not in (None, "complete"):
        add(
            "bad",
            "Incomplete run",
            f"The run ended with status '{archive.meta.get('status')}'" + (f": {archive.meta['error']}" if archive.meta.get("error") else "") + ".",
        )
    if archive.meta.get("stop_reason"):
        add("info", "Early stop", f"The run was stopped early: {archive.meta['stop_reason']}.")

    # ---- estimation / comms
    if "est_pos" in archive:
        from ..core import rotation as rot_
        from ..core.backend import get_backend

        ep = np.linalg.norm(archive["est_pos"][:, v] - pos, axis=-1)
        ang = np.degrees(rot_.angle_between(get_backend("numpy"), archive["est_quat"][:, v], archive["quat"][:, v]))
        add("info", "State estimation", f"Estimator position error RMS {np.sqrt((ep**2).mean()):.2f} m, mean attitude error {ang.mean():.1f}°.")
    if "link_sinr_db" in archive:
        s = archive["link_sinr_db"][:, v]
        thr = (archive.meta.get("wireless") or {}).get("outage_db", -6.0)
        out_frac = float((s < thr).mean())
        add(
            "warn" if out_frac > 0.05 else "good",
            "Wireless link",
            f"Median SINR {np.median(s):.1f} dB (5th percentile {np.quantile(s, 0.05):.1f} dB); outage {out_frac * 100:.1f} % of the time; "
            f"mean throughput {archive['link_rate_mbps'][:, v].mean():.1f} Mbit/s; line-of-sight {archive['link_los'][:, v].mean() * 100:.0f} % of the time.",
        )
    if "link_n_handovers" in archive:
        ho = int(archive["link_n_handovers"][-1, v])
        rlf = int(archive["link_n_rlf"][-1, v]) if "link_n_rlf" in archive else 0
        dur = float(archive.t[-1] - archive.t[0]) or 1.0
        add(
            "warn" if rlf else "info",
            "Mobility management",
            f"{ho} handover(s) ({ho / dur * 60:.1f} per minute) and {rlf} radio-link failure(s); "
            f"{int(archive['link_ho_interrupted'][:, v].sum()) if 'link_ho_interrupted' in archive else 0} recorded samples were spent in handover interruption.",
        )
    if "sensors/lidar" in archive:
        r = archive["sensors/lidar"][:, v]
        add("info", "LiDAR", f"{len(r)} scans recorded; {np.isfinite(r).mean() * 100:.0f} % of beams returned a hit.")
    return out


# ------------------------------------------------------------------------------ tables
METRIC_TABLE = [
    ("rmse_pos", "RMSE position", "m", 1.0, "{:.4f}"),
    ("rmse_xy", "RMSE horizontal", "m", 1.0, "{:.4f}"),
    ("rmse_z", "RMSE vertical", "m", 1.0, "{:.4f}"),
    ("max_err", "Max position error", "m", 1.0, "{:.4f}"),
    ("path_length", "Path length", "m", 1.0, "{:.2f}"),
    ("mean_speed", "Mean speed", "m/s", 1.0, "{:.2f}"),
    ("max_speed", "Max speed", "m/s", 1.0, "{:.2f}"),
    ("max_acc", "Max acceleration", "m/s²", 1.0, "{:.2f}"),
    ("max_tilt_deg", "Max tilt", "deg", 1.0, "{:.1f}"),
    ("max_rate_deg_s", "Max body rate", "deg/s", 1.0, "{:.0f}"),
    ("rms_jerk", "RMS jerk", "m/s³", 1.0, "{:.1f}"),
    ("energy_wh", "Energy", "Wh", 1.0, "{:.4f}"),
    ("mean_power_w", "Mean power", "W", 1.0, "{:.1f}"),
    ("soc_end", "Final state of charge", "", 100.0, "{:.1f}%"),
    ("rotor_saturation_frac", "Rotor saturation time", "", 100.0, "{:.1f}%"),
    ("thrust_margin", "Thrust margin", "", 100.0, "{:.0f}%"),
    ("collisions", "Collisions", "", 1.0, "{:.0f}"),
    ("min_clearance", "Min obstacle clearance", "m", 1.0, "{:.2f}"),
    ("link_mean_sinr_db", "Mean link SINR", "dB", 1.0, "{:.1f}"),
    ("link_outage_frac", "Link outage time", "", 100.0, "{:.1f}%"),
    ("link_mean_rate_mbps", "Mean link goodput", "Mbit/s", 1.0, "{:.1f}"),
    ("link_p5_rate_mbps", "5th-percentile goodput", "Mbit/s", 1.0, "{:.1f}"),
    ("link_handovers", "Handovers", "", 1.0, "{:.0f}"),
    ("link_rlf", "Radio-link failures", "", 1.0, "{:.0f}"),
    ("link_mean_bler", "Mean first-transmission BLER", "", 100.0, "{:.1f}%"),
]


def metrics_rows(archive, vehicle: int = 0) -> List[dict]:
    m = MT.flight_metrics(archive)
    rows = []
    for key, label, unit, scale, fmt in METRIC_TABLE:
        if key in m:
            rows.append(dict(key=key, label=label, unit=unit, value=fmt.format(float(m[key][vehicle]) * scale)))
    return rows


def metrics_latex(rows, caption="Flight metrics", label="tab:metrics") -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Metric & Value & Unit \\",
        r"\midrule",
    ]
    for r in rows:
        value = r["value"].replace("%", "\\%")
        unit = r["unit"].replace("²", "$^2$").replace("³", "$^3$")
        lines.append(f"{r['label']} & {value} & {unit} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def metrics_markdown(rows) -> str:
    return "\n".join(["| Metric | Value | Unit |", "|---|---:|---|"] + [f"| {r['label']} | {r['value']} | {r['unit']} |" for r in rows])


# ---------------------------------------------------------------------------- report
def _png_b64(fig, dpi=200) -> str:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return base64.b64encode(buf.getvalue()).decode("ascii")


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title} | Aevion report</title>
<style>
body{{font-family:"Times New Roman",Times,serif;max-width:1000px;margin:2.2rem auto;padding:0 1.2rem;color:#111;line-height:1.45}}
h1{{font-size:1.6rem;margin:.2rem 0}} h2{{font-size:1.15rem;border-bottom:1px solid #999;padding-bottom:.2rem;margin-top:2rem}}
.meta{{color:#555;font-size:.9rem}} table{{border-collapse:collapse;margin:.6rem 0}} th,td{{padding:.25rem .8rem;text-align:left}}
thead th{{border-top:2px solid #000;border-bottom:1px solid #000}} tbody tr:last-child td{{border-bottom:2px solid #000}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
figure{{margin:1.2rem 0;text-align:center}} figure img{{max-width:100%}} figcaption{{font-size:.92rem;color:#333;margin-top:.3rem}}
.ins{{list-style:none;padding:0}} .ins li{{margin:.45rem 0;padding:.45rem .7rem;border-left:4px solid #888;background:#f5f5f5}}
.ins li.good{{border-color:#2a9d5c}} .ins li.warn{{border-color:#d9822b}} .ins li.bad{{border-color:#c0392b}} .ins li.info{{border-color:#2b6cb0}}
code,pre{{font-family:Consolas,monospace;font-size:.85rem}} pre{{background:#f5f5f5;padding:.6rem;overflow:auto}} a{{color:#0b5cad}}
</style></head><body>
<h1>{title}</h1><div class="meta">{subtitle}</div>
<h2>Key findings</h2><ul class="ins">{insights}</ul>
<h2>Flight metrics</h2><table><thead><tr><th>Metric</th><th>Value</th><th>Unit</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Figures</h2>{figs}
<h2>Reproducibility</h2><p>Everything needed to reproduce this run is stored in the archive manifest; <code>aevion verify run.aev</code> re-simulates it and compares channel by channel.</p>
<pre>{manifest}</pre>
{replay}
</body></html>"""

CAPTIONS = {
    "overview": "Overview: (a) 3-D path coloured by speed, (b) top view, (c) position error components, (d) attitude, (e) rotor speeds, (f) power and state of charge.",
    "tracking": "Position tracking against the reference and error norm.",
    "attitude": "Attitude angles and body rates.",
    "rotors": "Rotor speeds and total thrust.",
    "energy": "Electrical power, cumulative energy and battery state of charge.",
    "error_distribution": "Distribution and empirical CDF of the tracking error.",
    "psd_error": "Power spectral density of the tracking error (Welch).",
    "phase": "Position–velocity phase portraits coloured by time.",
    "wind": "Wind disturbance experienced by the vehicle.",
    "lidar_scan": "Final LiDAR scan.",
    "imu": "Recorded IMU stream.",
    "estimation": "State-estimation error against ground truth.",
    "link": "Wireless link quality along the trajectory.",
    "link_stack": "Radio link through the protocol stack: SINR and outage, goodput and MCS, BLER, serving cell with handovers (dashed) and radio-link failures (crosses).",
    "outage_cdf": "Empirical CDF of the link SINR; the dotted line is the outage threshold.",
}


def build_report(
    archive,
    outdir,
    vehicle: int = 0,
    formats=("pdf", "png"),
    title: Optional[str] = None,
    venue: str = "ieee",
    with_replay: bool = True,
    offline_replay: bool = False,
) -> Path:
    """Generate the report directory and return the path of ``report.html``."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    out = Path(outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    title = title or archive.scenario.get("name") or "Aevion run"
    names = [
        "overview",
        "tracking" if "ref_pos" in archive else None,
        "attitude",
        "rotors",
        "energy",
        "error_distribution" if "ref_pos" in archive else None,
        "psd_error" if "ref_pos" in archive else None,
        "phase",
    ]
    if np.linalg.norm(archive["wind"][:, vehicle], axis=-1).max() > 0.2:
        names.append("wind")
    if "sensors/lidar" in archive:
        names.append("lidar_scan")
    if "sensors/imu" in archive:
        names.append("imu")
    if "est_pos" in archive:
        names.append("estimation")
    if "link_sinr_db" in archive:
        names.append("link_stack" if "link_mcs" in archive else "link")
        names.append("outage_cdf")
    figs_html = []
    for i, n in enumerate([x for x in names if x]):
        try:
            fig = F.ALL_RUN_FIGURES[n](archive, vehicle=vehicle, venue=venue)
        except Exception as ex:  # keep the report going if one panel fails
            figs_html.append(f"<p><em>Figure '{n}' unavailable: {html.escape(str(ex))}</em></p>")
            continue
        S.save(fig, out / "figures" / n, formats=formats)
        figs_html.append(
            f"<figure><img alt='{n}' src='data:image/png;base64,{_png_b64(fig)}'><figcaption><b>Fig. {i + 1}.</b> {CAPTIONS.get(n, n)}</figcaption></figure>"
        )
        import matplotlib.pyplot as plt

        plt.close(fig)

    rows = metrics_rows(archive, vehicle)
    (out / "metrics.csv").write_text("metric,value,unit\n" + "\n".join(f"{r['label']},{r['value']},{r['unit']}" for r in rows), encoding="utf-8")
    (out / "metrics.tex").write_text(metrics_latex(rows), encoding="utf-8")
    (out / "metrics.md").write_text(metrics_markdown(rows), encoding="utf-8")
    ins = insights(archive, vehicle)
    (out / "insights.md").write_text("\n".join(f"- {i['title']}: {i['text']}" for i in ins), encoding="utf-8")

    replay_html = ""
    if with_replay:
        from ..viewer import export_html

        try:
            export_html(archive, out / "replay.html", offline=offline_replay, title=title)
            replay_html = "<h2>Interactive replay</h2><p><a href='replay.html'>Open the 3-D replay</a> (WebGL; works in any modern browser).</p>"
        except Exception as ex:  # pragma: no cover
            replay_html = f"<p><em>Replay export failed: {html.escape(str(ex))}</em></p>"
    meta = {k: v for k, v in archive.meta.items() if k not in ("world", "vehicle")}
    meta["vehicle"] = {k: archive.meta.get("vehicle", {}).get(k) for k in ("name", "mass", "kT", "kQ", "w_max", "motor_tau")}
    sub = f"{archive.meta.get('vehicle', {}).get('name', '')} · {archive.n_vehicles} vehicle(s) · {archive.t[-1] - archive.t[0]:.1f} s · {archive.meta.get('created', '')[:19]}"
    page = HTML.format(
        title=html.escape(title),
        subtitle=html.escape(sub),
        insights="".join(f"<li class='{i['level']}'><b>{html.escape(i['title'])}.</b> {html.escape(i['text'])}</li>" for i in ins),
        rows="".join(
            f"<tr><td>{html.escape(r['label'])}</td><td class='n'>{html.escape(r['value'])}</td><td>{html.escape(r['unit'])}</td></tr>" for r in rows
        ),
        figs="".join(figs_html),
        manifest=html.escape(json.dumps(meta, indent=1, default=str)[:6000]),
        replay=replay_html,
    )
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
