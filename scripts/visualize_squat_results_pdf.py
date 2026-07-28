#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

OK_COLOR = "#2e9d43"
NG_COLOR = "#d73027"
LINE_COLOR = "#2369a1"
OK_BG = "#e7f5e9"
NG_BG = "#fdeaea"
TEXT_COLOR = "#202020"
MUTED_COLOR = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--rep-plot", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report-pdf", type=Path, default=Path("outputs/squat_report.pdf"))
    parser.add_argument("--depth-threshold", type=float, default=0.0)
    parser.add_argument("--deep-angle", type=float, default=105.0)
    parser.add_argument("--rounding-threshold", type=float, default=7.0)
    parser.add_argument("--torso-threshold", type=float, default=40.0)
    parser.add_argument("--heel-threshold", type=float, default=0.15)
    parser.add_argument("--toe-threshold", type=float, default=0.85)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))


def find_repetition_ranges(df: pd.DataFrame) -> list[tuple[int, int]]:
    phases = df["phase"].fillna("").astype(str).str.upper().to_numpy()
    result: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(1, len(phases)):
        if phases[i - 1] == "UP" and phases[i] == "DOWN":
            start = i
        elif phases[i - 1] == "DOWN" and phases[i] == "UP" and start is not None:
            result.append((start, i))
            start = None
    return result


def finite_max(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.max(v)) if len(v) else float("nan")


def finite_min(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.min(v)) if len(v) else float("nan")


def deepest_frame_index(rep_df: pd.DataFrame) -> int:
    depth = pd.to_numeric(rep_df["squat_depth_ratio"], errors="coerce")
    if depth.notna().any():
        return int(depth.idxmax())
    knee = pd.to_numeric(rep_df["knee_angle_deg"], errors="coerce")
    if knee.notna().any():
        return int(knee.idxmin())
    return int(rep_df.index[len(rep_df) // 2])


def foot_balance_label(min_ratio: float, max_ratio: float, heel: float, toe: float) -> str:
    if np.isfinite(max_ratio) and max_ratio > toe:
        return "Too much toes"
    if np.isfinite(min_ratio) and min_ratio < heel:
        return "Too much heels"
    if np.isfinite(min_ratio) or np.isfinite(max_ratio):
        return "Mid-foot"
    return "N/A"


def summarize_repetitions(df: pd.DataFrame, ranges: list[tuple[int, int]], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for rep, (start, end) in enumerate(ranges, start=1):
        rep_df = df.iloc[start:end + 1]
        deepest_idx = deepest_frame_index(rep_df)
        deepest_row = df.loc[deepest_idx]

        max_depth_ratio = finite_max(rep_df["squat_depth_ratio"])
        min_knee_angle = finite_min(rep_df["knee_angle_deg"])
        max_torso_lean = finite_max(rep_df["torso_lean_deg"])
        max_upper_back_drop = finite_max(rep_df["upper_back_drop_deg"]) if "upper_back_drop_deg" in rep_df.columns else float("nan")
        min_balance_ratio = finite_min(rep_df["balance_ratio"]) if "balance_ratio" in rep_df.columns else float("nan")
        max_balance_ratio = finite_max(rep_df["balance_ratio"]) if "balance_ratio" in rep_df.columns else float("nan")

        depth_ok = ((np.isfinite(max_depth_ratio) and max_depth_ratio >= args.depth_threshold) or
                    (np.isfinite(min_knee_angle) and min_knee_angle <= args.deep_angle))
        torso_ok = (not np.isfinite(max_torso_lean) or max_torso_lean <= args.torso_threshold)
        upper_back_ok = (not np.isfinite(max_upper_back_drop) or max_upper_back_drop <= args.rounding_threshold)
        balance_ok = True
        if np.isfinite(min_balance_ratio):
            balance_ok = balance_ok and min_balance_ratio >= args.heel_threshold
        if np.isfinite(max_balance_ratio):
            balance_ok = balance_ok and max_balance_ratio <= args.toe_threshold

        balance_label = foot_balance_label(min_balance_ratio, max_balance_ratio, args.heel_threshold, args.toe_threshold)

        errors = []
        if not depth_ok:
            errors.append("Shallow squat")
        if not torso_ok:
            errors.append("Excessive torso lean")
        if not upper_back_ok:
            errors.append("Upper-back rounding")
        if not balance_ok:
            if balance_label == "Too much toes":
                errors.append("Weight shifted to toes")
            elif balance_label == "Too much heels":
                errors.append("Weight shifted to heels")
            else:
                errors.append("Foot balance shift")

        overall_ok = depth_ok and torso_ok and upper_back_ok and balance_ok

        rows.append({
            "repetition": rep,
            "start_time_sec": float(rep_df["time_sec"].iloc[0]),
            "end_time_sec": float(rep_df["time_sec"].iloc[-1]),
            "deepest_time_sec": float(deepest_row["time_sec"]),
            "deepest_frame": int(deepest_row["frame"]),
            "max_depth_ratio": max_depth_ratio,
            "min_knee_angle_deg": min_knee_angle,
            "max_torso_lean_deg": max_torso_lean,
            "max_upper_back_drop_deg": max_upper_back_drop,
            "min_balance_ratio": min_balance_ratio,
            "max_balance_ratio": max_balance_ratio,
            "foot_balance_label": balance_label,
            "depth_ok": depth_ok,
            "torso_ok": torso_ok,
            "upper_back_ok": upper_back_ok,
            "balance_ok": balance_ok,
            "overall_ok": overall_ok,
            "errors": "; ".join(errors),
        })
    return pd.DataFrame(rows)


def draw_rep_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    row: pd.Series,
) -> None:
    """
    各スクワットの評価カードを描画する．

    Icons:
        ✓ : OK
        ↓ : 浅い
        → : つま先側
        ← : かかと側
        ↘ : 体幹前傾
        ⌒ : 背中の丸まり
    """
    is_ok = bool(row["overall_ok"])

    edge_color = (
        OK_COLOR
        if is_ok
        else NG_COLOR
    )

    face_color = (
        "#f4fff5"
        if is_ok
        else "#fff7f7"
    )

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            "round,pad=0.008,"
            "rounding_size=0.012"
        ),
        linewidth=1.0,
        edgecolor=edge_color,
        facecolor=face_color,
        transform=ax.transAxes,
        clip_on=False,
    )

    ax.add_patch(card)

    repetition = int(
        row["repetition"]
    )

    status = (
        "OK"
        if is_ok
        else "NG"
    )

    # Rep番号
    ax.text(
        x + w / 2,
        y + h * 0.82,
        f"Rep {repetition}",
        ha="center",
        va="center",
        fontsize=7.5,
        weight="bold",
        color=edge_color,
        transform=ax.transAxes,
    )

    # 大きなステータスアイコン
    status_icon = (
        "✓"
        if is_ok
        else "!"
    )

    ax.text(
        x + w / 2,
        y + h * 0.57,
        status_icon,
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=edge_color,
        transform=ax.transAxes,
    )

    ax.text(
        x + w / 2,
        y + h * 0.39,
        status,
        ha="center",
        va="center",
        fontsize=7.5,
        weight="bold",
        color=edge_color,
        transform=ax.transAxes,
    )

    # 主なエラーをアイコン付きで表示
    if is_ok:
        detail = "All passed"

    else:
        errors = str(
            row["errors"]
        )

        balance = str(
            row["foot_balance_label"]
        )

        if "Shallow squat" in errors:
            detail = "↓ Shallow"

        elif (
            balance
            == "Too much toes"
        ):
            detail = "→ Toes"

        elif (
            balance
            == "Too much heels"
        ):
            detail = "← Heels"

        elif (
            "Excessive torso lean"
            in errors
        ):
            detail = "↘ Torso"

        elif (
            "Upper-back rounding"
            in errors
        ):
            detail = "⌒ Back"

        else:
            detail = "Check form"

    ax.text(
        x + w / 2,
        y + h * 0.15,
        detail,
        ha="center",
        va="center",
        fontsize=6.7,
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )


def create_timeline_plot(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path, deep_angle: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = pd.to_numeric(df["time_sec"], errors="coerce")
    knee = pd.to_numeric(df["knee_angle_deg"], errors="coerce")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(time, knee, linewidth=2.0, color=LINE_COLOR, label="Knee angle", zorder=3)
    ax.axhline(deep_angle, color="#555555", linestyle="--", linewidth=1.3,
               label=f"Depth criterion ({deep_angle:.0f} deg)")
    for _, row in summary.iterrows():
        rep = int(row["repetition"])
        is_ok = bool(row["overall_ok"])
        color = OK_COLOR if is_ok else NG_COLOR
        ax.axvspan(float(row["start_time_sec"]), float(row["end_time_sec"]),
                   color=OK_BG if is_ok else NG_BG, alpha=0.8)
        ax.scatter(float(row["deepest_time_sec"]), float(row["min_knee_angle_deg"]),
                   s=95, color=color, edgecolor="white", linewidth=1.5, zorder=5)
        text = f"Rep {rep}: {'OK' if is_ok else 'NG'}"
        if not is_ok:
            text += "\n" + str(row["errors"]).replace("; ", "\n")
        y_offset = -90 - 45 * ((rep - 1) % 3)
        ax.annotate(text,
                    xy=(float(row["deepest_time_sec"]), float(row["min_knee_angle_deg"])),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.0, connectionstyle="angle3"),
                    annotation_clip=False, zorder=6)
    ax.set_title("Squat Form Analysis", fontsize=16)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Knee angle [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.38)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_repetition_plot(summary: pd.DataFrame, output_path: Path, deep_angle: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reps = summary["repetition"].astype(int).to_numpy()
    values = summary["min_knee_angle_deg"].to_numpy(dtype=float)
    colors = [OK_COLOR if bool(v) else NG_COLOR for v in summary["overall_ok"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(reps, values, color=colors)
    ax.axhline(deep_angle, linestyle="--", linewidth=1.3, color="#555555")
    ax.set_title("Minimum Knee Angle per Repetition")
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Minimum knee angle [deg]")
    ax.set_xticks(reps)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_pdf_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:

    args.report_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    time = pd.to_numeric(
        df["time_sec"],
        errors="coerce",
    )

    knee = pd.to_numeric(
        df["knee_angle_deg"],
        errors="coerce",
    )

    # 横長のレポート
    fig = plt.figure(
        figsize=(17, 10.5)
    )

    gs = GridSpec(
        4,
        12,
        figure=fig,
        height_ratios=[
            5.1,   # 時系列
            1.35,  # Repカード
            2.65,  # 凡例＋棒グラフ
            0.30,  # 注釈
        ],
        hspace=0.52,
        wspace=0.95,
    )

    # =====================================
    # 1. 時系列グラフ
    # =====================================
    ax_main = fig.add_subplot(
        gs[0, :]
    )

    ax_main.plot(
        time,
        knee,
        linewidth=2.0,
        color=LINE_COLOR,
        label="Knee angle",
        zorder=3,
    )

    ax_main.axhline(
        args.deep_angle,
        color="#666666",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label=(
            "Depth criterion "
            f"({args.deep_angle:.0f} deg)"
        ),
        zorder=1,
    )

    for _, row in summary.iterrows():

        repetition = int(
            row["repetition"]
        )

        start_time = float(
            row["start_time_sec"]
        )

        end_time = float(
            row["end_time_sec"]
        )

        deepest_time = float(
            row["deepest_time_sec"]
        )

        deepest_angle = float(
            row["min_knee_angle_deg"]
        )

        is_ok = bool(
            row["overall_ok"]
        )

        background_color = (
            OK_BG
            if is_ok
            else NG_BG
        )

        point_color = (
            OK_COLOR
            if is_ok
            else NG_COLOR
        )

        # スクワット区間
        ax_main.axvspan(
            start_time,
            end_time,
            color=background_color,
            alpha=0.72,
            zorder=0,
        )

        # 最深部
        ax_main.scatter(
            deepest_time,
            deepest_angle,
            s=90,
            color=point_color,
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )

        # 最深部からグラフ下部へ接続線
        label_y = 30.5

        ax_main.plot(
            [
                deepest_time,
                deepest_time,
            ],
            [
                deepest_angle,
                label_y + 2,
            ],
            color=point_color,
            linestyle=":",
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )

        # Rep番号を小さな丸として表示
        ax_main.text(
            deepest_time,
            label_y,
            str(repetition),
            ha="center",
            va="center",
            fontsize=7.0,
            weight="bold",
            color="white",
            bbox=dict(
                boxstyle="circle,pad=0.28",
                facecolor=point_color,
                edgecolor="white",
                linewidth=0.8,
            ),
            zorder=6,
        )

    ax_main.set_title(
        "Squat Form Analysis",
        fontsize=19,
        weight="bold",
        pad=12,
    )

    ax_main.set_xlabel(
        "Time [s]",
        fontsize=11,
        labelpad=8,
    )

    ax_main.set_ylabel(
        "Knee angle [deg]",
        fontsize=11,
    )

    # Rep番号を置くため下側に余裕を作る
    ax_main.set_ylim(
        24,
        min(
            190,
            float(
                np.nanmax(knee)
            ) + 5,
        ),
    )

    ax_main.grid(
        True,
        alpha=0.23,
        zorder=0,
    )

    ax_main.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    ax_main.spines[
        "top"
    ].set_visible(False)

    ax_main.spines[
        "right"
    ].set_visible(False)

    # =====================================
    # 2. Repごとの評価カード
    # =====================================
    ax_cards = fig.add_subplot(
        gs[1, :]
    )

    ax_cards.axis("off")

    number_of_reps = max(
        len(summary),
        1,
    )

    # カード間に余白を確保
    total_gap = 0.010 * (
        number_of_reps - 1
    )

    card_width = (
        1.0 - total_gap
    ) / number_of_reps

    card_height = 0.84

    for index, (_, row) in enumerate(
        summary.iterrows()
    ):
        x = index * (
            card_width + 0.010
        )

        draw_rep_card(
            ax_cards,
            x=x,
            y=0.08,
            w=card_width,
            h=card_height,
            row=row,
        )

    # =====================================
    # 3. 左下：凡例と評価基準
    # =====================================
    ax_legend = fig.add_subplot(
        gs[2, :5]
    )

    ax_legend.axis("off")

    legend_box = FancyBboxPatch(
        (0.02, 0.08),
        0.94,
        0.84,
        boxstyle=(
            "round,pad=0.02,"
            "rounding_size=0.02"
        ),
        linewidth=1.0,
        edgecolor="#cccccc",
        facecolor="#fafafa",
        transform=ax_legend.transAxes,
    )

    ax_legend.add_patch(
        legend_box
    )

    ax_legend.text(
        0.06,
        0.84,
        "Form indicators",
        fontsize=10.5,
        weight="bold",
        transform=ax_legend.transAxes,
    )

    # Foot balance
    ax_legend.text(
        0.07,
        0.70,
        "←",
        fontsize=15,
        weight="bold",
        color=NG_COLOR,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.13,
        0.705,
        "Too much heels",
        fontsize=7.8,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.43,
        0.70,
        "✓",
        fontsize=14,
        weight="bold",
        color=OK_COLOR,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.49,
        0.705,
        "Mid-foot",
        fontsize=7.8,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.70,
        0.70,
        "→",
        fontsize=15,
        weight="bold",
        color=NG_COLOR,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.76,
        0.705,
        "Too much toes",
        fontsize=7.8,
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.06,
        0.54,
        "Evaluation criteria",
        fontsize=10.5,
        weight="bold",
        transform=ax_legend.transAxes,
    )

    criteria_lines = [
        (
            f"↓  Depth: knee angle <= "
            f"{args.deep_angle:.0f} deg or "
            f"depth ratio >= {args.depth_threshold:.2f}"
        ),
        (
            f"↘  Torso lean: maximum <= "
            f"{args.torso_threshold:.0f} deg"
        ),
        (
            f"⌒  Upper back: maximum drop <= "
            f"{args.rounding_threshold:.0f} deg"
        ),
        (
            f"↔  Foot balance: "
            f"{args.heel_threshold:.2f} <= ratio <= "
            f"{args.toe_threshold:.2f}"
        ),
    ]

    criteria_y = [
        0.40,
        0.31,
        0.22,
        0.13,
    ]

    for text, y in zip(
        criteria_lines,
        criteria_y,
    ):
        ax_legend.text(
            0.08,
            y,
            text,
            fontsize=7.3,
            va="center",
            transform=ax_legend.transAxes,
        )

    # =====================================
    # 4. 右下：Repごとの最小膝角度
    # =====================================
    ax_bar = fig.add_subplot(
        gs[2, 5:]
    )

    repetitions = (
        summary["repetition"]
        .astype(int)
        .to_numpy()
    )

    minimum_angles = (
        summary["min_knee_angle_deg"]
        .to_numpy(dtype=float)
    )

    bar_colors = [
        (
            OK_COLOR
            if bool(value)
            else NG_COLOR
        )
        for value
        in summary["overall_ok"]
    ]

    bars = ax_bar.bar(
        repetitions,
        minimum_angles,
        color=bar_colors,
        width=0.68,
        zorder=3,
    )

    # 深さ基準
    ax_bar.axhline(
        args.deep_angle,
        color="#666666",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
    )

    # 各棒の上にOK/NG記号
    for bar, (_, row) in zip(
        bars,
        summary.iterrows(),
    ):
        is_ok = bool(
            row["overall_ok"]
        )

        icon = (
            "✓"
            if is_ok
            else "!"
        )

        color = (
            OK_COLOR
            if is_ok
            else NG_COLOR
        )

        ax_bar.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 2.5,
            icon,
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
            color=color,
            zorder=4,
        )

    # OK / NG 凡例
    legend_handles = [
        Patch(
            facecolor=OK_COLOR,
            edgecolor="none",
            label="OK",
        ),
        Patch(
            facecolor=NG_COLOR,
            edgecolor="none",
            label="NG",
        ),
    ]

    ax_bar.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8.5,
        frameon=True,
    )

    ax_bar.set_title(
        "Minimum Knee Angle per Repetition",
        fontsize=11,
        pad=10,
    )

    ax_bar.set_xlabel(
        "Repetition",
        fontsize=9,
        labelpad=8,
    )

    ax_bar.set_ylabel(
        "Angle [deg]",
        fontsize=9,
        labelpad=10,
    )

    ax_bar.set_xticks(
        repetitions
    )

    ax_bar.tick_params(
        axis="x",
        labelsize=8,
        pad=4,
    )

    ax_bar.tick_params(
        axis="y",
        labelsize=8,
        pad=3,
    )

    # 下側と上側に少し余白
    ax_bar.set_ylim(
        0,
        max(
            args.deep_angle + 20,
            float(
                np.nanmax(
                    minimum_angles
                )
            ) + 15,
        ),
    )

    ax_bar.grid(
        True,
        axis="y",
        alpha=0.24,
        zorder=0,
    )

    ax_bar.spines[
        "top"
    ].set_visible(False)

    ax_bar.spines[
        "right"
    ].set_visible(False)

    # x軸ラベルの重なり防止
    ax_bar.margins(
        x=0.03
    )

    # =====================================
    # 5. 注釈
    # =====================================
    ax_footer = fig.add_subplot(
        gs[3, :]
    )

    ax_footer.axis("off")

    ax_footer.text(
        0.5,
        0.5,
        (
            "* Each repetition is evaluated "
            "using the full motion and its "
            "deepest point."
        ),
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED_COLOR,
        transform=ax_footer.transAxes,
    )

    fig.savefig(
        args.report_pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.18,
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    require_columns(df, ["frame", "time_sec", "phase", "knee_angle_deg", "torso_lean_deg", "squat_depth_ratio"])
    ranges = find_repetition_ranges(df)
    if not ranges:
        raise RuntimeError("No complete repetitions were detected.")
    summary = summarize_repetitions(df, ranges, args)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False, encoding="utf-8")
    create_timeline_plot(df, summary, args.timeline, args.deep_angle)
    create_repetition_plot(summary, args.rep_plot, args.deep_angle)
    create_pdf_report(df, summary, args)
    print("Finished.")
    print(f"Timeline plot:      {args.timeline}")
    print(f"Repetition plot:    {args.rep_plot}")
    print(f"Repetition summary: {args.summary}")
    print(f"PDF report:         {args.report_pdf}")


if __name__ == "__main__":
    main()
