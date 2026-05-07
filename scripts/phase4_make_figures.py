from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")


RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path.relative_to(ROOT)}")


def plot_penalty_feasibility() -> None:
    path = RESULTS / "phase1" / "penalty_sweep_summary.csv"
    df = pd.read_csv(path)

    labels = [f"{x:g}" for x in df["penalty_multiplier"]]
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(x, df["feasibility_rate"], marker="o", linewidth=2.2, color="#276FBF")
    ax1.set_ylabel("Feasibility rate", color="#276FBF")
    ax1.tick_params(axis="y", labelcolor="#276FBF")
    ax1.set_ylim(-0.04, 1.04)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["mean_cardinality_violation"],
        marker="s",
        linewidth=2.0,
        color="#D95D39",
    )
    ax2.set_ylabel("Mean cardinality violation", color="#D95D39")
    ax2.tick_params(axis="y", labelcolor="#D95D39")
    ax2.set_ylim(bottom=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Penalty multiplier")
    ax1.set_title("QUBO Penalty Scaling Controls Cardinality Feasibility")
    ax1.grid(True, axis="y", alpha=0.25)
    save_current_figure(FIGURES / "penalty_feasibility.png")


def plot_equity_curves(input_name: str, output_name: str, title: str) -> None:
    path = RESULTS / "phase3" / input_name
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    equity = (1.0 + returns).cumprod()

    plt.figure(figsize=(8.5, 4.8))
    colors = {
        "exact": "#276FBF",
        "local_search": "#2A9D8F",
        "top_k_mu": "#D95D39",
        "neal": "#6D597A",
    }
    for column in equity.columns:
        plt.plot(
            equity.index,
            equity[column],
            label=column,
            linewidth=2.0,
            color=colors.get(column),
        )

    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    save_current_figure(FIGURES / output_name)


def plot_risk_return_scatter(input_name: str, output_name: str, title: str) -> None:
    path = RESULTS / "phase3" / input_name
    summary = pd.read_csv(path)

    plt.figure(figsize=(7.8, 5.0))
    colors = {
        "exact": "#276FBF",
        "local_search": "#2A9D8F",
        "top_k_mu": "#D95D39",
        "neal": "#6D597A",
    }
    label_offsets = {
        "exact": (9, 12),
        "local_search": (9, -4),
        "top_k_mu": (-2, 9),
        "neal": (12, -2),
    }

    summary["coordinate_key"] = list(
        zip(
            summary["annual_volatility"].round(12),
            summary["annual_return"].round(12),
        )
    )
    for _, group in summary.groupby("coordinate_key", sort=False):
        row = group.iloc[0]
        strategies = list(group["strategy"])
        strategy_label = "/".join(strategies)
        if len(group) == 1:
            sharpe_label = f"S={row['sharpe']:.2f}"
            color = colors.get(strategies[0], "#444444")
            offset = label_offsets.get(strategies[0], (8, 5))
        else:
            sharpe_values = group["sharpe"].round(2).unique()
            sharpe_label = (
                f"S={sharpe_values[0]:.2f}"
                if len(sharpe_values) == 1
                else f"S={group['sharpe'].min():.2f}-{group['sharpe'].max():.2f}"
            )
            color = "#444444"
            offset = (9, 10)

        plt.scatter(
            row["annual_volatility"],
            row["annual_return"],
            s=95,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        plt.annotate(
            f"{strategy_label}\n{sharpe_label}",
            (row["annual_volatility"], row["annual_return"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )

    plt.title(title)
    plt.xlabel("Annual volatility")
    plt.ylabel("Annual return")
    plt.grid(True, alpha=0.25)
    save_current_figure(FIGURES / output_name)


def plot_qk_sharpe_heatmap() -> None:
    path = RESULTS / "phase3" / "historical_q_k_sweep.csv"
    df = pd.read_csv(path)
    local = df[df["strategy"] == "local_search"].copy()
    pivot = local.pivot(index="cardinality", columns="risk_aversion", values="sharpe")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Risk aversion q")
    ax.set_ylabel("Cardinality K")
    ax.set_title("Historical-Mean Local Search Sharpe Across q/K")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white")

    fig.colorbar(image, ax=ax, label="Sharpe")
    save_current_figure(FIGURES / "historical_qk_sharpe_heatmap_local_search.png")


def plot_qk_sharpe_delta() -> None:
    path = RESULTS / "phase3" / "historical_q_k_sweep.csv"
    df = pd.read_csv(path)
    local = df[df["strategy"] == "local_search"].copy()
    topk = df[df["strategy"] == "top_k_mu"].copy()
    merged = local.merge(
        topk,
        on=["cardinality", "risk_aversion"],
        suffixes=("_local", "_topk"),
    )
    merged["sharpe_delta"] = merged["sharpe_local"] - merged["sharpe_topk"]
    pivot = merged.pivot(index="cardinality", columns="risk_aversion", values="sharpe_delta")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    limit = float(np.nanmax(np.abs(pivot.values)))
    image = ax.imshow(pivot.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Risk aversion q")
    ax.set_ylabel("Cardinality K")
    ax.set_title("Sharpe Difference: Local Search Minus Top-K")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Sharpe difference")
    save_current_figure(FIGURES / "historical_qk_sharpe_delta.png")


def plot_selection_frequency() -> None:
    path = RESULTS / "phase3" / "historical_selections.csv"
    selections = pd.read_csv(path)
    local = selections[selections["strategy"] == "local_search"]

    counts: Counter[str] = Counter()
    for selected in local["selected"].dropna():
        counts.update(ticker.strip() for ticker in selected.split(",") if ticker.strip())

    labels, values = zip(*counts.most_common())
    y = np.arange(len(labels))

    plt.figure(figsize=(8.5, 6.2))
    plt.barh(y, values, color="#2A9D8F")
    plt.yticks(y, labels)
    plt.xlabel("Number of monthly selections")
    plt.title("Historical-Mean Local Search Asset Selection Frequency")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.25)
    save_current_figure(FIGURES / "historical_selection_frequency_local_search.png")


def plot_solver_runtime_scaling() -> None:
    path = RESULTS / "phase5" / "solver_scaling_summary.csv"
    if not path.exists():
        return

    summary = pd.read_csv(path)
    plt.figure(figsize=(8.5, 5.0))
    for solver, group in summary.groupby("solver"):
        plt.plot(
            group["n_assets"],
            group["mean_runtime_seconds"],
            marker="o",
            linewidth=2.0,
            label=solver,
        )

    plt.yscale("log")
    plt.title("Solver Runtime Scaling On Synthetic QUBOs")
    plt.xlabel("Number of assets")
    plt.ylabel("Mean runtime, seconds (log scale)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    save_current_figure(FIGURES / "solver_runtime_scaling.png")


def plot_solver_gap_scaling() -> None:
    path = RESULTS / "phase5" / "solver_scaling_summary.csv"
    if not path.exists():
        return

    summary = pd.read_csv(path)
    summary = summary[
        ~summary["solver"].isin(["exact_constrained", "exact_qubo"])
    ].copy()

    plt.figure(figsize=(8.5, 5.0))
    for solver, group in summary.groupby("solver"):
        plt.plot(
            group["n_assets"],
            group["mean_gap_vs_exact"],
            marker="o",
            linewidth=2.0,
            label=solver,
        )

    plt.title("Solver Objective Gap Scaling On Synthetic QUBOs")
    plt.xlabel("Number of assets")
    plt.ylabel("Mean objective gap vs exact constrained optimum")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    save_current_figure(FIGURES / "solver_gap_scaling.png")


def main() -> None:
    plot_penalty_feasibility()
    plot_equity_curves(
        "historical_daily_returns.csv",
        "historical_equity_curves.png",
        "Historical-Mean Backtest Equity Curves",
    )
    plot_equity_curves(
        "lasso_daily_returns.csv",
        "lasso_equity_curves.png",
        "LASSO Forecast Backtest Equity Curves",
    )
    if (RESULTS / "phase3" / "gradient_boosting_daily_returns.csv").exists():
        plot_equity_curves(
            "gradient_boosting_daily_returns.csv",
            "gradient_boosting_equity_curves.png",
            "Gradient Boosting Forecast Backtest Equity Curves",
        )
    plot_risk_return_scatter(
        "historical_summary.csv",
        "historical_risk_return_scatter.png",
        "Historical-Mean Risk-Return Tradeoff",
    )
    plot_risk_return_scatter(
        "lasso_summary.csv",
        "lasso_risk_return_scatter.png",
        "LASSO Forecast Risk-Return Tradeoff",
    )
    if (RESULTS / "phase3" / "gradient_boosting_summary.csv").exists():
        plot_risk_return_scatter(
            "gradient_boosting_summary.csv",
            "gradient_boosting_risk_return_scatter.png",
            "Gradient Boosting Forecast Risk-Return Tradeoff",
        )
    plot_qk_sharpe_heatmap()
    plot_qk_sharpe_delta()
    plot_selection_frequency()
    plot_solver_runtime_scaling()
    plot_solver_gap_scaling()


if __name__ == "__main__":
    main()
