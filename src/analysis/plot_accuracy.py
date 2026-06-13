"""Curvas de accuracy por ronda: media +- desviacion tipica sobre semillas.

Genera, para cada combinacion (dataset, particion, ataque, % maliciosos),
una figura comparando los agregadores, con la linea base limpia (FedAvg sin
ataque) como referencia discontinua.

Uso:  python -m src.analysis.plot_accuracy [--results-dir results/raw]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.analysis.common import (AGGREGATOR_COLORS, AGGREGATOR_LABELS,
                                 ATTACK_LABELS, load_rounds, save_figure)


def plot_metric_curves(rounds, metric: str, ylabel: str, output_dir: str,
                       prefix: str) -> int:
    """Una figura por (dataset, particion, ataque, fraccion) con una curva
    por agregador. Devuelve el numero de figuras generadas."""
    count = 0
    scenarios = rounds[rounds["attack"] != "none"][
        ["dataset", "partition", "attack", "malicious_pct"]].drop_duplicates()
    for _, scenario in scenarios.iterrows():
        subset = rounds[
            (rounds["dataset"] == scenario["dataset"])
            & (rounds["partition"] == scenario["partition"])
            & (rounds["attack"] == scenario["attack"])
            & (rounds["malicious_pct"] == scenario["malicious_pct"])]
        baseline = rounds[
            (rounds["dataset"] == scenario["dataset"])
            & (rounds["partition"] == scenario["partition"])
            & (rounds["attack"] == "none")
            & (rounds["aggregator"] == "fedavg")]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for aggregator, group in subset.groupby("aggregator"):
            # Media y dispersion sobre semillas: las rondas se alinean por su
            # numero porque todas las ejecuciones de una configuracion comparten
            # el mismo horizonte temporal.
            stats = group.groupby("round")[metric].agg(["mean", "std"])
            color = AGGREGATOR_COLORS.get(aggregator, "gray")
            ax.plot(stats.index, stats["mean"], color=color,
                    label=AGGREGATOR_LABELS.get(aggregator, aggregator))
            if stats["std"].notna().any():
                ax.fill_between(stats.index,
                                stats["mean"] - stats["std"],
                                stats["mean"] + stats["std"],
                                color=color, alpha=0.15)
        if not baseline.empty:
            # La linea limpia FedAvg fija el nivel de referencia visual de cada
            # particion, independientemente del agregador bajo ataque.
            base_stats = baseline.groupby("round")[metric].mean()
            ax.plot(base_stats.index, base_stats.values, "k--", linewidth=1,
                    label="FedAvg sin ataque", alpha=0.7)

        attack_label = ATTACK_LABELS.get(scenario["attack"], scenario["attack"])
        ax.set_xlabel("Ronda federada")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{attack_label} ({scenario['malicious_pct']}% maliciosos) "
                     f"- {scenario['dataset']} {scenario['partition']}")
        ax.legend(fontsize=8)
        name = (f"{prefix}_{scenario['dataset']}_{scenario['partition']}_"
                f"{scenario['attack']}_m{scenario['malicious_pct']}")
        save_figure(fig, output_dir, name)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="plots/accuracy")
    args = parser.parse_args()
    rounds = load_rounds(args.results_dir)
    if rounds.empty:
        print("No hay resultados que graficar")
        return
    n = plot_metric_curves(rounds, "test_accuracy", "Accuracy (test)",
                           args.output_dir, "accuracy")
    print(f"{n} figuras de accuracy en {args.output_dir}")


if __name__ == "__main__":
    main()
