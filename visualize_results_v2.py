"""
Vizualizacija rezultata - Loto 7/39 lottery ticket (v2)

Cita experiment_results_v2.json (metrika: pogodaka@7) i pravi grafike.
Determinizam nije bitan (samo crtanje), ali drzim se v2 imena fajlova.

Izlaz: *_v2.png
"""


import os
import json

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    _PALETTE = sns.color_palette("viridis", 8)
except Exception:
    _PALETTE = None

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(HERE, "experiment_results_v2.json")
IMG_DIR = HERE


def load_results(filename=RESULTS_JSON):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_hits_vs_pruning(results, save_path):
    pruning_pcts = [r["pruning_percentage"] for r in results["pruning_results"]]
    hits = [r["avg_hits"] for r in results["pruning_results"]]
    baseline = results["baseline_hits"]
    rand_ref = results.get("random_reference_hits", 7 * 7 / 39)

    plt.figure(figsize=(12, 7))
    plt.plot(pruning_pcts, hits, "o-", linewidth=2.5, markersize=8,
             color="#2E86AB", label="Pruned mreza - pogodaka@7")
    plt.axhline(y=baseline, color="#A23B72", linestyle="--", linewidth=2,
                label=f"Baseline ({baseline:.3f})")
    plt.axhline(y=rand_ref, color="#F18F01", linestyle=":", linewidth=1.5,
                label=f"Slucajna referenca ({rand_ref:.3f})", alpha=0.8)
    plt.xlabel("Pruning procenat (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Prosek pogodaka@7", fontsize=13, fontweight="bold")
    plt.title("Loto 7/39: pogodaka@7 vs pruning\n(Lottery Ticket Hypothesis, v2)",
              fontsize=15, fontweight="bold", pad=20)
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_parameter_reduction(results, save_path):
    pruning_pcts = [r["pruning_percentage"] for r in results["pruning_results"]]
    remaining = [r["remaining_parameters"] for r in results["pruning_results"]]
    total = results["pruning_results"][0]["total_parameters"]

    plt.figure(figsize=(12, 7))
    colors = _PALETTE if _PALETTE is not None else None
    bars = plt.bar(range(len(pruning_pcts)), remaining, color=colors)
    for bar, params in zip(bars, remaining):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h,
                 f"{params:,}\n({100 * params / total:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.xlabel("Pruning procenat (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Broj parametara", fontsize=13, fontweight="bold")
    plt.title("Smanjenje modela kroz pruning (v2)", fontsize=15, fontweight="bold", pad=20)
    plt.xticks(range(len(pruning_pcts)), [f"{p}%" for p in pruning_pcts])
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_hits_difference(results, save_path):
    pruning_pcts = [r["pruning_percentage"] for r in results["pruning_results"]]
    diffs = [r["hits_difference"] for r in results["pruning_results"]]

    plt.figure(figsize=(12, 7))
    colors = ["#06A77D" if d >= 0 else "#D62246" for d in diffs]
    bars = plt.bar(range(len(pruning_pcts)), diffs, color=colors, alpha=0.8)
    for bar, d in zip(bars, diffs):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{d:+.3f}",
                 ha="center", va="bottom" if d >= 0 else "top", fontsize=10)
    plt.axhline(y=0, color="black", linewidth=1)
    plt.xlabel("Pruning procenat (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Razlika pogodaka@7 od baseline", fontsize=13, fontweight="bold")
    plt.title("Uticaj pruning-a na pogotke (v2)", fontsize=15, fontweight="bold", pad=20)
    plt.xticks(range(len(pruning_pcts)), [f"{p}%" for p in pruning_pcts])
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_combined_metrics(results, save_path):
    pruning_pcts = [r["pruning_percentage"] for r in results["pruning_results"]]
    hits = [r["avg_hits"] for r in results["pruning_results"]]
    remaining = [r["remaining_parameters"] for r in results["pruning_results"]]
    total = results["pruning_results"][0]["total_parameters"]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    c1 = "#2E86AB"
    ax1.set_xlabel("Pruning procenat (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("pogodaka@7", color=c1, fontsize=13, fontweight="bold")
    l1 = ax1.plot(pruning_pcts, hits, "o-", color=c1, linewidth=2.5, markersize=8, label="pogodaka@7")
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    c2 = "#F18F01"
    ax2.set_ylabel("Velicina modela (% originala)", color=c2, fontsize=13, fontweight="bold")
    sizes = [100 * p / total for p in remaining]
    l2 = ax2.plot(pruning_pcts, sizes, "s--", color=c2, linewidth=2.5, markersize=8, label="Velicina modela")
    ax2.tick_params(axis="y", labelcolor=c2)

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=11)
    plt.title("pogodaka@7 vs velicina modela (v2)", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(results, save_path):
    data = []
    for r in results["pruning_results"]:
        data.append([
            f"{r['pruning_percentage']}%",
            f"{r['avg_hits']:.4f}",
            f"{r['hits_difference']:+.4f}",
            f"{r['remaining_parameters']:,}",
            f"{100 * r['remaining_parameters'] / r['total_parameters']:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=["Pruning %", "pogodaka@7", "Δ baseline", "Parametara", "% originala"],
        cellLoc="center", loc="center",
        colWidths=[0.15, 0.18, 0.18, 0.25, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    for i in range(5):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")
    for i in range(1, len(data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")
    plt.title("Rezime rezultata - Loto 7/39 (v2)", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERISANJE VIZUALIZACIJA (v2)")
    print("=" * 60)
    os.makedirs(IMG_DIR, exist_ok=True)
    results = load_results()
    plot_hits_vs_pruning(results, os.path.join(IMG_DIR, "hits_vs_pruning_v2.png"))
    plot_parameter_reduction(results, os.path.join(IMG_DIR, "parameter_reduction_v2.png"))
    plot_hits_difference(results, os.path.join(IMG_DIR, "hits_difference_v2.png"))
    plot_combined_metrics(results, os.path.join(IMG_DIR, "combined_metrics_v2.png"))
    create_summary_table(results, os.path.join(IMG_DIR, "results_table_v2.png"))
    print("\n" + "=" * 60)
    print("Sve vizualizacije generisane (sufiks _v2).")
    print("=" * 60)


if __name__ == "__main__":
    main()



"""
============================================================
GENERISANJE VIZUALIZACIJA (v2)
============================================================
Saved: /Loto-7-39-Srbija-pruning/hits_vs_pruning_v2.png
Saved: /Loto-7-39-Srbija-pruning/parameter_reduction_v2.png
Saved: /Loto-7-39-Srbija-pruning/hits_difference_v2.png
Saved: /Loto-7-39-Srbija-pruning/combined_metrics_v2.png
Saved: /Loto-7-39-Srbija-pruning/results_table_v2.png

============================================================
Sve vizualizacije generisane (sufiks _v2).
============================================================
"""



"""
Pravi 5 izlaza:

hits_vs_pruning_v2.png — najvažniji graf: pogodaka@7 kroz pruning nivoe, sa baseline linijom i slučajnom referencom.
parameter_reduction_v2.png — koliko parametara ostaje po pruning nivou.
hits_difference_v2.png — razlika svakog pruning modela u odnosu na baseline.
combined_metrics_v2.png — pogodaka@7 i veličina modela zajedno.
results_table_v2.png — tabela svih rezultata.
Analitički, vizualizacije potvrđuju isto što i TXT:

nema pruning nivoa koji jasno probija slučajnu referencu,
80% pruning skoro čuva baseline rezultat sa samo ~20% parametara,
70% pruning je najlošiji pad,
cela kriva je „ravna i šumna", 
što znači da lottery-ticket pruning ne otkriva jak prediktivni signal, 
nego mreža uglavnom nema šta stabilno da nauči.
Fajl je dobar kao završna prezentacija rezultata: 
grafici jasno pokazuju i model compression i činjenicu da prediktivni dobitak nije dobijen.
"""
