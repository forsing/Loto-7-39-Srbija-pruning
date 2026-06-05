"""
Vizualizacija rezultata - Loto 7/39 lottery ticket (v3, bez frekvencije).

Cita results/experiment_results_v3.json i pravi slike *_v3.png.
"""

import json
import os

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    _PALETTE = sns.color_palette("viridis", 8)
except Exception:
    _PALETTE = None

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(HERE, "experiment_results_v3.json")
IMG_DIR = HERE

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def load_results(filename=RESULTS_JSON):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_val_test_vs_pruning(results, save_path):
    pruning = [r["pruning_percentage"] for r in results["pruning_results"]]
    val_hits = [r["val_hits"] for r in results["pruning_results"]]
    test_hits = [r["test_hits"] for r in results["pruning_results"]]
    baseline_test = results["baseline_test_hits"]
    rand_ref = results["random_reference_hits"]

    plt.figure(figsize=(12, 7))
    plt.plot(pruning, val_hits, "o-", linewidth=2.2, label="val pogodaka@7", color="#0f766e")
    plt.plot(pruning, test_hits, "s-", linewidth=2.2, label="test pogodaka@7", color="#2563eb")
    plt.axhline(baseline_test, color="#7c2d12", linestyle="--", linewidth=1.8,
                label=f"baseline test ({baseline_test:.3f})")
    plt.axhline(rand_ref, color="#f59e0b", linestyle=":", linewidth=1.8,
                label=f"slucajna ref. ({rand_ref:.3f})")
    plt.xlabel("Pruning procenat (%)", fontweight="bold")
    plt.ylabel("pogodaka@7", fontweight="bold")
    plt.title("v3 bez frekvencije: val/test pogodaka@7 vs pruning", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_parameter_reduction(results, save_path):
    pruning = [r["pruning_percentage"] for r in results["pruning_results"]]
    remaining = [r["remaining_weight_parameters"] for r in results["pruning_results"]]
    total = results["pruning_results"][0]["total_weight_parameters"]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(pruning)), remaining, color=_PALETTE if _PALETTE is not None else None)
    for bar, params in zip(bars, remaining):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{params:,}\n({100 * params / total:.1f}%)",
                 ha="center", va="bottom", fontsize=9)
    plt.xticks(range(len(pruning)), [f"{p}%" for p in pruning])
    plt.xlabel("Pruning procenat (%)", fontweight="bold")
    plt.ylabel("Preostali weight parametri", fontweight="bold")
    plt.title("v3: redukcija weight parametara", fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_baseline_history(results, save_path):
    hist = results["baseline_history"]
    epochs = [h["epoch"] for h in hist]
    val_hits = [h["val_hits"] for h in hist]
    test_hits = [h["test_hits"] for h in hist]
    best_epoch = results["baseline_best_epoch"]

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, val_hits, "-", linewidth=2.0, label="val pogodaka@7", color="#0f766e")
    plt.plot(epochs, test_hits, "-", linewidth=2.0, label="test pogodaka@7", color="#2563eb")
    plt.axvline(best_epoch, color="#dc2626", linestyle="--", linewidth=1.8,
                label=f"best_epoch={best_epoch}")
    plt.xlabel("Epoha", fontweight="bold")
    plt.ylabel("pogodaka@7", fontweight="bold")
    plt.title("v3 baseline: best-epoch izbor protiv overfit-a", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(results, save_path):
    data = []
    data.append([
        "baseline",
        f"{results['baseline_val_hits']:.4f}",
        f"{results['baseline_test_hits']:.4f}",
        "-",
        str(results["baseline_best_epoch"]),
        "100.0%",
    ])
    for r in results["pruning_results"]:
        pct_orig = 100 * r["remaining_weight_parameters"] / r["total_weight_parameters"]
        data.append([
            f"{r['pruning_percentage']}%",
            f"{r['val_hits']:.4f}",
            f"{r['test_hits']:.4f}",
            f"{r['test_hits_difference']:+.4f}",
            str(r["best_epoch"]),
            f"{pct_orig:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=["model", "val@7", "test@7", "diff", "epoch", "% weight-a"],
        cellLoc="center",
        loc="center",
        colWidths=[0.18, 0.15, 0.15, 0.15, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for i in range(6):
        table[(0, i)].set_facecolor("#134e4a")
        table[(0, i)].set_text_props(weight="bold", color="white")
    for i in range(1, len(data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f3f4f6")
    plt.title("v3 bez frekvencije - rezime", fontweight="bold", pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERISANJE VIZUALIZACIJA (v3, bez frekvencije)")
    print("=" * 60)
    os.makedirs(IMG_DIR, exist_ok=True)
    results = load_results()
    plot_val_test_vs_pruning(results, os.path.join(IMG_DIR, "val_test_vs_pruning_v3.png"))
    plot_parameter_reduction(results, os.path.join(IMG_DIR, "parameter_reduction_v3.png"))
    plot_baseline_history(results, os.path.join(IMG_DIR, "baseline_history_v3.png"))
    create_summary_table(results, os.path.join(IMG_DIR, "results_table_v3.png"))
    print("\nSve v3 vizualizacije generisane.")


if __name__ == "__main__":
    main()




"""
============================================================
GENERISANJE VIZUALIZACIJA (v3, bez frekvencije)
============================================================
Saved: /Loto-7-39-Srbija-pruning/val_test_vs_pruning_v3.png
Saved: /Loto-7-39-Srbija-pruning/parameter_reduction_v3.png
Saved: /Loto-7-39-Srbija-pruning/baseline_history_v3.png
Saved: /Loto-7-39-Srbija-pruning/results_table_v3.png

Sve v3 vizualizacije generisane.
"""



"""
Analiza visualize_results_v3.py
visualize_results_v3.py je završni prikaz v3 rezultata. 
Ne trenira ništa, samo čita: experiment_results_v3.json i pravi slike *_v3.png

Glavne slike:
val_test_vs_pruning_v3.png
Najvažniji graf. 
Pokazuje da baseline ne prelazi slučajnu referencu, 
ali pruning modeli 20%, 40%, 60%, 90% prelaze. 
Vizuelno se vidi da 20% ima najbolji test rezultat, 
dok 40% ima najbolju validaciju.

parameter_reduction_v3.png
Pokazuje koliko weight parametara ostaje. 
Posebno je važan 90% pruning: 
samo oko 10% težina, a i dalje jak test rezultat 1.2835.

baseline_history_v3.png
Ovo je najbolji graf za objašnjenje overfit-a. 
Loss pada, ali val/test@7 osciluju. 
Best epoch nije poslednja epoha nego validacioni pik. 
Time se opravdava zašto v3 koristi best-epoch selection.

results_table_v3.png
Sažeta tabela: 
baseline + svi pruning nivoi, val@7, test@7, razlika, epoha i procenat težina.

Zaključak iz vizualizacije: 
v3 jasno pokazuje da pruning stvarno pomaže, ali ne monotono. 
Najbolji test je 20%, najbolja validacija je 40%, a najzanimljiviji sparse ticket je 90%. 
Grafici lepo potvrđuju da v3 ima prediktivni dobitak bez frekvencijskih feature-a.
"""


"""
Zato što v2 i v3 ne crtaju isti set grafika.

v2 ima 5 slika:
hits_vs_pruning_v2.png
parameter_reduction_v2.png
hits_difference_v2.png
combined_metrics_v2.png
results_table_v2.png

v3 ima 4 slike:
val_test_vs_pruning_v3.png
parameter_reduction_v3.png
baseline_history_v3.png
results_table_v3.png

Razlog: 
u v3 sam hits_difference i combined_metrics izbacio kao manje bitne, 
a dodao baseline_history_v3.png, 
jer v3 ima best-epoch validaciju i tu je najvažnije videti overfit kroz epohe.

v3 ima drugačiju analitiku: više fokus na val/test i best-epoch, manje na dekorativne poredbene grafike.
"""
