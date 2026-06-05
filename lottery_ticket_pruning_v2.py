# Inspiracija/Inspiration - Nadogradnja/Upgrade
# https://github.com/whoisadi19/lottery-ticket-pruning/tree/main


"""
Hipoteza srećnog tiketa — Proređivanje neuronske mreže
— ideja da unutar velike mreže već postoji mali podsklop („srećni tiket") 
— izbacivanje nepotrebnih težina.
"""


"""
Lottery Ticket Hypothesis - Loto 7/39 (v2)

  - ulaz  = zadnjih LAG izvlacenja, svako kao 39-dim multi-hot -> LAG*39 feature-a
  - izlaz = sledece izvlacenje kao 39-dim multi-hot (7 jedinica)
  - mreza 195->300->100->39, BCEWithLogitsLoss
  - metrika: pogodaka@7 (top-7 predvidjenih vs stvarnih 7), prosek po test skupu

Loto 7/39 tiket:
  - treniraj baseline, sacuvaj POCETNE tezine
  - magnitude pruning na razlicitim procentima
  - reset na pocetnu inicijalizaciju + maska (winning ticket)
  - retrain sa maskom, meri pogotke

Determinizam:
  - SEED = 39 svuda (random / numpy / torch), bez shuffle-a, CPU
  - vremenski podeljen train/test (bez mesanja)

Ulaz:
  /data/loto7_4626_k44.csv
Izlaz:
  experiment_results_v2.json
  lottery_ticket_pruning_v2.txt
"""


import os
import json
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ─── Determinizam: seed=39, bez random ───────────────────────────────
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/data/loto7_4626_k44.csv"
RESULTS_JSON = os.path.join(HERE, "experiment_results_v2.json")
TXT_OUT = os.path.join(HERE, "lottery_ticket_pruning_v2.txt")

N_NUMBERS = 39
K_PICK = 7
LAG = 5
INPUT_DIM = LAG * N_NUMBERS
HIDDEN1 = 300
HIDDEN2 = 100
OUTPUT_DIM = N_NUMBERS

EPOCHS = 40
LR = 0.001
BATCH_SIZE = 64
TEST_SIZE = 462  # ~10% poslednjih izvlacenja kao test (vremenski)
PRUNING_PERCENTAGES = [0, 20, 40, 60, 70, 80, 90, 95]
DEVICE = torch.device("cpu")  # CPU radi determinizma


class LotoNN(nn.Module):
    """
    Multi-label mreza za Loto 7/39.
    Arhitektura: INPUT_DIM -> 300 -> 100 -> 39 (logiti, sigmoid preko loss-a).
    """
    def __init__(self):
        super(LotoNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_loto_csv(path):
    """Citanje izvlacenja: svaki red bar K_PICK brojeva 1..39."""
    import csv
    draws = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vals = []
            for cell in row:
                try:
                    vals.append(int(str(cell).strip()))
                except ValueError:
                    continue
            combo = tuple(sorted(vals[:K_PICK]))
            if len(combo) == K_PICK and len(set(combo)) == K_PICK and all(1 <= x <= N_NUMBERS for x in combo):
                draws.append(combo)
    if len(draws) < LAG + TEST_SIZE + 10:
        raise ValueError("Premalo izvlacenja u CSV-u.")
    return draws


def draws_to_multihot(draws):
    """Lista kombinacija -> matrica (N, 39) multi-hot."""
    mat = np.zeros((len(draws), N_NUMBERS), dtype=np.float32)
    for i, combo in enumerate(draws):
        for num in combo:
            mat[i, num - 1] = 1.0
    return mat


def build_xy(multihot, lag=LAG):
    """X[i] = konkatenacija zadnjih `lag` redova; Y[i] = tekuci red."""
    x_rows, y_rows = [], []
    for i in range(lag, len(multihot)):
        x_rows.append(multihot[i - lag:i].reshape(-1))
        y_rows.append(multihot[i])
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return x, y


def hits_at_k(logits, targets, k=K_PICK):
    """Prosek pogodaka: top-k predvidjenih brojeva vs stvarnih jedinica."""
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k, dim=1).indices
    total_hits = 0
    for i in range(targets.size(0)):
        true_idx = set(torch.nonzero(targets[i], as_tuple=False).view(-1).tolist())
        pred_idx = set(topk[i].tolist())
        total_hits += len(true_idx & pred_idx)
    return total_hits / max(1, targets.size(0))


def evaluate_model(model, x_test, y_test):
    """Prosek pogodaka@7 na test skupu."""
    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(DEVICE))
        return hits_at_k(logits.cpu(), y_test.cpu())


def apply_mask(model, mask):
    """Primena pruning maske na tezine modela."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name])


def create_pruning_mask(model, pruning_percentage):
    """Magnitude-based maska: izbacuju se tezine najmanje apsolutne vrednosti."""
    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_abs = torch.abs(param.data)
            threshold_index = int(pruning_percentage / 100.0 * weight_abs.numel())
            sorted_weights = torch.sort(weight_abs.view(-1))[0]
            if threshold_index < len(sorted_weights):
                threshold = sorted_weights[threshold_index]
                masks[name] = (weight_abs > threshold).float()
            else:
                masks[name] = torch.zeros_like(param.data)
        else:
            masks[name] = torch.ones_like(param.data)
    return masks


def train_model(model, train_loader, x_test, y_test, mask=None, epochs=EPOCHS, verbose=True):
    """Trening (opciono sa odrzavanjem pruning maske posle svakog koraka)."""
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if mask is not None:
                apply_mask(model, mask)
            train_loss += loss.item()
        if verbose:
            hits = evaluate_model(model, x_test, y_test)
            print(f"  Epoch {epoch + 1:02d}/{epochs}  loss={train_loss / len(train_loader):.4f}  hits@7={hits:.4f}")

    return evaluate_model(model, x_test, y_test)


def lottery_ticket_experiment(x_train, y_train, x_test, y_test):
    """Glavni lottery-ticket eksperiment nad Loto podacima."""
    print("=" * 60)
    print("LOTTERY TICKET HYPOTHESIS - LOTO 7/39 (v2)")
    print("=" * 60)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    print("\n[Korak 1] Trening baseline modela...")
    baseline_model = LotoNN()
    initial_weights = copy.deepcopy(baseline_model.state_dict())
    baseline_hits = train_model(baseline_model, train_loader, x_test_t, y_test_t)

    total_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"\nBaseline pogodaka@7: {baseline_hits:.4f}  ({100 * baseline_hits / K_PICK:.2f}%)")
    print(f"Ukupno parametara:  {total_params}")

    results = {
        "baseline_hits": baseline_hits,
        "random_reference_hits": K_PICK * K_PICK / N_NUMBERS,
        "pruning_results": [],
    }

    print("\n" + "=" * 60)
    print("PRUNING EKSPERIMENTI")
    print("=" * 60)

    for prune_pct in PRUNING_PERCENTAGES:
        print(f"\n[Pruning {prune_pct}%]")
        pruned_model = LotoNN()
        pruned_model.load_state_dict(initial_weights)

        mask = create_pruning_mask(baseline_model, prune_pct)
        apply_mask(pruned_model, mask)

        remaining_params = sum(mask[name].sum().item() for name in mask if "weight" in name)
        actual_prune_pct = 100 * (1 - remaining_params / total_params)
        print(f"Preostalo parametara: {int(remaining_params)} ({100 - actual_prune_pct:.1f}% originala)")

        print("Retrain pruned mreze...")
        pruned_hits = train_model(pruned_model, train_loader, x_test_t, y_test_t, mask=mask, verbose=False)

        print(f"Pruned pogodaka@7: {pruned_hits:.4f}  ({100 * pruned_hits / K_PICK:.2f}%)")
        print(f"Razlika od baseline: {pruned_hits - baseline_hits:+.4f}")

        results["pruning_results"].append({
            "pruning_percentage": prune_pct,
            "actual_pruning_percentage": actual_prune_pct,
            "avg_hits": pruned_hits,
            "hits_difference": pruned_hits - baseline_hits,
            "remaining_parameters": int(remaining_params),
            "total_parameters": total_params,
        })

    return results, baseline_model


def predict_next(model, multihot, draws):
    """NEXT predikcija: top-7 brojeva za sledece kolo iz zadnjih LAG izvlacenja."""
    model.eval()
    x_next = multihot[-LAG:].reshape(1, -1)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_next.astype(np.float32)).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().view(-1).numpy()
    order = np.argsort(-probs)
    top7 = sorted(int(i) + 1 for i in order[:K_PICK])
    ranked = [(int(order[j]) + 1, float(probs[order[j]])) for j in range(N_NUMBERS)]
    return tuple(top7), ranked


def save_results(results):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nRezultati sacuvani -> {RESULTS_JSON}")


def main():
    print(f"Device: {DEVICE}  |  SEED={SEED}  |  LAG={LAG}")

    draws = load_loto_csv(CSV_PATH)
    multihot = draws_to_multihot(draws)
    x, y = build_xy(multihot, LAG)

    # Vremenski split (bez mesanja): poslednjih TEST_SIZE kao test.
    split = len(x) - TEST_SIZE
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    print(f"Izvlacenja: {len(draws)}  |  uzoraka: {len(x)}  |  train: {len(x_train)}  test: {len(x_test)}")

    results, baseline_model = lottery_ticket_experiment(x_train, y_train, x_test, y_test)
    save_results(results)

    next_combo, ranked = predict_next(baseline_model, multihot, draws)

    # ─── Ispis + TXT ─────────────────────────────────────────────────
    lines = []
    lines.append("Lottery Ticket Hypothesis - Loto 7/39 (v2)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  CSV:                  {CSV_PATH}")
    lines.append(f"  Izvlacenja:           {len(draws)}")
    lines.append(f"  LAG (ulaznih kola):   {LAG}")
    lines.append(f"  Ulaz/izlaz mreze:     {INPUT_DIM} -> {HIDDEN1} -> {HIDDEN2} -> {OUTPUT_DIM}")
    lines.append(f"  Epoha:                {EPOCHS}   LR: {LR}   batch: {BATCH_SIZE}")
    lines.append(f"  Train/Test:           {len(x_train)} / {len(x_test)} (vremenski)")
    lines.append(f"  SEED:                 {SEED} (bez random, bez shuffle, CPU)")
    lines.append("")
    lines.append(f"  Baseline pogodaka@7:  {results['baseline_hits']:.4f}  ({100 * results['baseline_hits'] / K_PICK:.2f}%)")
    lines.append(f"  Slucajna referenca:   {results['random_reference_hits']:.4f}  (7*7/39)")
    lines.append("")
    lines.append("PRUNING REZULTATI")
    lines.append("=" * 60)
    lines.append(f"  {'Pruning %':<11}{'pogodaka@7':>12}{'razlika':>11}{'param':>12}{'% orig':>9}")
    for r in results["pruning_results"]:
        pct_orig = 100 * r["remaining_parameters"] / r["total_parameters"]
        lines.append(
            f"  {str(r['pruning_percentage']) + '%':<11}"
            f"{r['avg_hits']:>12.4f}{r['hits_difference']:>+11.4f}"
            f"{r['remaining_parameters']:>12,}{pct_orig:>8.1f}%"
        )
    lines.append("")
    lines.append("PREDIKCIJA: NEXT / lottery_ticket_v2 (baseline model)")
    lines.append("=" * 60)
    lines.append(f"  Top-7 brojeva:        {next_combo}")
    lines.append("")
    lines.append("  Rang svih 39 brojeva po verovatnoci (broj: p):")
    for j in range(0, N_NUMBERS, 5):
        chunk = ranked[j:j + 5]
        lines.append("    " + "   ".join(f"{num:>2}:{p:.3f}" for num, p in chunk))
    lines.append("")
    lines.append(f"  JSON:                 {RESULTS_JSON}")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"TXT sacuvan -> {TXT_OUT}")

    print("\n" + "=" * 60)
    print("Zakljucak: lottery-ticket pruning nad Loto 7/39 - pogodaka@7 po pruning nivou.")
    print("=" * 60)


if __name__ == "__main__":
    main()



"""
Device: cpu  |  SEED=39  |  LAG=5
Izvlacenja: 4626  |  uzoraka: 4621  |  train: 4159  test: 462
============================================================
LOTTERY TICKET HYPOTHESIS - LOTO 7/39 (v2)
============================================================

[Korak 1] Trening baseline modela...
  Epoch 01/40  loss=0.5077  hits@7=1.2771
  Epoch 02/40  loss=0.4706  hits@7=1.2403
  Epoch 03/40  loss=0.4693  hits@7=1.2965
  Epoch 04/40  loss=0.4673  hits@7=1.2641
  Epoch 05/40  loss=0.4644  hits@7=1.2879
  Epoch 06/40  loss=0.4608  hits@7=1.2554
  Epoch 07/40  loss=0.4565  hits@7=1.2771
  Epoch 08/40  loss=0.4518  hits@7=1.2511
  Epoch 09/40  loss=0.4464  hits@7=1.2532
  Epoch 10/40  loss=0.4405  hits@7=1.2316
  Epoch 11/40  loss=0.4342  hits@7=1.1753
  Epoch 12/40  loss=0.4273  hits@7=1.2056
  Epoch 13/40  loss=0.4202  hits@7=1.2229
  Epoch 14/40  loss=0.4127  hits@7=1.1753
  Epoch 15/40  loss=0.4052  hits@7=1.1580
  Epoch 16/40  loss=0.3976  hits@7=1.1364
  Epoch 17/40  loss=0.3902  hits@7=1.1515
  Epoch 18/40  loss=0.3827  hits@7=1.1905
  Epoch 19/40  loss=0.3754  hits@7=1.1926
  Epoch 20/40  loss=0.3681  hits@7=1.1861
  Epoch 21/40  loss=0.3610  hits@7=1.1840
  Epoch 22/40  loss=0.3540  hits@7=1.2251
  Epoch 23/40  loss=0.3471  hits@7=1.2251
  Epoch 24/40  loss=0.3402  hits@7=1.2294
  Epoch 25/40  loss=0.3335  hits@7=1.2251
  Epoch 26/40  loss=0.3267  hits@7=1.2294
  Epoch 27/40  loss=0.3202  hits@7=1.2251
  Epoch 28/40  loss=0.3138  hits@7=1.2424
  Epoch 29/40  loss=0.3076  hits@7=1.2208
  Epoch 30/40  loss=0.3013  hits@7=1.2208
  Epoch 31/40  loss=0.2954  hits@7=1.2273
  Epoch 32/40  loss=0.2898  hits@7=1.2359
  Epoch 33/40  loss=0.2842  hits@7=1.2229
  Epoch 34/40  loss=0.2789  hits@7=1.2294
  Epoch 35/40  loss=0.2735  hits@7=1.2597
  Epoch 36/40  loss=0.2684  hits@7=1.2511
  Epoch 37/40  loss=0.2637  hits@7=1.2316
  Epoch 38/40  loss=0.2591  hits@7=1.2294
  Epoch 39/40  loss=0.2547  hits@7=1.2619
  Epoch 40/40  loss=0.2507  hits@7=1.2359

Baseline pogodaka@7: 1.2359  (17.66%)
Ukupno parametara:  92839

============================================================
PRUNING EKSPERIMENTI
============================================================

[Pruning 0%]
Preostalo parametara: 92397 (99.5% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2424  (17.75%)
Razlika od baseline: +0.0065

[Pruning 20%]
Preostalo parametara: 73917 (79.6% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2186  (17.41%)
Razlika od baseline: -0.0173

[Pruning 40%]
Preostalo parametara: 55437 (59.7% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.1926  (17.04%)
Razlika od baseline: -0.0433

[Pruning 60%]
Preostalo parametara: 36957 (39.8% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2208  (17.44%)
Razlika od baseline: -0.0152

[Pruning 70%]
Preostalo parametara: 27717 (29.9% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.1580  (16.54%)
Razlika od baseline: -0.0779

[Pruning 80%]
Preostalo parametara: 18477 (19.9% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2338  (17.63%)
Razlika od baseline: -0.0022

[Pruning 90%]
Preostalo parametara: 9237 (9.9% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2100  (17.29%)
Razlika od baseline: -0.0260

[Pruning 95%]
Preostalo parametara: 4617 (5.0% originala)
Retrain pruned mreze...
Pruned pogodaka@7: 1.2273  (17.53%)
Razlika od baseline: -0.0087

Rezultati sacuvani -> /Loto-7-39-Srbija-pruning/experiment_results_v2.json

Lottery Ticket Hypothesis - Loto 7/39 (v2)
============================================================

  CSV:                  /data/loto7_4626_k44.csv
  Izvlacenja:           4626
  LAG (ulaznih kola):   5
  Ulaz/izlaz mreze:     195 -> 300 -> 100 -> 39
  Epoha:                40   LR: 0.001   batch: 64
  Train/Test:           4159 / 462 (vremenski)
  SEED:                 39 (bez random, bez shuffle, CPU)

  Baseline pogodaka@7:  1.2359  (17.66%)
  Slucajna referenca:   1.2564  (7*7/39)

PRUNING REZULTATI
============================================================
  Pruning %    pogodaka@7    razlika       param   % orig
  0%               1.2424    +0.0065      92,397    99.5%
  20%              1.2186    -0.0173      73,917    79.6%
  40%              1.1926    -0.0433      55,437    59.7%
  60%              1.2208    -0.0152      36,957    39.8%
  70%              1.1580    -0.0779      27,717    29.9%
  80%              1.2338    -0.0022      18,477    19.9%
  90%              1.2100    -0.0260       9,237     9.9%
  95%              1.2273    -0.0087       4,617     5.0%

PREDIKCIJA: NEXT / lottery_ticket_v2 (baseline model)
============================================================
  Top-7 brojeva:        (3, x, 17, y, 20, z, 32)

  Rang svih 39 brojeva po verovatnoci (broj: p):
    32:0.951    y:0.896   20:0.617    z:0.452    3:0.433
    17:0.274    x:0.228    6:0.217    9:0.212   34:0.192
     2:0.156   13:0.117   30:0.099    1:0.097   19:0.096
    27:0.093   35:0.092   23:0.086   38:0.080   26:0.064
     5:0.055   10:0.052    4:0.034   25:0.029    7:0.025
     8:0.016   28:0.012   37:0.012   36:0.010   33:0.009
    15:0.009   14:0.007   21:0.007   39:0.005   24:0.004
    31:0.004   22:0.002   12:0.001   11:0.000

  JSON:                 /Loto-7-39-Srbija-pruning/experiment_results_v2.json

TXT sacuvan -> /Loto-7-39-Srbija-pruning/lottery_ticket_pruning_v2.txt

============================================================
Zakljucak: lottery-ticket pruning nad Loto 7/39 - pogodaka@7 po pruning nivou.
============================================================
"""





"""
Analiza lottery_ticket_pruning_v2.py:

Skripta je dobro prilagođena za Loto 7/39. 
multi-label zadatak: zadnjih 5 kola kao ulaz (195 feature-a), 
sledeće kolo kao 39 izlaza, pa se bira top-7 brojeva. 
Determinizam je ispoštovan: SEED=39, shuffle=False, CPU, vremenski split train/test.

Rezultat je ocekivan:
Baseline: 1.2359 pogodaka po kolu.
Slučajna referenca: 1.2564.
Znači baseline je malo ispod slučajne reference. 
Model nije našao korisnu prediktivnu zakonitost u ovom obliku.
Najbolji pruning rezultat je 0% sa 1.2424, 
ali i to je i dalje ispod slučajne reference.
80% pruning daje 1.2338, skoro isto kao baseline sa samo ~20% težina. 
To govori da je mreža prevelika za signal koji ima, 
tj. većina parametara ne doprinosi stvarno.
70% pruning je najlošiji (1.1580), ali nema lep monotoni obrazac — 
rezultati se ponašaju kao slabi/noisy signal.

Glavna NEXT predikcija iz baseline modela je:
(3, 16, 17, y, 20, 29, 32)
Najveće verovatnoće: 32=0.951, y=0.896, 20=0.617, zatim z, 3, 17, x.

Zaključak: tehnički skripta radi kako treba. 
Analitički rezultat je slab: 
pruning potvrđuje da mreža nema jak signal za predikciju, 
nego uglavnom uči stabilne obrasce/frekventnost u podacima.
"""





"""
lottery_ticket_pruning_v2.py — jezgro: 
Loto 7/39 multi-label mreža 195→300→100→39, 
lottery-ticket pruning (0–95%), metrika pogodaka@7, NEXT predikcija top-7. 
Determinizam seed=39, bez shuffle-a, CPU.

visualize_results_v2.py — 
grafici nad results/experiment_results_v2.json 
(pogodaka@7 vs pruning, redukcija parametara, razlika, kombinovano, tabela) → images/*_v2.png.

run_experiment_v2.py — wrapper sa potisnutim warning-ima.

Ključne odluke: 
ulaz = zadnjih LAG=5 kola kao multi-hot; 
cilj = sledeće kolo; test = poslednjih 462 izvlačenja vremenski; 
metrika pogodaka@7 sa slučajnom referencom 7·7/39≈1.26.



Glavni eksperiment:
python lottery_ticket_pruning_v2.py ili kraći wrapper:

python run_experiment_v2.py
Tek kad se napravi experiment_results_v2.json, pokreni grafike:
python visualize_results_v2.py
Glavni rezultat za analizu biće u:
lottery_ticket_pruning_v2.txt
experiment_results_v2.json
*_v2.png
"""
