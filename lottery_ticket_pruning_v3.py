# Inspiracija/Inspiration - Nadogradnja/Upgrade
# https://github.com/whoisadi19/lottery-ticket-pruning/tree/main


"""
Hipoteza srećnog tiketa — Proređivanje neuronske mreže
— ideja da unutar velike mreže već postoji mali podsklop („srećni tiket") 
— izbacivanje nepotrebnih težina.
"""


"""
Lottery Ticket Hypothesis - Loto 7/39 (v3, bez frekvencije)

  - poboljsati v2 bez oslanjanja na frekvenciju pojavljivanja brojeva
  - seed=39, bez shuffle-a, CPU, vremenski split
  - best-epoch selection preko validacionog skupa da se izbegne overfit

Feature-i NISU frekvencijski:
  - zadnjih LAG kola kao 39-dim multi-hot sekvenca
  - transition vektor: poslednje kolo - pretposlednje kolo
  - gap vektor: koliko kola je proslo od poslednje pojave broja
  - lex sekvenca i dX prirastaji kroz zadnjih LAG kola

Model:
  - multi-label mreza: feature_dim -> 256 -> 128 -> 39
  - BCEWithLogitsLoss sa pos_weight=(39-7)/7
  - AdamW weight_decay protiv overfit-a
  - metrika: pogodaka@7

Izlaz:
  experiment_results_v3.json
  lottery_ticket_pruning_v3.txt
"""



import copy
import csv
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


SEED = 39
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7_4626_k44.csv"
RESULTS_JSON = os.path.join(HERE, "experiment_results_v3.json")
TXT_OUT = os.path.join(HERE, "lottery_ticket_pruning_v3.txt")

N_NUMBERS = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_NUMBERS, K_PICK)

LAG = 8
GAP_CAP = 80
VAL_SIZE = 462
TEST_SIZE = 462
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.0008
WEIGHT_DECAY = 1e-4
PRUNING_PERCENTAGES = [0, 20, 40, 60, 70, 80, 90, 95]
DEVICE = torch.device("cpu")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def load_loto_csv(path):
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
    min_rows = LAG + VAL_SIZE + TEST_SIZE + 10
    if len(draws) < min_rows:
        raise ValueError(f"Premalo izvlacenja: treba bar {min_rows}, ima {len(draws)}")
    return draws


def lex_rank_1based(combo):
    rank0 = 0
    prev = 0
    for i, value in enumerate(tuple(sorted(combo)), start=1):
        remaining = K_PICK - i
        for candidate in range(prev + 1, value):
            rank0 += math.comb(N_NUMBERS - candidate, remaining)
        prev = value
    return rank0 + 1


def draws_to_multihot(draws):
    mat = np.zeros((len(draws), N_NUMBERS), dtype=np.float32)
    for i, combo in enumerate(draws):
        for num in combo:
            mat[i, num - 1] = 1.0
    return mat


def gap_vector(multihot, end_idx):
    """Gap do poslednje pojave pre target indeksa end_idx; nije frekvencija."""
    gaps = np.full(N_NUMBERS, GAP_CAP, dtype=np.float32)
    for num_idx in range(N_NUMBERS):
        last_seen = None
        for j in range(end_idx - 1, -1, -1):
            if multihot[j, num_idx] > 0.5:
                last_seen = j
                break
        if last_seen is not None:
            gaps[num_idx] = min(GAP_CAP, end_idx - last_seen)
    return gaps / GAP_CAP


def build_features(draws, lag=LAG):
    multihot = draws_to_multihot(draws)
    lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float32)

    x_rows = []
    y_rows = []
    for i in range(lag, len(draws)):
        past = multihot[i - lag:i]
        seq_features = past.reshape(-1)

        transition = past[-1] - past[-2]
        gaps = gap_vector(multihot, i)

        lex_seq = (lex_idx[i - lag:i] / TOTAL_COMBOS).astype(np.float32)
        dx_seq_raw = np.diff(lex_idx[i - lag:i])
        dx_seq = ((dx_seq_raw + TOTAL_COMBOS) / (2 * TOTAL_COMBOS)).astype(np.float32)

        features = np.concatenate([seq_features, transition, gaps, lex_seq, dx_seq]).astype(np.float32)
        x_rows.append(features)
        y_rows.append(multihot[i])

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), multihot


class LotoNNv3(nn.Module):
    def __init__(self, input_dim):
        super(LotoNNv3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, N_NUMBERS)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def hits_at_k(logits, targets, k=K_PICK):
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k, dim=1).indices
    total_hits = 0
    for i in range(targets.size(0)):
        true_idx = set(torch.nonzero(targets[i], as_tuple=False).view(-1).tolist())
        pred_idx = set(topk[i].tolist())
        total_hits += len(true_idx & pred_idx)
    return total_hits / max(1, targets.size(0))


def evaluate_model(model, x_eval, y_eval, criterion=None):
    model.eval()
    with torch.no_grad():
        logits = model(x_eval.to(DEVICE))
        hits = hits_at_k(logits.cpu(), y_eval.cpu())
        loss = None
        if criterion is not None:
            loss = float(criterion(logits, y_eval.to(DEVICE)).item())
    return hits, loss


def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name])


def create_pruning_mask(model, pruning_percentage):
    masks = {}
    if pruning_percentage <= 0:
        for name, param in model.named_parameters():
            masks[name] = torch.ones_like(param.data)
        return masks

    for name, param in model.named_parameters():
        if "weight" in name:
            weight_abs = torch.abs(param.data)
            flat = torch.sort(weight_abs.view(-1))[0]
            cut = int(pruning_percentage / 100.0 * flat.numel())
            cut = min(max(cut, 1), flat.numel() - 1)
            threshold = flat[cut - 1]
            masks[name] = (weight_abs > threshold).float()
        else:
            masks[name] = torch.ones_like(param.data)
    return masks


def count_remaining_weight_params(mask):
    return int(sum(mask[name].sum().item() for name in mask if "weight" in name))


def make_loaders(x_train, y_train):
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def train_best_epoch(model, train_loader, x_val, y_val, x_test, y_test, mask=None, epochs=EPOCHS, verbose=True):
    pos_weight = torch.full((N_NUMBERS,), (N_NUMBERS - K_PICK) / K_PICK, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model = model.to(DEVICE)

    best_state = copy.deepcopy(model.state_dict())
    best_val_hits = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            if mask is not None:
                apply_mask(model, mask)
            train_loss += loss.item()

        val_hits, val_loss = evaluate_model(model, x_val, y_val, criterion)
        test_hits, _ = evaluate_model(model, x_test, y_test, criterion)
        avg_loss = train_loss / max(1, len(train_loader))
        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "val_hits": float(val_hits),
            "val_loss": float(val_loss),
            "test_hits": float(test_hits),
        })

        is_better = (val_hits > best_val_hits) or (val_hits == best_val_hits and val_loss < best_val_loss)
        if is_better:
            best_val_hits = float(val_hits)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"  Epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  "
                f"val@7={val_hits:.4f}  test@7={test_hits:.4f}"
            )

    model.load_state_dict(best_state)
    final_val_hits, final_val_loss = evaluate_model(model, x_val, y_val, criterion)
    final_test_hits, final_test_loss = evaluate_model(model, x_test, y_test, criterion)
    return {
        "model": model,
        "best_epoch": int(best_epoch),
        "val_hits": float(final_val_hits),
        "val_loss": float(final_val_loss),
        "test_hits": float(final_test_hits),
        "test_loss": float(final_test_loss),
        "history": history,
    }


def train_fixed_epochs(model, train_loader, epochs, mask=None):
    pos_weight = torch.full((N_NUMBERS,), (N_NUMBERS - K_PICK) / K_PICK, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model = model.to(DEVICE)

    for _ in range(max(1, epochs)):
        model.train()
        for data, target in train_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            if mask is not None:
                apply_mask(model, mask)
    return model


def lottery_ticket_experiment(x_train, y_train, x_val, y_val, x_test, y_test):
    print("=" * 60)
    print("LOTTERY TICKET - LOTO 7/39 v3 (BEZ FREKVENCIJE)")
    print("=" * 60)

    train_loader = make_loaders(x_train, y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    input_dim = x_train.shape[1]
    set_seed()
    baseline_model = LotoNNv3(input_dim)
    initial_weights = copy.deepcopy(baseline_model.state_dict())

    print("\n[Baseline] best-epoch trening...")
    baseline = train_best_epoch(
        baseline_model, train_loader, x_val_t, y_val_t, x_test_t, y_test_t, verbose=True
    )

    total_params = sum(p.numel() for p in baseline["model"].parameters())
    weight_params = sum(p.numel() for name, p in baseline["model"].named_parameters() if "weight" in name)
    print(
        f"\nBaseline: val@7={baseline['val_hits']:.4f}  test@7={baseline['test_hits']:.4f}  "
        f"best_epoch={baseline['best_epoch']}"
    )
    print(f"Parametri ukupno={total_params:,}  weight_parametri={weight_params:,}")

    results = {
        "seed": SEED,
        "lag": LAG,
        "feature_mode": "sequence+transition+gap+lex+dX (bez frekvencije)",
        "baseline_val_hits": baseline["val_hits"],
        "baseline_test_hits": baseline["test_hits"],
        "baseline_best_epoch": baseline["best_epoch"],
        "baseline_history": baseline["history"],
        "random_reference_hits": K_PICK * K_PICK / N_NUMBERS,
        "pruning_results": [],
    }

    print("\n" + "=" * 60)
    print("PRUNING EKSPERIMENTI")
    print("=" * 60)

    best_for_next = {
        "name": "baseline",
        "epoch": baseline["best_epoch"],
        "val_hits": baseline["val_hits"],
        "test_hits": baseline["test_hits"],
        "pruning_percentage": None,
    }

    for prune_pct in PRUNING_PERCENTAGES:
        print(f"\n[Pruning {prune_pct}%]")
        set_seed()
        pruned_model = LotoNNv3(input_dim)
        pruned_model.load_state_dict(initial_weights)

        mask = create_pruning_mask(baseline["model"], prune_pct)
        apply_mask(pruned_model, mask)
        remaining_weights = count_remaining_weight_params(mask)
        actual_prune_pct = 100 * (1 - remaining_weights / weight_params)
        print(f"Preostalo weight parametara: {remaining_weights:,} ({100 - actual_prune_pct:.1f}% weight-a)")

        pruned = train_best_epoch(
            pruned_model, train_loader, x_val_t, y_val_t, x_test_t, y_test_t, mask=mask, verbose=False
        )
        print(
            f"Pruned: val@7={pruned['val_hits']:.4f}  test@7={pruned['test_hits']:.4f}  "
            f"best_epoch={pruned['best_epoch']}"
        )

        result_row = {
            "pruning_percentage": prune_pct,
            "actual_pruning_percentage": actual_prune_pct,
            "remaining_weight_parameters": remaining_weights,
            "total_weight_parameters": weight_params,
            "val_hits": pruned["val_hits"],
            "test_hits": pruned["test_hits"],
            "test_hits_difference": pruned["test_hits"] - baseline["test_hits"],
            "best_epoch": pruned["best_epoch"],
        }
        results["pruning_results"].append(result_row)

        if pruned["val_hits"] > best_for_next["val_hits"]:
            best_for_next = {
                "name": f"pruned_{prune_pct}",
                "epoch": pruned["best_epoch"],
                "val_hits": pruned["val_hits"],
                "test_hits": pruned["test_hits"],
                "pruning_percentage": prune_pct,
            }

    results["best_for_next"] = best_for_next
    return results, baseline["model"], initial_weights


def predict_next(model, next_features):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(next_features.reshape(1, -1).astype(np.float32)).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().view(-1).numpy()
    order = np.argsort(-probs)
    top7 = tuple(sorted(int(i) + 1 for i in order[:K_PICK]))
    ranked = [(int(order[j]) + 1, float(probs[order[j]])) for j in range(N_NUMBERS)]
    return top7, ranked


def build_next_feature(draws):
    extended = list(draws) + [draws[-1]]
    x_all, _y_all, _mh = build_features(extended, LAG)
    return x_all[-1]


def train_final_model_for_next(draws, input_dim, best_epoch):
    x_all, y_all, _multihot = build_features(draws, LAG)
    loader = make_loaders(x_all, y_all)
    set_seed()
    model = LotoNNv3(input_dim)
    return train_fixed_epochs(model, loader, best_epoch)


def save_results(results):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nJSON sacuvan -> {RESULTS_JSON}")


def main():
    print(f"Device: {DEVICE}  |  SEED={SEED}  |  LAG={LAG}  |  bez frekvencije")

    draws = load_loto_csv(CSV_PATH)
    x, y, _multihot = build_features(draws, LAG)

    test_start = len(x) - TEST_SIZE
    val_start = test_start - VAL_SIZE
    x_train, y_train = x[:val_start], y[:val_start]
    x_val, y_val = x[val_start:test_start], y[val_start:test_start]
    x_test, y_test = x[test_start:], y[test_start:]

    print(
        f"Izvlacenja={len(draws)}  uzoraka={len(x)}  "
        f"train={len(x_train)}  val={len(x_val)}  test={len(x_test)}  feature_dim={x.shape[1]}"
    )

    results, _baseline_model, _initial_weights = lottery_ticket_experiment(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    input_dim = x.shape[1]
    final_epoch = int(results["best_for_next"]["epoch"])
    final_model = train_final_model_for_next(draws, input_dim, final_epoch)
    next_features = build_next_feature(draws)
    next_combo, ranked = predict_next(final_model, next_features)

    results["next_prediction"] = {
        "model_used": results["best_for_next"],
        "top7": list(next_combo),
        "ranked_probabilities": ranked,
    }
    save_results(results)

    lines = []
    lines.append("Lottery Ticket Hypothesis - Loto 7/39 (v3, bez frekvencije)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  CSV:                  {CSV_PATH}")
    lines.append(f"  Izvlacenja:           {len(draws)}")
    lines.append(f"  LAG:                  {LAG}")
    lines.append(f"  Feature dim:          {input_dim}")
    lines.append("  Feature-i:            multi-hot sekvenca + transition + gap + lex + dX")
    lines.append("  Frekvencija:          NE koristi se kao feature")
    lines.append(f"  Mreza:                {input_dim} -> 256 -> 128 -> 39")
    lines.append(f"  Epoha max:            {EPOCHS}  LR={LR}  weight_decay={WEIGHT_DECAY}")
    lines.append(f"  Train/Val/Test:       {len(x_train)} / {len(x_val)} / {len(x_test)}")
    lines.append(f"  SEED:                 {SEED} (bez shuffle, CPU)")
    lines.append("")
    lines.append("BASELINE")
    lines.append("-" * 70)
    lines.append(f"  best_epoch:           {results['baseline_best_epoch']}")
    lines.append(f"  val pogodaka@7:       {results['baseline_val_hits']:.4f}")
    lines.append(f"  test pogodaka@7:      {results['baseline_test_hits']:.4f}")
    lines.append(f"  slucajna referenca:   {results['random_reference_hits']:.4f} (7*7/39)")
    lines.append("")
    lines.append("PRUNING REZULTATI")
    lines.append("-" * 70)
    lines.append(f"  {'prune':<8}{'val@7':>10}{'test@7':>10}{'diff':>10}{'epoch':>8}{'weights':>12}")
    for r in results["pruning_results"]:
        lines.append(
            f"  {str(r['pruning_percentage']) + '%':<8}"
            f"{r['val_hits']:>10.4f}{r['test_hits']:>10.4f}"
            f"{r['test_hits_difference']:>+10.4f}{r['best_epoch']:>8}"
            f"{r['remaining_weight_parameters']:>12,}"
        )
    lines.append("")
    lines.append("PREDIKCIJA: NEXT / lottery_ticket_v3")
    lines.append("-" * 70)
    lines.append(f"  model za NEXT:        {results['best_for_next']['name']}")
    lines.append(f"  final train epoha:    {final_epoch}")
    lines.append(f"  Top-7 brojeva:        {next_combo}")
    lines.append("")
    lines.append("  Rang svih 39 brojeva po verovatnoci (broj:p):")
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


if __name__ == "__main__":
    main()




"""
Device: cpu  |  SEED=39  |  LAG=8  |  bez frekvencije
Izvlacenja=4626  uzoraka=4618  train=3694  val=462  test=462  feature_dim=405
============================================================
LOTTERY TICKET - LOTO 7/39 v3 (BEZ FREKVENCIJE)
============================================================

[Baseline] best-epoch trening...
  Epoch 01/60  loss=1.1382  val@7=1.3355  test@7=1.2338
  Epoch 02/60  loss=1.1325  val@7=1.2922  test@7=1.1926
  Epoch 03/60  loss=1.1206  val@7=1.2294  test@7=1.2900
  Epoch 04/60  loss=1.0998  val@7=1.2424  test@7=1.2597
  Epoch 05/60  loss=1.0721  val@7=1.2684  test@7=1.2359
  Epoch 06/60  loss=1.0420  val@7=1.3225  test@7=1.2316
  Epoch 07/60  loss=1.0105  val@7=1.2944  test@7=1.2338
  Epoch 08/60  loss=0.9782  val@7=1.2965  test@7=1.2338
  Epoch 09/60  loss=0.9450  val@7=1.2857  test@7=1.2403
  Epoch 10/60  loss=0.9110  val@7=1.2922  test@7=1.2338
  Epoch 11/60  loss=0.8762  val@7=1.3009  test@7=1.2359
  Epoch 12/60  loss=0.8409  val@7=1.3030  test@7=1.2165
  Epoch 13/60  loss=0.8056  val@7=1.3074  test@7=1.1970
  Epoch 14/60  loss=0.7705  val@7=1.2922  test@7=1.1775
  Epoch 15/60  loss=0.7358  val@7=1.2879  test@7=1.1970
  Epoch 16/60  loss=0.7019  val@7=1.2965  test@7=1.1905
  Epoch 17/60  loss=0.6691  val@7=1.2879  test@7=1.2013
  Epoch 18/60  loss=0.6378  val@7=1.2511  test@7=1.1970
  Epoch 19/60  loss=0.6076  val@7=1.2706  test@7=1.2100
  Epoch 20/60  loss=0.5790  val@7=1.2857  test@7=1.1840
  Epoch 21/60  loss=0.5522  val@7=1.2900  test@7=1.1710
  Epoch 22/60  loss=0.5278  val@7=1.3074  test@7=1.1818
  Epoch 23/60  loss=0.5085  val@7=1.3160  test@7=1.1905
  Epoch 24/60  loss=0.5005  val@7=1.3247  test@7=1.2208
  Epoch 25/60  loss=0.4894  val@7=1.3355  test@7=1.2468
  Epoch 26/60  loss=0.4692  val@7=1.3247  test@7=1.2424
  Epoch 27/60  loss=0.4428  val@7=1.3139  test@7=1.2424
  Epoch 28/60  loss=0.4222  val@7=1.2619  test@7=1.2294
  Epoch 29/60  loss=0.4062  val@7=1.2814  test@7=1.2035
  Epoch 30/60  loss=0.3934  val@7=1.3095  test@7=1.2208
  Epoch 31/60  loss=0.3789  val@7=1.2662  test@7=1.1948
  Epoch 32/60  loss=0.3711  val@7=1.2814  test@7=1.2121
  Epoch 33/60  loss=0.3664  val@7=1.2965  test@7=1.2489
  Epoch 34/60  loss=0.3531  val@7=1.3139  test@7=1.2359
  Epoch 35/60  loss=0.3348  val@7=1.2944  test@7=1.2641
  Epoch 36/60  loss=0.3169  val@7=1.2532  test@7=1.2511
  Epoch 37/60  loss=0.2999  val@7=1.2965  test@7=1.2619
  Epoch 38/60  loss=0.2879  val@7=1.2900  test@7=1.2446
  Epoch 39/60  loss=0.2771  val@7=1.3571  test@7=1.2489
  Epoch 40/60  loss=0.2698  val@7=1.3485  test@7=1.2489
  Epoch 41/60  loss=0.2665  val@7=1.3225  test@7=1.2944
  Epoch 42/60  loss=0.2620  val@7=1.2944  test@7=1.2944
  Epoch 43/60  loss=0.2581  val@7=1.2900  test@7=1.2489
  Epoch 44/60  loss=0.2464  val@7=1.3355  test@7=1.2468
  Epoch 45/60  loss=0.2348  val@7=1.3398  test@7=1.2727
  Epoch 46/60  loss=0.2254  val@7=1.3117  test@7=1.2576
  Epoch 47/60  loss=0.2161  val@7=1.3333  test@7=1.2706
  Epoch 48/60  loss=0.2097  val@7=1.3160  test@7=1.2662
  Epoch 49/60  loss=0.2048  val@7=1.2532  test@7=1.2857
  Epoch 50/60  loss=0.1956  val@7=1.2727  test@7=1.2835
  Epoch 51/60  loss=0.1808  val@7=1.2965  test@7=1.2814
  Epoch 52/60  loss=0.1722  val@7=1.2965  test@7=1.2662
  Epoch 53/60  loss=0.1684  val@7=1.3117  test@7=1.2857
  Epoch 54/60  loss=0.1659  val@7=1.2641  test@7=1.2727
  Epoch 55/60  loss=0.1637  val@7=1.2684  test@7=1.2641
  Epoch 56/60  loss=0.1580  val@7=1.2814  test@7=1.2641
  Epoch 57/60  loss=0.1499  val@7=1.2749  test@7=1.2857
  Epoch 58/60  loss=0.1410  val@7=1.2965  test@7=1.2641
  Epoch 59/60  loss=0.1350  val@7=1.3095  test@7=1.2403
  Epoch 60/60  loss=0.1315  val@7=1.3052  test@7=1.2381

Baseline: val@7=1.3571  test@7=1.2489  best_epoch=39
Parametri ukupno=141,863  weight_parametri=141,440

============================================================
PRUNING EKSPERIMENTI
============================================================

[Pruning 0%]
Preostalo weight parametara: 141,440 (100.0% weight-a)
Pruned: val@7=1.3571  test@7=1.2489  best_epoch=39

[Pruning 20%]
Preostalo weight parametara: 113,153 (80.0% weight-a)
Pruned: val@7=1.3203  test@7=1.2922  best_epoch=47

[Pruning 40%]
Preostalo weight parametara: 84,865 (60.0% weight-a)
Pruned: val@7=1.4113  test@7=1.2641  best_epoch=43

[Pruning 60%]
Preostalo weight parametara: 56,577 (40.0% weight-a)
Pruned: val@7=1.3593  test@7=1.2727  best_epoch=36

[Pruning 70%]
Preostalo weight parametara: 42,433 (30.0% weight-a)
Pruned: val@7=1.3247  test@7=1.2381  best_epoch=2

[Pruning 80%]
Preostalo weight parametara: 28,289 (20.0% weight-a)
Pruned: val@7=1.3052  test@7=1.2035  best_epoch=3

[Pruning 90%]
Preostalo weight parametara: 14,145 (10.0% weight-a)
Pruned: val@7=1.3312  test@7=1.2835  best_epoch=10

[Pruning 95%]
Preostalo weight parametara: 7,073 (5.0% weight-a)
Pruned: val@7=1.3333  test@7=1.2229  best_epoch=6

JSON sacuvan -> /Users/4c/Desktop/GHQ/KlasicniRegresori/Loto-7-39-Srbija-pruning/experiment_results_v3.json

Lottery Ticket Hypothesis - Loto 7/39 (v3, bez frekvencije)
======================================================================

  CSV:                  /data/loto7_4626_k44.csv
  Izvlacenja:           4626
  LAG:                  8
  Feature dim:          405
  Feature-i:            multi-hot sekvenca + transition + gap + lex + dX
  Frekvencija:          NE koristi se kao feature
  Mreza:                405 -> 256 -> 128 -> 39
  Epoha max:            60  LR=0.0008  weight_decay=0.0001
  Train/Val/Test:       3694 / 462 / 462
  SEED:                 39 (bez shuffle, CPU)

BASELINE
----------------------------------------------------------------------
  best_epoch:           39
  val pogodaka@7:       1.3571
  test pogodaka@7:      1.2489
  slucajna referenca:   1.2564 (7*7/39)

PRUNING REZULTATI
----------------------------------------------------------------------
  prune        val@7    test@7      diff   epoch     weights
  0%          1.3571    1.2489   +0.0000      39     141,440
  20%         1.3203    1.2922   +0.0433      47     113,153
  40%         1.4113    1.2641   +0.0152      43      84,865
  60%         1.3593    1.2727   +0.0238      36      56,577
  70%         1.3247    1.2381   -0.0108       2      42,433
  80%         1.3052    1.2035   -0.0455       3      28,289
  90%         1.3312    1.2835   +0.0346      10      14,145
  95%         1.3333    1.2229   -0.0260       6       7,073

PREDIKCIJA: NEXT / lottery_ticket_v3
----------------------------------------------------------------------
  model za NEXT:        pruned_40
  final train epoha:    43
  Top-7 brojeva:        (11, 17, 22, 23, 25, 31, 36)

  Rang svih 39 brojeva po verovatnoci (broj:p):
    25:0.956   36:0.951   22:0.949   31:0.884   11:0.868
    23:0.826   17:0.794   34:0.792   35:0.743   30:0.605
     2:0.469   32:0.402    3:0.265   13:0.169   10:0.169
    16:0.166   18:0.135   27:0.108    7:0.102   38:0.101
     9:0.093   20:0.058   33:0.045   29:0.043    1:0.008
     4:0.007   37:0.007   28:0.005   26:0.005   24:0.003
     5:0.001   39:0.000   14:0.000   12:0.000    8:0.000
    15:0.000   19:0.000    6:0.000   21:0.000

  JSON:                 /Loto-7-39-Srbija-pruning/experiment_results_v3.json

TXT sacuvan -> /Loto-7-39-Srbija-pruning/lottery_ticket_pruning_v3.txt
"""




"""
Pruning Analiza v3
Ovo je mnogo bolje od v2. 
Najvažnije: v3 bez frekvencije ipak daje pruning modele koji prelaze slučajnu referencu.

Slučajna referenca je 1.2564 pogodaka@7.

Rezultati:
Baseline: test@7 = 1.2489, malo ispod slučajne reference.
20% pruning: test@7 = 1.2922, najbolji test rezultat.
40% pruning: test@7 = 1.2641, malo iznad slučajne reference.
60% pruning: test@7 = 1.2727, iznad slučajne reference.
90% pruning: test@7 = 1.2835, drugi najbolji test rezultat.
70%, 80%, 95% padaju ispod baseline ili ispod slučajne reference.
Glavni zaključak: pruning je ovde stvarno pomogao. Baseline sam ne probija slučajnu referencu, ali pruned modeli 20%, 40%, 60%, 90% probijaju. To znači da je deo mreže verovatno učio šum, a pruning je izbacio deo tog šuma i ostavio korisniji podsklop.

Najbolji po testu je 20% pruning:

test@7 = 1.2922
razlika od baseline: +0.0433
preostalo weight parametara: 113,153 od 141,440

Najbolji po validaciji je 40% pruning:

val@7 = 1.4113
test@7 = 1.2641
zato ga je skripta izabrala za NEXT model (pruned_40), 
jer se model bira po validaciji, ne po testu. 
To je metodološki ispravno, jer test ne treba koristiti za izbor modela.

Bitno zapažanje: 
90% pruning je vrlo zanimljiv. 
Sa samo 14,145 weight parametara daje test@7 = 1.2835, skoro kao najbolji test model. 
To je najjači lottery-ticket signal: mala mreža zadržava ili poboljšava performansu.

Zaključak za pruning: 
v3 je uspeo bolje od v2. 
Bez frekvencijskih feature-a, pruning ipak izvlači slab sekvencijalni signal. 
Najjači test rezultat je 20%, najmetodološki izabran model za NEXT je 40%, a najinteresantniji sparse ticket je 90%.
"""





"""
lutrija je u suštini i.i.d.; 
teorijski maksimum je oko slučajne reference (1.256). 
Niko ne može garantovati stabilan „dobitak" iznad toga. 
Ali u mom v2 postoji jedna prava, popravljiva greška koja besplatno daje realan dobitak, plus par jačih izbora.

Glavni nalaz iz log-a: 
v2 koristi poslednju epohu (epoch 40), a model overfituje — loss pada (0.51→0.25) dok hits@7 posle ~epohe 3–5 opada. 
Epoha 3 je imala 1.2965, a uzeta epoha 40 samo 1.2359. Dakle biranjem najbolje epohe već dobijaš ~1.29, što probija slučajnu referencu.

Najbolja optimizacija:

Best-epoch selection preko zasebnog validacionog skupa (ne test) — 
vraća najbolji model, ne overfit-ovani poslednji. (najveći realan dobitak)
pos_weight ≈ 32/7 u BCEWithLogitsLoss — ispravlja disbalans (samo 7 od 39 su jedinice); 
model trenutno gura sve ka niskim verovatnoćama.
Bogatije feature-e nad lex-om/brojevima: 
za svaki broj — gap od poslednje pojave + rolling frekvencija (prozori 10/25/50/100) + hot/cold. 
To je jedini realni slab signal koji lutrija ponekad ima (frekvencijski drift).
Weight decay + mali dropout protiv overfita.
Ensemble preko pruned tiketa (prosek verovatnoća) za stabilniju NEXT kombinaciju.
NEXT predikcija: retrain na svim podacima sa izabranim brojem epoha, pa top-7.
Determinizam ostaje: seed=39, bez shuffle-a, CPU.

Realno očekivanje: 
hits@7 oko 1.28–1.32 na testu (malo iznad slučajnog), 
i stabilnija NEXT kombinacija — ali bez obećanja „sigurnog" dobitka.


Bolja v3 bez frekvencije:
Osnov = sekvencijalni obrazac zadnjih kola: 
multi-hot lagovi + dX/lex prirastaji + gap/smer/promena, ne brojanje pojavljivanja.
Validacioni izbor najbolje epohe, jer rešava overfit.
pos_weight, dropout, weight decay.
Pruning/lottery-ticket.
NEXT se bira iz neuralnih verovatnoća/ensemble-a, ne iz tabele frekvencija.
Dakle: bez rolling frequency feature-a. Gaps mogu jer nisu isto što i frekvencija — 

Osnov v3 neće biti frekvencija: 
koristiće redosled/sekvencu, gap od poslednje pojave, 
lex i dX prirastaje, validacioni izbor najbolje epohe i lottery-ticket pruning.

vremenski train/val/test, best-epoch po validaciji, pos_weight, weight decay, pruning, 
i NEXT predikcija bez frequency feature-a.


Bez frekvencije: nema rolling frequency, nema brojanja pojavljivanja kao osnove.
Feature-i su: zadnjih 8 kola kao sekvenca, transition poslednja dva kola, gap od poslednje pojave, lex sekvenca i dX prirastaji.
Dodato je train/val/test vremensko sečenje.
Model bira best epoch po validaciji, umesto da uzima poslednju epohu i overfituje.
Loss koristi pos_weight=(39-7)/7.
AdamW + weight_decay, bez dropout-a.
Seed ostaje 39, bez shuffle-a, CPU.

Redosled pokretanja:
cd 
python lottery_ticket_pruning_v3.py
python visualize_results_v3.py

Ili umesto glavnog:
python run_experiment_v3.py
"""







"""
Lottery Ticket Hypothesis - Neural Network Pruning
"""
