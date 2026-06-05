"""
Wrapper za pokretanje Loto 7/39 lottery-ticket eksperimenta (v2),
sa potisnutim warning-ima. Deterministicki (seed=39).
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "lottery_ticket_pruning_v2.py")

print("Pokretanje Loto 7/39 lottery-ticket eksperimenta (v2)...")
print("(warning-i potisnuti radi cistijeg izlaza)\n")

try:
    with open(MAIN, "r", encoding="utf-8") as f:
        code = compile(f.read(), MAIN, "exec")
    exec(code, {"__name__": "__main__", "__file__": MAIN})
except Exception as e:
    print(f"\nGreska: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


"""
Pokretanje Loto 7/39 lottery-ticket eksperimenta (v2)...
(warning-i potisnuti radi cistijeg izlaza)

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
  Top-7 brojeva:        (3, 16, 17, 18, 20, 29, 32)

  Rang svih 39 brojeva po verovatnoci (broj: p):
    32:0.951   18:0.896   20:0.617   29:0.452    3:0.433
    17:0.274   16:0.228    6:0.217    9:0.212   34:0.192
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
samo wrapper, ne model.

Radi tri stvari:
Potiskuje warning-e (warnings.filterwarnings, PYTHONWARNINGS=ignore) da izlaz bude čist.
Nađe lottery_ticket_pruning_v2.py u istom folderu.
Pokrene ga preko compile + exec, sa __name__="__main__" i pravilnim __file__.
Dakle rezultat u dnu fajla nije rezultat wrappera kao modela, 
nego kompletan izlaz iz lottery_ticket_pruning_v2.py. 
Wrapper je korektan i nema posebnu analitiku osim toga da je praktičan za pokretanje.

Wrapper je samo čistiji terminal izlaz.
"""
