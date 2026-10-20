# FL DoS Swarm

A federated DoS/DDoS detection simulation for a 10-drone UAV swarm on UAV-NIDD. The server combines models with Jiang et al. (2024): a distance filter, then voting, then DQN weighting. Component J26-CS-306.

## Run it

- **App:** double-click `FL DoS Swarm.app`. It opens the dashboard at `http://localhost:8501`, showing the `FINAL` run.
  - Start new runs from the sidebar under **➕ New experiment**.
  - Build the carried-over DQN under **🧠 Pre-train DQN**.
  - If you move this folder, run `./build_app.sh` again.
- **Dashboard only:** `.venv/bin/streamlit run app.py`
- **Command line:**
  ```
  .venv/bin/python run_experiment.py --name my-run                  # configs/default.json, 5 seeds, all cores
  .venv/bin/python run_experiment.py --seeds 100 101 --set clients.num_poisoned=1 --name check
  .venv/bin/python run_experiment.py --pretrain-dqn --seeds 100 101 102
  .venv/bin/python -m pytest -q tests
  ```

Jobs (one per method and seed) run in parallel, one per CPU core. The 30-job final run takes about 2 minutes.

## Dashboard: what each chart is for

| Tab | Chart | Paper counterpart | What it should show |
|---|---|---|---|
| Overview | Four-panel bars: all classes, rare class, attacks caught, normal traffic kept (1 − false alarms) | Table 2 | FedAvg keeps the rare class but misses attacks; Multi-Krum catches attacks but loses the rare class; Jiang does both |
| Overview | Share of the global model per drone type | Fig. 6 | Poisoned drones 0%; the rare drone ≥ a regular drone |
| Overview | Share of votes that readmitted a dropped model, per drone type | Sec. 3, voting layer | Poisoned 0%; rare and regular drones high |
| Overview | Attacks missed / macro F1 vs number of poisoned drones | – | FedAvg breaks as poison grows; Jiang stays flat |
| Overview | Robustness and sanity table | – | Jiang holds under boosted, sign-flipped and colluding attacks; costs nothing with no attackers |
| Swarm monitor | Per-drone table: this round plus record so far | – | One round is a snapshot; the "so far" columns show who really gets back in |
| Decisions | Per-drone share of rounds in each state; share over time | Fig. 6 over time | Poisoned drones solid red; others green or blue |
| Curves | Smoothed validation curves with the spread across seeds | Fig. 5 | Jiang tracks Multi-Krum until voting starts, then climbs to the no-attacker reference |
| Results | Table (incl. false alarms, detection rate); significance tests; confusion matrices summed over seeds | – | De-authentication row on the diagonal for Jiang, off it for Multi-Krum; FedAvg's Normal column absorbs attacks |

## Data (`data/processed/`)

| File | Content |
|---|---|
| `uavnidd_clean.csv` | Cleaned UAV-NIDD: 7 classes, identifier and capture-setting columns removed, window features added, exact duplicates removed. `_order` (capture order) and `_source` (original attack type) are bookkeeping for the split, never model inputs |
| `clean_report.json` | Which columns were dropped and why, plus class counts before and after |
| `seed_<s>/drone_XX_train.parquet` | One drone's training rows. `TrainLabel` is flipped for poisoned drones |
| `seed_<s>/drone_XX_test.parquet` | That drone's local test rows (its voting set) |
| `seed_<s>/validation`, `test`, `public_exam` | Validation (tuning), test (final results only), and the balanced public exam (server side, DQN reward) |

### Classes

Normal, DDoS, UDP Flooding, ICMP Flooding, DoS, **De-authentication** (the rare class), and Other attack (BruteForce, MITM, FakeLanding, Replay, Jamming, Scanning, Reconnaissance).

### Preparation (standard practice, chosen on principle, not on results)

1. **Behaviour-only inputs** (`data.feature_set = size_timing`). The model sees only **frame length, inter-arrival time**, and their causal window statistics. UAV-NIDD shifts its header columns between capture sessions, so header fields (and the frequent-value flags used before 2026-10-13) identify the *recording*, not the attack. They are all dropped. Size and timing mean the same thing in every capture, even encrypted ones.
2. **Causal window features** (`data.window_sizes = [10, 100, 1000]`), as in packet-based IDSs such as Kitsune (Mirsky et al., NDSS 2018). Over the previous 10, 100 and 1,000 packets of the same stream: frame length mean/std/min/max, gap mean/std/max, packet rate and byte rate. That makes **29 features** in total. Streams are **capture files found without labels** (packet numbers continue +1 inside a capture; 28 captures, none mixing labels). In mixed traffic the windows run over the merged link stream.
3. **True duplicates removed.** Only the same packet stored twice is removed (same label, capture position, size and gap; UAV-NIDD's DoS capture appears twice). Repeated traffic later in time (periodic telemetry, floods) is real and stays. **845,846 rows** remain.
4. **Split.** Rolling-origin time folds per attack type (see *Rolling-origin evaluation*); Normal, recorded as 5 separate captures, rotates whole captures between drones, validation and test.
5. **Tuning on validation only.** The test split is evaluated once, in the final run.
6. **Balance.** No rows are deleted or synthesised. Each drone uses an inverse-frequency class-weighted loss on its own labels.

Central gradient-boosted tree on validation: macro-F1 0.912 with 100% of the training rows and 0.914 with 10% (flat learning curve), so **there is enough data**.

### Capture-family caveat (state it in the viva)

UAV-NIDD recorded Normal traffic only on the Ethernet/IP side, and the Wi-Fi attacks (UDP, ICMP, DoS, De-authentication, Replay) only as over-the-air radio captures. So "which kind of capture" partly gives away the answer, and no cleaning can remove that, because the dataset contains no normal Wi-Fi traffic. Evidence that the model still learns behaviour: DoS and UDP flooding are both Wi-Fi captures with identical per-packet fields, yet they are separated (test F1 DoS 0.95, UDP 0.92) using size and timing alone.

### Why the rare class is De-authentication, not plain "DoS"

De-authentication has a genuine signature: very small management frames (median 39 bytes) aimed at the access point (`wlan.bssid` set in 72% of its rows, almost never elsewhere). It is naturally scarce (4.6%). Plain DoS stays in the data as a common class.

### Why "Other attack" is included

Without it, De-authentication frames were "detected" even by models that never trained on them, because they were the only management-frame traffic. Replay and Scanning share De-authentication's frame layout, so including them means the rare class has to be genuinely learned.

## Swarm setup (paper Sec. 5.1)

10 drones per seed, with roles assigned randomly by the seed:

- **6 regular drones.** Each holds **every common class** in different amounts, as in the paper. Each class is split with proportions `0.5 · Dirichlet(α = 1) + 0.5 · even share` (`clients.class_floor = 0.5`).
- **1 rare drone** has a common share **plus every training De-authentication row**. No other drone trains on it.
- **3 poisoned drones** have a common share, but **every label is flipped to Normal** (the paper's attack). They vote randomly. 3 of 10 is under the paper's limit of fewer than half.

Each drone votes on **its own local test traffic plus a small third-party sample with all labels** (the public exam, 200 rows per class), as in Jiang Sec. 5.1. The server uses the same public exam for the DQN's state and reward. No drone trains on it, and no drone's data leaves the drone.

Training: the same MLP on every drone (256-128), **80 local SGD steps per round, lr 0.1, 500 rounds** (as in the paper; voting from round 100), inverse-frequency class weights, plain cross-entropy (as in Jiang). Chosen on validation by a method-neutral rule (attack-free FedAvg's validation score, `x01`-`x08`); the curve plateaus at 500 rounds.

### Mixed traffic (how training and testing data look)

A UAV link never carries one attack in isolation. Every split is rebuilt as **normal traffic with attack bursts interleaved in time** (`data.traffic = mixed`, `fldos/traffic.py`), from that split's own rows only. Each burst (200-2,000 packets of one attack type) starts at a random time inside a normal background (0.5-2× the burst length); gaps and window statistics are recomputed over the merged stream. Each packet is labelled with what the link is doing at that moment: the attack type if attack packets make up ≥ 50% of the last 10 packets, otherwise Normal. This is how link-level DoS detectors are judged.
- **Normal traffic is scarce in UAV-NIDD**, so normal packets are reused as background across bursts (each roughly 50-190 times, each time in a different attack context). This is a form of augmentation, stated openly.
- The per-capture (unmixed) test is kept as a **clean reference** (Results tab, expander).

### Rolling-origin evaluation
The 5 final seeds are 5 different **time folds**: the test block starts at 60/65/70/75/80% of each attack's timeline (validation is the block just before; drones train on everything earlier), and the Normal test capture rotates. Each seed tests on different traffic that is later than its training traffic, so the spread across seeds reflects data variability, not just training randomness.

### DQN variants
| Method | DQN starts from | Learns during the run |
|---|---|---|
| `jiang_fixed` | – (fixed weight 1/(K+1)) | – |
| `jiang_dqn` | scratch | yes |
| `jiang_dqn_pre` | the stored DQN for **this setup**, accumulated over every pre-training run of the setup (`dqn_store/pre_setup_<key>.pt`) | yes (final runs never save back) |
| `jiang_dqn_prexv` | the DQN carried **across versions** (`dqn_store/pre_xv.pt`): network kept through setup changes, replay memory and input statistics reset at each change, then accumulated like the one above | yes |

Pre-training uses non-final seeds only (≥ 100; enforced in code), and each store keeps a lineage log of what it learned from.

## Aggregation methods

| Method | What it does |
|---|---|
| `fedavg` | FedAvg: average of every update, weighted by data size |
| `multi_krum` | Keeps the K = 5 updates with the lowest Krum score (f = 3) |
| `jiang_fixed` | Jiang filter + vote; voted-in models get the fixed weight 1/(K+1) (the paper's ablation) |
| `jiang_dqn` | Jiang filter + vote + a DQN that sets the weight of every accepted and voted-in model (online: a fresh DQN each run) |
| `jiang_dqn_pre` | Same, starting from the DQN accumulated over this setup's pre-training runs (the paper's pre-trained DRL) |
| `jiang_dqn_prexv` | Same, starting from the DQN carried across versions (transfer, then fine-tune) |
| `fedavg_clean` | Reference only: FedAvg with the poisoned drones removed by an oracle |

This is a controlled experiment: data, split, drones, attackers, local model and local training are identical for every method; only the server-side aggregation differs.

### Jiang et al. (Algorithm 1), as implemented

- **Filter.** Each update's score is the sum of its squared distances to all other updates. The K = 5 lowest form M_B (data-size weighted, times the DQN's weight for each). The other 5 go to the vote.
- **Vote (Eq. 7-8).** Same round as the filter, frozen for the first 20% of rounds. Every drone tests each dropped model on its local test traffic plus the third-party sample against M_B and votes yes if its balanced accuracy is ≥ M_B's within three standard errors. Readmitted if yes > N/2.
- **Combine (Eq. 9).** M_G = (M_B + Σ w·M_V) / (1 + Σ w).
- **DQN.**
  - **Decides for** every accepted and every readmitted model (the proposal: "per accepted/readmitted update").
  - **State (paper Sec. 4):** round / R, freshness r′/r, the model's public-exam score, its yes-vote share; standardised.
  - **Actions:** a multiple of the paper's fixed weight, {0, 0.5, 1, 2, 6, 18} × 1/(K+1) for readmitted models and × |D| for accepted ones. Action 1 *is* the fixed-weight method. **A model the vote readmitted can't get weight 0** (the DQN may down-weight it but not silently overrule the vote).
  - **Reward:** the gain in public-exam score of the Eq. 9 aggregate over giving this model the fixed weight, with every other decision unchanged. The exam is scored as natural traffic: macro-F1 with each class's rows reweighted to its share of normal traffic (`reward_metric = macro_f1_prior`).
  - **Training:** standard DQN (replay, target network, ε-greedy, Huber loss; γ 0.1, 16 updates of batch 64 per decision).
- **Roles.** The code never reads drone roles, except for the attackers' own behaviour and the oracle reference run.

### Where this departs from the paper, and why

0. **Voters' test sets (now as in the paper).** From 2026-10-07 to 2026-10-13 voters used local data only; with mixed traffic and realistic local training that made voters reject the rare drone 46-48% of the time (`x10`), because none of them had De-authentication traffic. The paper's design (local test + third-party sample with all labels) was restored: rare drone voted out 1-2%, poisoned 100% (`x11`).
1. **Balanced accuracy for votes.** With De-authentication at a few percent, plain accuracy barely notices it.
2. **"≥" within three standard errors.** M_B is an average, which beats any single honest model by a hair; with fine-grained balanced accuracy an exact "≥" rejected honest drones ~70% of the time. Poisoned models score ~50 points lower and are still always rejected.
3. **Same-round vote** (`vote_lag = 0`): the fast-network case of the paper's bounded lag.
4. **The DQN also weighs accepted models** (the proposal's architecture table). Jiang's DRL only sets w_Vq.
5. **DQN reward and actions.** The paper rewards the accuracy improvement. Ours is the improvement **over the fixed weight**, scored on the public exam **as natural traffic**, and its actions are multiples of the fixed weight. Reason: the public exam is class-balanced, so a plain reward never saw the cost of false De-authentication alarms on big classes and over-weighted the rare drone, ending slightly below the fixed weight (0.885 vs 0.894, `u03`). With the change it is above it on every tuning seed (`u10`).
6. **Pre-trained DQN keeps fine-tuning** (frozen didn't transfer between seeds).
7. **Input features, mixed traffic and window statistics** belong to the client model and data, not to Jiang's method; every method uses them.
8. **No weight 0 after a readmission** (see DQN actions). The paper doesn't say whether 0 is allowed; this follows the vote's decision.
9. **Pre-trained DQN definition.** The paper only says the DRL is "pre-trained" / "conducted beforehand". We provide two readings: accumulated over this setup's earlier runs, and carried across versions (transfer, then fine-tune).

## Final results: v12c (`runs/FINAL` = `runs/v12c-FINAL` = v12b + Fed-GraB fixed with the authors' class gating, 5 rolling time folds = seeds 1-5, mixed-traffic test, mean of the last 10 global models)

v12b = v10 + learning rate 0.05 (v9's; fixes Multi-Krum) + cluster-robust vote standard error (fixes honest drones being voted out) + DQN weighs voted-in models only (Jiang) + FedWGCA and Fed-GraB baselines, 750 rounds (validation plateau). Why each change: `KEY_FINDINGS.md` 30-40, `notes/VERSION_HISTORY.md`.

| Method | Macro F1 | Accuracy | De-auth F1 | Attacks missed | False alarms | Poisoned share | Rare share |
|---|---|---|---|---|---|---|---|
| FedAvg, no attackers (reference) | 0.780 | 0.944 | 0.70 | 0.6% | 3.7% | – | 18% |
| FedAvg | 0.770 | 0.905 | 0.77 | **13.0%** | 0.8% | 9% | 13% |
| Multi-Krum (robust) | 0.666 | 0.938 | **0.00** | 0.6% | 3.2% | 0% | **0%** |
| FedWGCA (robust pole) | 0.659 | 0.912 | **0.00** | 9.7% | 0.8% | 8% | **0%** |
| Fed-GraB (rare pole) | 0.706 | 0.911 | 0.69 | **12.5%** | 0.5% | 9% | 12% |
| Jiang, fixed weight | 0.777 | 0.940 | 0.68 | 0.6% | 4.6% | 0% | 13% |
| Jiang + DQN | 0.774 | 0.939 | 0.70 | 0.5% | 4.6% | 0% | 20% |
| Jiang + pre-trained DQN | 0.774 | 0.941 | 0.69 | 0.6% | 4.3% | 0% | 21% |
| Jiang + DQN pre-trained across versions | 0.774 | 0.939 | 0.71 | 0.5% | 4.6% | 0% | 25% |

- Jiang vs FedAvg: missed attacks 0.5% vs 13.0% (p = 0.003, 5/5 seeds). Jiang vs Multi-Krum: macro +0.11 (p = 0.003), De-auth 0.70 vs 0 (p = 0.001). DQN vs fixed: not significant in macro; the DQN gives the rare drone more weight (13% → 20-25%) and De-auth F1 varies less across seeds.
- Every honest drone is voted back in ≥ 55% of the times it is dropped; poisoned 0%; rare 100%. No confusion-matrix row ≥ 99.5%.
- Robustness/sanity runs for v12b: not yet run.

## Previous final results: v10 (`runs/FINAL`, 5 rolling time folds = seeds 1-5, mixed-traffic test, mean of the last 10 global models)

| Method | Macro-F1 | Accuracy | DoS detected | DoS-type F1 | De-auth F1 | Attacks missed | False alarms | Rare voted back in | Poison voted back in |
|---|---|---|---|---|---|---|---|---|---|
| Jiang + DQN | **0.783 ± 0.061** | 0.943 | **99.8%** | 0.71 | **0.75** | **0.5%** | 4.1% | 99% | **0%** |
| Jiang + pre-trained DQN | 0.782 ± 0.060 | 0.943 | 99.8% | 0.71 | 0.74 | 0.6% | 4.1% | 99% | **0%** |
| Jiang + DQN pre-trained across versions | 0.780 ± 0.055 | 0.943 | 99.8% | 0.71 | 0.73 | 0.6% | 3.9% | 99% | **0%** |
| Jiang, fixed weight | 0.774 ± 0.066 | 0.940 | 99.8% | 0.70 | 0.69 | 0.5% | 4.5% | 99% | **0%** |
| FedAvg | 0.764 ± 0.069 | 0.898 | 92.8% | 0.71 | 0.77 | **14.4%** | 0.8% | – (always in) | – (always in) |
| Multi-Krum | 0.465 ± 0.131 | 0.716 | 59.7% | 0.41 | **0.00** | **53.6%** | 0.7% | – (always dropped) | picks them on 4/5 seeds |
| FedAvg, no attackers (ref.) | 0.784 ± 0.057 | 0.945 | 99.8% | 0.71 | 0.73 | 0.5% | 3.8% | – | – |

Every attack burst is detected (100%) with the first alert immediately (median 0 packets). Jiang's false alarms are about 11 per minute of normal traffic, against FedAvg's 2. The poisoned FedAvg calls more traffic "Normal": fewer false alarms, many more missed attacks.

**Significance** (paired over the 5 folds): Jiang + DQN vs FedAvg: macro-F1 +0.019 (p = 0.048), missed attacks −13.9 points (p = 0.001, 5/5); vs Multi-Krum: +0.32 (p = 0.006), De-auth +0.75 (p = 0.0001); vs fixed weight: +0.009 (n.s.; better on 1/5 folds for macro-F1, 2/5 for De-auth).

**What this shows**
- **Middle ground:** FedAvg keeps the rare class but lets the attackers in (attacks called Normal: DoS 19%, Other 21%, ICMP 11%). Multi-Krum always drops the rare drone, and with 80 local steps the 3 identical poisoned models form the tightest cluster, so Krum keeps them (a known Krum weakness under non-IID data). Jiang keeps the rare drone (99%), rejects the poisoned drones (0%) and equals the attack-free reference.
- **DQN vs fixed weight:** the DQN variants are slightly ahead on average (0.780-0.783 vs 0.774; De-auth 0.73-0.75 vs 0.69), but not consistently across folds. Pre-trained ≈ online.
- **1 poisoned drone** (`FINAL-poison1`, 3 folds): FedAvg 0.814 (2.7% missed), Jiang + DQN 0.800 (0.6%), Multi-Krum 0.677. FedAvg's damage grows with the number of attackers. The other robustness runs were not completed for v10.

## Findings and limitations to report

1. **IDS-level accuracy, honestly measured:** Jiang + DQN accuracy 0.943, DoS detection 99.8%, 0.5% of attacks missed, every burst detected immediately, on mixed traffic with test folds later in time than training. Macro-F1 is 0.78 because naming the exact DoS type is hard (DoS ↔ UDP flood). Even a central MLP on the same data reaches only 0.78 (tree 0.83).
2. **False alarms (~4%, ~11/min)** come from normal packets just after an attack burst, whose windows still hold attack packets. An alert-confirmation rule (alert only after several flagged packets) is the standard remedy.
3. **The DQN beats the fixed weight only slightly and not on every fold**; pre-trained ≈ online. In the 20-step pilots, weighting had no measurable lever.
4. **Multi-Krum is sensitive to local training length:** at 80 steps it keeps the colluding poisoned drones; at 20 steps (pilots p01-p03) it cuts both the poisoned and the rare drone.
5. **UAV-NIDD limits:** one recording per attack, only 5 short normal captures (reused as background in mixed traffic), no low-rate DoS.
6. **Sign-flip attacks** can pass distance filters (earlier versions); a direction check is future work.
7. **Simulation limits:** one machine; drones train after landing (offline), so dropouts/asynchrony aren't modelled; voters see other drones' models (the paper's open-FL assumption).

## Tuning protocol

- Tuning used seeds 100–104 only (the `t*`, `u*` and `retune-*` runs); the final pre-trained DQN was trained on 100–102. Seeds 1–5 were run once with frozen settings.
- Every attempt and its outcome is in `notes/TRIED_EXPERIMENTS.md` (one line each) and `notes/CHANGELOG_EXPERIMENTS.md` (full numbers). Earlier final runs are kept as `runs/old-FINAL-v*`.
- An earlier version had a bug: drones' parameters were views of the global vector. All numbers come from the fixed code; `tests/test_core.py` guards against it, alongside tests for the vote, attacks, window features and the natural-mix reward (8 tests).

## Mixed-traffic test (separate evaluation; `mixed_traffic.py`, dashboard tab 🧪)

The standard test computes window statistics over one recording at a time, so every window holds a single class, which is cleaner than real traffic. This separate test rebuilds test traffic the way a drone would see it: all Normal test packets as background, with one attack burst (up to 2,000 packets of one attack type's **test** traffic) inserted at a random time, 3 placements per attack type (47,931 packets). The two streams are merged by arrival time, and gaps and window statistics are recomputed over the mix. The final global models (round 300) are reused without retraining; the other results are unchanged.

| Method | Macro-F1 | Accuracy | De-auth F1 | Attacks missed | False alarms |
|---|---|---|---|---|---|
| Jiang, fixed weight | 0.553 ± 0.012 | 0.626 | 0.83 | 11.3% | 29.0% |
| Jiang + pre-trained DQN | 0.551 ± 0.008 | 0.621 | 0.72 | **9.1%** | 31.9% |
| Jiang + DQN | 0.532 ± 0.026 | 0.614 | 0.72 | 9.7% | 31.0% |
| FedAvg, no attackers (ref.) | 0.559 ± 0.004 | 0.630 | 0.76 | 10.6% | 29.4% |
| FedAvg | 0.457 ± 0.070 | 0.588 | 0.68 | **41.6%** | 6.4% |
| Multi-Krum | 0.411 ± 0.035 | 0.526 | **0.00** | 10.7% | 30.2% |

Scores drop for every method: normal packets just after a burst still have attack packets in their window (~30% false alarms), and the first packets of a burst still look normal. The comparison keeps its shape: Jiang is at the attack-free reference, FedAvg misses four times as many attacks, and Multi-Krum is still blind to the rare class. These numbers are the realistic ceiling for a per-packet window IDS trained on pure single-recording windows. Training on mixed windows would be the next step.
