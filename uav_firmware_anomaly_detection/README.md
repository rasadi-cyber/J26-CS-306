# UAV Firmware Anomaly Detection Using Machine Learning

Component of the group project **Smart Drone Forensics and Cyber Investigation System Using Machine Learning**.

It screens UAV flight-controller firmware (ArduPilot `.apj`, PX4 `.px4`) **statically** (never executed) and reports an anomaly score, the vehicle type, and UAV-specific reasons such as missing geofence, failsafe or return-to-launch content.

## What makes it UAV-specific

- Reads the vehicle type from the firmware (ArduCopter, ArduPlane, helicopter, PX4) and uses a model trained for that vehicle class. Non-aerial builds (Rover, Sub, Tracker, Blimp) are refused.
- 60 flight-stack marker features: presence and density of flight-control, navigation/RTL, arming/failsafe, geofence and state-estimation parameter and message names.
- Explanations are phrased in flight-safety terms ("geofence content not found"), not raw byte statistics.

## Method

- **Main model:** One-Class SVM (RBF) on the UAV marker features, one model per vehicle class (ArduPilot multirotor, ArduPilot fixed-wing, ArduPilot general, PX4). Pipeline: variance filter, standard scaler, model.
- **Baseline:** Isolation Forest on the same features.
- **Also compared:** generic byte/entropy/string features (404 features) and the combined 464-feature set.
- **Data:** 701 unique firmware images (ArduPilot Copter/Plane releases 4.0.x to 4.7.x from the official firmware server, plus PX4 v1.15 to v1.17). Split by board group so no board appears in two splits: 357 train, 120 validation, 224 test. Thresholds are the 95th percentile of clean validation scores.

## Results

Controlled stress tests on the reserved **test split** (used once, frozen models; `reports/v2/final_test_summary.csv`):

| Scope | Clean flagged | Byte-level edits detected | Safety-content removal detected | Isolation Forest baseline (removal) |
|---|---|---|---|---|
| ArduPilot multirotor (114 clean) | 10.5% | 0.5% | 100% | 38.9% |
| ArduPilot fixed-wing (88 clean) | 11.4% | 1.9% | 98.7% | 23.6% |
| ArduPilot helicopter (15 clean, general model) | 6.7% | 0.0% | 100% | 22.2% |
| PX4 (7 clean) | 0.0% | 0.0% | 100% | 0.0% |

The generic byte-feature model detected essentially none of the safety-content removals (0 to 1% for ArduPilot).

## Limitations

- The "modified" firmware is **synthetic** (byte edits and zeroed flight-safety marker names). Detecting removal of marker content is largely by construction. This is not evidence against real attackers.
- Small generic byte edits in ArduPilot firmware are **not** detected.
- On unseen test firmware about 10% of clean ArduPilot builds were flagged, double the 5% target. Legitimate builds may omit features, so an alert needs investigation.
- PX4 has few samples (31 images, 7 clean test); treat PX4 results as indicative only.
- Helicopter builds had no validation samples; they use the general ArduPilot model.
- Not a malware detector and not proof of authenticity or tampering.

## Setup and use

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt pytest
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m streamlit run app.py
```

Command line:

```powershell
python src\analyze_firmware.py <firmware.apj|.px4> --output reports\result.json
```

Demo file: `demo\copter_safety_content_removed.apj` (Copter 4.7.2 with safety markers zeroed).

Model files (`models/`) and raw firmware (`data/raw/`) are git-ignored. Rebuild with:

1. `src\build_feature_dataset_v2.py`
2. `src\prepare_splits_v2.py` (fixed splits, seed 42)
3. `src\build_stress_v2.py` (validation stress samples)
4. `src\train_eval_v2.py` (trains and selects on validation only)
5. `src\final_test_v2.py` (run once on the test split)

## Structure

- `src/` feature extraction, training, analysis, vehicle routing (`model_routing.py`), UAV reasoning (`uav_reasoning.py`), PDF report
- `app.py` Streamlit interface
- `reports/v2/` split, validation and final test results
- `tests/` unit tests
