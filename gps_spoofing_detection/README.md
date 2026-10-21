# GPS Spoofing Detection for UAVs under Unseen Flight Conditions

GPS spoofing component of the Smart Drone Forensics and Cyber Investigation System (J26-CS-306).
Random Forest and XGBoost detect GPS spoofing in UAV flight telemetry, and the evaluation measures how much
performance drops on a flight type the models were not trained on.

## Key findings (simulated data, one dataset)
- Same model and features: row-level ROC-AUC falls from 0.90 (random row split) to 0.66 (flight-level split) to 0.52 (unseen flight condition).
- Whole-flight detection with multi-window features judged 83% of flights correctly on known flight types (guessing = 60%), but only about 0.67 balanced accuracy on an unseen flight type (guessing = 0.50). Both use exploratory features.
- Random Forest and XGBoost performed almost the same.

## Quick start
1. Install packages: `python -m pip install -r requirements.txt`
2. Download the Merged folder from https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset and place it here as `Merged/`.
3. Run `step3.py` (builds `combined.pkl`), then the experiment scripts below.

| Script | Purpose |
|---|---|
| step3.py | Load 360 flights, clean, resample to about 25 Hz |
| step9.py | Three evaluation protocols (random split, flight-level, leave-one-condition-out) |
| step10.py, step11.py, step12.py | Feature ablation, alarm thresholds, class weighting |
| step14.py | Final detector and `gps_spoofing_results.json` |
| step15.py | Failure analysis, speed test, integrity manifest, leakage test |
| step16.py + demo_app.py | Demo model and Flight Analyser window |
| step21.py, step22.py | Multi-window features on known and unseen flight types |
| gps_common.py | Shared feature code and `analyze_flight()` |

(Other step files are data checks, extra experiments and charts.)

## Limitations
- All data are simulated (PX4 SITL), so real-flight performance is unknown.
- Only three flight conditions; unseen-condition results vary a lot between them.
- Refined and multi-window features were designed after seeing results, so they are exploratory.
- Reasons for missed attacked flights were not identified.
- A model score is an indicator to investigate, not proof of an attack.

## Output for the system
`gps_spoofing_results.json` holds one record per flight (condition, detected event, confidence score, threshold, evidence) plus a reliability note. Field names are still to be agreed with the other components.

## Author
Rasadi W.A.P(IT23331822) 
B.Sc. (Hons) in Information Technology (Cyber Security), SLIIT.
Parts of the code and documentation were developed.
