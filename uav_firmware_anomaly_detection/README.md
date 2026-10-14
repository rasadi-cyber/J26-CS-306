# UAV Firmware Anomaly Detection Using Machine Learning

## Project Overview

This component focuses on identifying potentially suspicious modifications in UAV firmware using static analysis and machine learning techniques.

It is part of the group project, **Smart Drone Forensics and Cyber Investigation System Using Machine Learning**.

## Objectives

- Collect and validate PX4 and ArduPilot firmware samples.
- Extract static features from firmware files.
- Use a One-Class Support Vector Machine (One-Class SVM) to identify unusual firmware samples.
- Compare the results with an Isolation Forest baseline.
- Present detection results to support further forensic investigation.

## Planned Workflow

1. Firmware file intake and validation.
2. SHA-256 hash calculation and basic metadata collection.
3. Static feature extraction.
4. Feature preprocessing and anomaly detection.
5. Presentation of results and supporting information.

## Technologies

- Python
- scikit-learn
- NumPy
- pandas
- Python standard libraries for file handling and hashing

## Project Structure

- `data/` — Dataset storage or instructions for accessing the dataset.
- `src/` — Source code for firmware analysis and anomaly detection.
- `tests/` — Tests for the implemented functions.

## Development Status

Initial project setup is in progress. Implementation and testing will be documented as development proceeds.

## Important Note

An anomaly detection result indicates that a firmware sample differs from the learned baseline. It does not, by itself, prove that the firmware is malicious.