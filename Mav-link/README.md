# 🚁 UAV MAVLink Network Forensics & Attack Detection

**PCAP-to-CSV Conversion, Network Forensics, and Machine Learning-Based UAV Traffic Analysis**

## 📌 Project Overview

Unmanned Aerial Vehicles (UAVs) use communication protocols such as MAVLink to exchange telemetry and control messages between drones and Ground Control Stations (GCS).

Depending on the MAVLink version, configuration, and security mechanisms enabled, UAV communication may be exposed to threats such as unauthorized command injection and replay attacks.

This project focuses on **MAVLink network forensics and machine learning-based attack detection**. It provides a workflow for processing captured network traffic, extracting structured features, preparing datasets, training machine learning models, and analyzing UAV traffic for potential attacks.

The system is designed to support the investigation and classification of three traffic categories:

- **Normal (0):** Benign UAV communication.
- **Command Injection (1):** Traffic associated with unauthorized or injected commands.
- **Replay Attack (2):** Previously captured messages replayed into a communication stream.

The project is developed as a cybersecurity research component for UAV communication analysis and digital forensics.

## 🎯 Project Objectives

- Process captured UAV network traffic from PCAP files.
- Decode and extract relevant MAVLink message information.
- Convert packet data into structured CSV datasets.
- Calculate timing, sequence, and command-related features.
- Prepare labelled datasets for supervised machine learning.
- Train and compare Random Forest and XGBoost classifiers.
- Analyze packet-level predictions and potential attack events.
- Generate investigation results for further examination.

## ✨ Key Features

### 1. PCAP-to-CSV Conversion

The conversion component processes supported packet capture files, including `.pcap`, `.pcapng`, and `.cap`, and extracts structured packet information.

Key capabilities include:

- Packet capture loading and parsing.
- UDP traffic filtering based on supported configurations.
- MAVLink message decoding using the relevant parsing libraries.
- Extraction of message identifiers, sequence numbers, system IDs, component IDs, and command information.
- Export of extracted information to CSV format.

### 2. MAVLink Feature Extraction

The extracted features help describe the characteristics and behaviour of UAV communication.

| Feature | Description |
|---|---|
| `timestamp` | Capture timestamp associated with a packet. |
| `delta_time_ms` | Time difference between relevant packets in milliseconds. |
| `msg_id` | Numerical MAVLink message identifier. |
| `msg_name` | MAVLink message name, where available. |
| `seq` | MAVLink packet sequence counter. |
| `delta_seq` | Sequence difference calculated within the relevant stream. |
| `sys_id` | MAVLink source system identifier. |
| `comp_id` | MAVLink source component identifier. |
| `command_id` | MAVLink command identifier, where applicable. |
| `command_name` | Human-readable command name, where available. |
| `param1`–`param7` | Command parameters, where applicable. |
| `label` | Ground-truth class used for supervised learning. |

Feature availability depends on the capture contents and successfully decoded messages. Missing or inapplicable values should be handled during dataset preparation.

### 3. Dataset Preparation

The machine learning workflow uses structured CSV data containing extracted features and appropriate labels.

The dataset preparation process may include:

- Checking missing and invalid values.
- Verifying packet-level labels against trusted annotations or controlled attack-generation records.
- Reviewing class distributions.
- Separating training and testing data by capture file or flight/session.
- Applying class-balancing techniques to training data when justified.

**Important:** Dataset labels must be validated before they are used to train or evaluate a classifier. Synthetic or incorrectly assigned labels can lead to misleading results.

### 4. Machine Learning-Based Attack Detection

The project supports a workflow for training and evaluating supervised classification models.

**Random Forest**

Combines predictions from multiple decision trees to classify network traffic.

**XGBoost**

Builds boosted decision trees sequentially to improve the overall predictive model.

The models are intended to classify traffic into:

1. Normal
2. Command Injection
3. Replay Attack

Model performance should be evaluated using a held-out test dataset that is separated from the training data.

### 5. Forensic Analysis and Reporting

Depending on the implemented application functionality, the analysis workflow can support:

- Reviewing packet-level predictions.
- Identifying potential attack events.
- Examining relevant timestamps and command information.
- Reviewing file-level analysis results.
- Exporting prediction data and investigation reports.

A model prediction is an analytical result, not definitive proof that an attack occurred. Findings should be interpreted alongside packet evidence, dataset labels, and other forensic information.

## 🏗️ System Architecture

```text
          UAV Network Traffic
                   |
                   v
          PCAP / PCAPNG Files
                   |
                   v
        PCAP-to-CSV Conversion
                   |
                   v
       MAVLink Parsing and Decoding
                   |
                   v
          Feature Extraction
                   |
                   v
       Dataset Validation and Labels
                   |
                   v
          Train / Test Split
                   |
                   v
       Random Forest / XGBoost
                   |
                   v
          Model Evaluation
                   |
                   v
       CSV-Based Attack Detection
                   |
                   v
       Predictions and Reports
```

The workflow separates dataset preparation and model training from the subsequent analysis of input data.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scapy | Network packet processing |
| pymavlink | MAVLink message parsing |
| pandas | Structured data processing |
| NumPy | Numerical operations |
| scikit-learn | Machine learning and evaluation |
| XGBoost | Gradient-boosted classification |
| imbalanced-learn | Optional SMOTE-based class balancing |
| PyQt5 | Desktop graphical user interface, where used |
| Git and GitHub | Version control and source-code management |

## 💻 Installation

### Prerequisites

Before starting, install:

- Python compatible with the project's dependencies.
- Git.
- Visual Studio Code or another Python-compatible editor (optional).

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/DimuthuWork/UAV_mavlink-network-forensics.git
cd UAV_mavlink-network-forensics
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

If Windows cannot find Python, install Python and ensure it is available from the command line.

### 3. Install dependencies

If the repository contains a complete and up-to-date `requirements.txt`, use:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the required packages individually:

```bash
python -m pip install scapy pymavlink pandas numpy scikit-learn xgboost imbalanced-learn PyQt5
```

The required packages depend on the application components you intend to run.

## ▶️ Running the Application

The following commands should be used after confirming that the corresponding files exist in the repository.

### Launch the main application

```bash
python main.py
```

### Run the standalone ML detector, if included

```bash
python detector_app/main.py
```

### Run automated tests, if the corresponding test files are present

```bash
python test_pipeline.py
python test_gui.py
python -m unittest test_ml_pipeline
python -m unittest test_csv_attack_check
```

Some graphical tests may require a desktop display or a supported headless configuration.

### Typical analysis workflow

1. Open the PCAP-to-CSV conversion component.
2. Select a supported packet capture file.
3. Execute packet extraction and MAVLink parsing.
4. Export the structured CSV file.
5. Prepare and validate labelled CSV datasets for training.
6. Train a Random Forest or XGBoost model using the supported ML interface.
7. Load the trained model into the detector.
8. Submit an appropriate CSV file for prediction.
9. Review the predictions and available evaluation or reporting outputs.

**Input limitation:** If the detector accepts CSV files only, PCAP files must first be converted using the parsing component.

## 📊 Model Evaluation

Model performance should be assessed using a separate test dataset.

Recommended evaluation metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted members of a class that are correct.
- **Recall:** Proportion of actual members of a class that are detected.
- **F1-score:** Combined measure of precision and recall.
- **Macro-F1:** Average F1-score across the target classes.
- **Confusion Matrix:** Summary of correct and incorrect class predictions.

For a fair evaluation:

- Split data by capture file or flight/session where appropriate.
- Fit preprocessing operations using training data only.
- Apply SMOTE only to the training data, if needed.
- Prevent ground-truth labels from becoming model input features.
- Evaluate Normal, Injection, and Replay separately.
- Report measured results rather than assumed performance values.

Actual performance results should be added after running the evaluation on the selected dataset.

## 🔐 Security and Research Considerations

- Use authorized datasets and controlled test environments.
- Validate attack labels against reliable ground truth.
- Protect sensitive telemetry and captured network data.
- Treat model predictions as indicators requiring investigation.
- Document known limitations, false positives, and missed attacks.
- Avoid presenting synthetic test results as real-world validation.

The detection capability depends on the quality and diversity of the dataset, the selected features, the training methodology, and the evaluation environment.

## 🚧 Current Development and Future Improvements

Potential improvements include:

- Expanding validated UAV traffic datasets.
- Improving packet-level ground-truth validation.
- Evaluating additional temporal and behavioural features.
- Comparing model performance across independent flights.
- Improving detection visualizations and reporting.
- Testing the system on previously unseen capture files.
- Documenting reproducible experimental results.

Only completed and tested capabilities should be described as implemented features in project demonstrations and research reports.

## 👨‍💻 Author

**Ekanayaka E.M.D.B.**

B.Sc. (Hons) in Information Technology  
Specialization: Cyber Security  
Sri Lanka Institute of Information Technology (SLIIT)
