import glob
import os
import pandas as pd

STEP = 10   # keep every 10th row (about 25 Hz instead of 250 Hz)

files = glob.glob("Merged/**/*.csv", recursive=True)
frames = []

for i, f in enumerate(files, 1):
    parts = f.split(os.sep)
    df = pd.read_csv(f)
    df = df.iloc[::STEP].copy()
    df["condition"] = parts[1].lower()      # curved / random / straight
    df["flight_type"] = parts[2].lower()    # attacked / normal
    df["flight_id"] = f                     # one id per flight file
    frames.append(df)
    if i % 50 == 0:
        print("Loaded", i, "of", len(files), "files")

data = pd.concat(frames, ignore_index=True)

# ---- Cleaning ----
print("\nRows before cleaning:", data.shape[0])
print("Missing values:", int(data.isna().sum().sum()))
print("Duplicate rows:", int(data.duplicated().sum()))

data = data.drop_duplicates()
data = data.replace([float("inf"), float("-inf")], float("nan"))
data = data.dropna()
data["attack_enabled"] = data["attack_enabled"].astype(int)
print("Rows after cleaning:", data.shape[0])

# ---- Summary (take screenshots of this for your report) ----
print("\nFlights per condition:")
print(data.groupby("condition")["flight_id"].nunique())

print("\nRows per condition and class (0 = normal, 1 = attack):")
print(pd.crosstab(data["condition"], data["attack_enabled"]))

print("\nOverall class balance:")
print(data["attack_enabled"].value_counts(normalize=True).round(3))

data.to_pickle("combined.pkl")
print("\nSaved combined.pkl. Done!")