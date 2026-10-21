import glob
import os
import pandas as pd

files = glob.glob("Merged/**/*.csv", recursive=True)
print("Total files:", len(files))

# Count flights per condition and per Attacked/Normal
counts = {}
for f in files:
    parts = f.split(os.sep)
    key = (parts[1], parts[2])
    counts[key] = counts.get(key, 0) + 1

print("\nFlights per condition:")
for k, v in sorted(counts.items()):
    print(k, v)

# Look inside one file
df = pd.read_csv(files[0])
print("\nRows and columns:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df.head(3).T)

label_cols = [c for c in df.columns if "attack" in c.lower()]
print("\nLabel column(s):", label_cols)
for c in label_cols:
    print(df[c].value_counts())