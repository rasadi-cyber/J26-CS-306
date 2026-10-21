import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import joblib
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gps_common import prepare_flight

BUNDLE = "gps_detector_unseen.pkl"

NOTE = ("Unseen-condition demonstration: the model was trained only on Straight "
        "and Curved flights, and Random flights were never used for training. The "
        "alarm line was set from 20 normal Random flights. Simulated PX4-SITL data, "
        "exploratory features, prototype, limited reliability.")


class App:
    def __init__(self, root):
        self.root = root
        root.title("GPS Spoofing Detection - Unseen Flight Condition Demo")
        root.geometry("920x740")
        self.bundle = joblib.load(BUNDLE)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)
        self.path = tk.StringVar()
        ttk.Button(top, text="Choose flight CSV...", command=self.choose).pack(side="left")
        ttk.Label(top, textvariable=self.path).pack(side="left", padx=8)

        row = ttk.Frame(root)
        row.pack(fill="x", padx=10)
        ttk.Label(row, text="Flight condition:").pack(side="left")
        self.cond = tk.StringVar(value="random")
        ttk.Combobox(row, textvariable=self.cond, state="readonly", width=12,
                     values=["random"]).pack(side="left", padx=6)
        ttk.Button(row, text="Analyse", command=self.analyse).pack(side="left", padx=6)

        self.verdict = tk.Label(root, text="No flight analysed yet",
                                font=("Segoe UI", 20, "bold"), pady=10)
        self.verdict.pack(fill="x")
        self.details = tk.Label(root, text="", justify="left", font=("Segoe UI", 11))
        self.details.pack(anchor="w", padx=12)

        self.fig = Figure(figsize=(8.5, 3.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

        tk.Label(root, text=NOTE, wraplength=860, justify="left", anchor="w",
                 fg="#555", font=("Segoe UI", 9)).pack(fill="x", padx=12, pady=6)

    def choose(self):
        p = filedialog.askopenfilename(
            initialdir="demo_flights_unseen" if os.path.isdir("demo_flights_unseen") else ".",
            filetypes=[("CSV files", "*.csv")])
        if p:
            self.path.set(p)

    def analyse(self):
        p = self.path.get()
        if not p:
            messagebox.showinfo("Choose a file", "Choose a flight CSV first.")
            return
        try:
            df = prepare_flight(pd.read_csv(p))
            if len(df) < 10:
                raise ValueError("Flight is too short after preprocessing.")
            prob = self.bundle["model"].predict_proba(df[self.bundle["features"]])[:, 1]
        except Exception as e:
            messagebox.showerror("Could not analyse this file", str(e))
            return

        score = float(prob.mean())
        thr = self.bundle["thresholds"]["random"]
        detected = score >= thr

        if detected:
            self.verdict.config(text="GPS SPOOFING DETECTED", fg="white", bg="#c0392b")
        else:
            self.verdict.config(text="NORMAL FLIGHT (no spoofing flagged)", fg="white", bg="#27ae60")
        self.details.config(text=(
            f"File: {os.path.basename(p)}\n"
            f"Flight condition: random (never seen in training)\n"
            f"Flight score: {score:.3f}    Threshold: {thr:.3f}\n"
            f"Mean position-vs-velocity residual: {df['res_mag'].mean():.3f}\n"
            f"Rows analysed: {len(df)}"))

        t = (df["time_usec"] - df["time_usec"].iloc[0]) / 1e6
        smooth = pd.Series(prob).rolling(25, min_periods=1).mean().values
        self.ax.clear()
        self.ax.plot(t.values, smooth, label="Attack probability (1 s average)")
        self.ax.axhline(score, color="blue", linestyle=":", label="Flight score")
        self.ax.axhline(thr, color="red", linestyle="--", label="Alarm threshold")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Probability")
        self.ax.set_ylim(0, 1)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw()


if not os.path.exists(BUNDLE):
    print("Run step19.py first to create", BUNDLE)
else:
    root = tk.Tk()
    App(root)
    root.mainloop()