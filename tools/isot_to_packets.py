"""Convert the ISOT Drone Anomaly Detection captures (raw pcap) into one packet table in UAV-NIDD's column format,
so the same pipeline (size + timing features, time folds, mixed traffic) runs unchanged.

  .venv/bin/python tools/isot_to_packets.py [max_packets_per_recording]

Per packet: frame.number (1.. within its recording, so recordings are found like UAV-NIDD captures), frame.len
(bytes on the wire), frame.time_delta_displayed (gap to the previous packet of the same recording) and Label.
Rules (fixed before any result):
  - only Ethernet captures (link type 1); the de-authentication capture ("Password Cracking") is a Wi-Fi monitor
    capture (radiotap headers change every packet size), the same capture-setting leak removed from UAV-NIDD
  - the first N consecutive packets of each recording (used: 25,000 - the most that fits 26 GB RAM with good
    parallelism; user asked to keep the maximum rows);
    recordings with fewer packets are kept whole
  - recordings of one label are written in chronological order (first packet time), as a timeline
Labels: Normal, UDP Flood, TCP Flood, HTTP Flood (Hulk), SlowHTTPTest, Slowloris, and the non-DoS attacks under
their own names (merged into "Other attack" by the config).
"""
import csv, glob, json, os, sys
import dpkt

RAW = os.path.expanduser("~/Desktop/datasets/ISOT Drone Dataset/Raw Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "isot")
FOLDERS = {"Regular": "Normal", "MITM": "MITM", "Video": "Video interception", "IP Spoofing": "IP spoofing",
           "Manipulation": "Payload manipulation", "Replay": "Replay", "Unauth": "Unauthorized UDP",
           "Injection": "Instruction injection"}


def dos_label(name):
    n = name.lower()
    return ("UDP Flood" if n.startswith("udp") else "Slowloris" if n.startswith("slowloris") else
            "SlowHTTPTest" if n.startswith("slowhttp") else "HTTP Flood" if n.startswith("hulk") else "TCP Flood")


def read(path, limit):
    with open(path, "rb") as f:
        r = dpkt.pcap.Reader(f)
        if r.datalink() != dpkt.pcap.DLT_EN10MB:
            return None, []
        rows = []
        for ts, buf in r:
            rows.append((ts, len(buf)))
            if len(rows) >= limit:
                break
        return r.datalink(), rows


def main(limit):
    recs = [(os.path.join(RAW, d, f), lab) for d, lab in FOLDERS.items() for f in sorted(os.listdir(os.path.join(RAW, d)))
            if f.endswith(".pcap")]
    recs += [(p, dos_label(os.path.basename(p))) for p in sorted(glob.glob(os.path.join(RAW, "DoS", "*.pcap")))]
    loaded = []
    for path, lab in recs:
        lt, rows = read(path, limit)
        if not rows:
            print("skipped (not Ethernet or empty):", path); continue
        loaded.append((lab, rows[0][0], path, rows))
    loaded.sort(key=lambda x: (x[0], x[1]))  # per label, chronological
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    with open(os.path.join(OUT, "isot_packets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame.number", "frame.len", "frame.time_delta_displayed", "Label"])
        # frame.number is unique across the whole table (recording r starts at r * 100,000,000 + 1; was 1,000,000, which let recordings over 1 M packets overlap the next and lose 55,379 packets to the duplicate rule): packet numbers
        # still continue +1 inside a recording (so recordings are found as captures) and jump between recordings,
        # and the true-duplicate rule (same label, number, size, gap) cannot match packets of different recordings.
        for r_i, (lab, t0, path, rows) in enumerate(loaded):
            prev = rows[0][0]
            for i, (ts, ln) in enumerate(rows, 1):
                w.writerow([r_i * 100_000_000 + i, ln, f"{max(ts - prev, 0.0):.9f}", lab]); prev = ts
            manifest.append({"recording": os.path.relpath(path, RAW), "label": lab, "packets_kept": len(rows),
                             "start_time": t0})
    with open(os.path.join(OUT, "recordings.json"), "w") as f:
        json.dump({"max_packets_per_recording": limit, "recordings": manifest}, f, indent=1)
    by = {}
    for m in manifest:
        by.setdefault(m["label"], [0, 0]); by[m["label"]][0] += m["packets_kept"]; by[m["label"]][1] += 1
    for k, (n, c) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"{k:22s} {n:9,d} packets from {c} recordings")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
