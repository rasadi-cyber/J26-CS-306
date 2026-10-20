"""Per-packet protocol fields for ISOT, row-aligned with data/isot/isot_packets.csv (feature check P15b).

Reads every recording in data/isot/recordings.json order (= the order isot_to_packets.py wrote the packets), so
row i here is row i of isot_packets.csv. frame.len is re-read and compared with the packet table as an alignment
check. Fields are behaviour-level header flags (Kitsune / CICIoT2023 style), no addresses, no counters, no clocks:
  is_ip, is_arp, is_tcp, is_udp, is_icmp, syn, ack, fin, rst, psh, urg, payload_len, web_port (80/443/8080)

  .venv/bin/python tools/isot_proto_extract.py [processes]
Output: data/isot/isot_proto.npy (float32, rows x 13) + isot_proto_cols.json
"""
import json, os, sys
from multiprocessing import Pool

import dpkt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.expanduser("~/Desktop/datasets/ISOT Drone Dataset/Raw Data")
COLS = ["is_ip", "is_arp", "is_tcp", "is_udp", "is_icmp", "syn", "ack", "fin", "rst", "psh", "urg", "payload_len",
        "web_port"]
WEB = {80, 443, 8080}


def parse(buf):
    f = [0.0] * len(COLS)
    try:
        eth = dpkt.ethernet.Ethernet(buf)
    except Exception:
        return f
    if isinstance(eth.data, dpkt.arp.ARP):
        f[1] = 1.0
        return f
    ip = eth.data
    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
        return f
    f[0] = 1.0
    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        fl = l4.flags
        f[2] = 1.0
        f[5], f[6], f[7] = float(bool(fl & dpkt.tcp.TH_SYN)), float(bool(fl & dpkt.tcp.TH_ACK)), float(bool(fl & dpkt.tcp.TH_FIN))
        f[8], f[9], f[10] = float(bool(fl & dpkt.tcp.TH_RST)), float(bool(fl & dpkt.tcp.TH_PUSH)), float(bool(fl & dpkt.tcp.TH_URG))
        f[11] = float(len(l4.data))
        f[12] = float(l4.sport in WEB or l4.dport in WEB)
    elif isinstance(l4, dpkt.udp.UDP):
        f[3] = 1.0
        f[11] = float(len(l4.data))
        f[12] = float(l4.sport in WEB or l4.dport in WEB)
    elif isinstance(l4, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
        f[4] = 1.0
        f[11] = float(len(bytes(l4.data)))
    return f


def one(job):
    path, limit = job
    out, lens = [], []
    with open(os.path.join(RAW, path), "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            out.append(parse(buf)); lens.append(len(buf))
            if len(out) >= limit:
                break
    return np.asarray(out, np.float32).reshape(-1, len(COLS)), np.asarray(lens, np.int32)


def main(procs):
    man = json.load(open(os.path.join(ROOT, "data", "isot", "recordings.json")))
    recs = man["recordings"]
    jobs = [(r["recording"], r["packets_kept"]) for r in recs]
    with Pool(procs) as p:
        res = p.map(one, jobs, chunksize=1)
    for r, (F, L) in zip(recs, res):
        assert len(F) == r["packets_kept"], (r["recording"], len(F), r["packets_kept"])
    F = np.vstack([a for a, _ in res]); L = np.concatenate([b for _, b in res])
    import pandas as pd
    ref = pd.read_csv(os.path.join(ROOT, "data", "isot", "isot_packets.csv"), usecols=["frame.len"], engine="pyarrow")["frame.len"].to_numpy()
    assert len(ref) == len(L), (len(ref), len(L))
    mism = int((ref != L).sum())
    print(f"rows {len(L):,}; frame.len mismatches vs isot_packets.csv: {mism}")
    assert mism == 0
    np.save(os.path.join(ROOT, "data", "isot", "isot_proto.npy"), F)
    json.dump(COLS, open(os.path.join(ROOT, "data", "isot", "isot_proto_cols.json"), "w"))
    print("per-column mean:", dict(zip(COLS, np.round(F.mean(0), 4).tolist())))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
