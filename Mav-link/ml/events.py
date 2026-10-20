"""Turn packet-level attack predictions into concise, file-level attack events."""

from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from .features import CLASS_MAP


def _most_common_text(values):
    cleaned = [str(value) for value in values if pd.notna(value) and str(value)]
    return Counter(cleaned).most_common(1)[0][0] if cleaned else ""


def build_attack_events(predictions: pd.DataFrame, max_gap_s=2.0, minimum_packets=3):
    """Group predicted attack frames by class and timestamp; omit undersized groups."""
    if predictions.empty or "pred" not in predictions:
        return []

    ordered = predictions.sort_values("timestamp", kind="mergesort").copy()
    ordered["timestamp"] = pd.to_numeric(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"])
    capture_start = float(ordered["timestamp"].min()) if not ordered.empty else 0.0
    groups = []
    current = []
    current_class = None
    previous_time = None

    for _, row in ordered.iterrows():
        prediction = int(row["pred"])
        timestamp = float(row["timestamp"])
        is_attack = prediction in (1, 2)
        if (
            not is_attack
            or prediction != current_class
            or (previous_time is not None and timestamp - previous_time > max_gap_s)
        ):
            if current:
                groups.append((current_class, current))
            current = []
            current_class = prediction if is_attack else None
        if is_attack:
            current.append(row)
            current_class = prediction
            previous_time = timestamp
        else:
            previous_time = None
    if current:
        groups.append((current_class, current))

    events = []
    for event_id, (attack_class, rows) in enumerate(groups, start=1):
        if len(rows) < minimum_packets:
            continue
        event = pd.DataFrame(rows)
        times = pd.to_numeric(event["timestamp"], errors="coerce")
        start_epoch = float(times.min())
        end_epoch = float(times.max())
        duration = max(0.0, end_epoch - start_epoch)
        commands = []
        if "command_id" in event:
            names = []
            for _, row in event.iterrows():
                command_id = int(row.get("command_id", 0) or 0)
                if command_id:
                    name = str(row.get("command_name", "") or f"MAV_CMD_{command_id}")
                    names.append(name)
            commands = [f"{name} x{count}" for name, count in Counter(names).most_common()]

        sequence_jumps = pd.to_numeric(
            event.get("seq_jump", event.get("delta_seq", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0)
        max_jump = int(sequence_jumps.max()) if not sequence_jumps.empty else 0
        command_long_count = (
            int(event["msg_name"].astype(str).eq("COMMAND_LONG").sum())
            if "msg_name" in event
            else 0
        )
        primary_source = (
            Counter(zip(event["sys_id"].astype(int), event["comp_id"].astype(int))).most_common(1)[0][0]
            if "sys_id" in event and "comp_id" in event
            else (0, 0)
        )
        burst_rate = len(event) / max(duration, 0.001)
        reason = (
            f"{command_long_count} COMMAND_LONG packets with sequence jumps up to {max_jump} "
            f"within {duration:.1f} s from sys {primary_source[0]}/comp {primary_source[1]}, "
            f"burst-rate {burst_rate:.0f} pkt/s"
        )
        events.append(
            {
                "event_id": f"EVT-{len(events) + 1:04d}",
                "attack_type": CLASS_MAP[attack_class],
                "first_packet_no": int(event["packet_no"].min()) if "packet_no" in event else None,
                "last_packet_no": int(event["packet_no"].max()) if "packet_no" in event else None,
                "start_time_relative_s": round(start_epoch - capture_start, 6),
                "end_time_relative_s": round(end_epoch - capture_start, 6),
                "start_time_utc": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
                "end_time_utc": datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
                "duration_s": round(duration, 6),
                "packet_count": len(event),
                "avg_confidence": float(pd.to_numeric(event["confidence"]).mean())
                if "confidence" in event
                else 0.0,
                "max_confidence": float(pd.to_numeric(event["confidence"]).max())
                if "confidence" in event
                else 0.0,
                "source_sys_id": primary_source[0],
                "source_comp_id": primary_source[1],
                "target_commands": commands,
                "max_seq_jump": max_jump,
                "src_ip_port": _most_common_text(event.get("_src_addr", [])),
                "dst_ip_port": _most_common_text(event.get("_dst_addr", [])),
                "reason": reason,
            }
        )
    return events
