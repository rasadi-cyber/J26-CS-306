"""
MAVLink Protocol Decapsulation & Stateful Feature Extraction Engine
Includes Per-Stream (sys_id, comp_id) Sequence Tracking, BAD_DATA Filtering,
and Packet-Level Ground-Truth Labeling.
"""

import os
import time
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd

from scapy.all import PcapReader, IP, IPv6, UDP, Raw
from pymavlink import mavutil

# Human-readable fallback map for common MAV_CMD IDs
COMMON_MAV_CMDS: Dict[int, str] = {
    16: "MAV_CMD_NAV_WAYPOINT",
    17: "MAV_CMD_NAV_LOITER_UNLIM",
    18: "MAV_CMD_NAV_LOITER_TURNS",
    19: "MAV_CMD_NAV_LOITER_TIME",
    20: "MAV_CMD_NAV_RETURN_TO_LAUNCH",
    21: "MAV_CMD_NAV_LAND",
    22: "MAV_CMD_NAV_TAKEOFF",
    23: "MAV_CMD_NAV_LAND_LOCAL",
    24: "MAV_CMD_NAV_TAKEOFF_LOCAL",
    30: "MAV_CMD_NAV_CONTINUE_AND_CHANGE_ALT",
    31: "MAV_CMD_NAV_LOITER_TO_ALT",
    84: "MAV_CMD_NAV_VTOL_TAKEOFF",
    85: "MAV_CMD_NAV_VTOL_LAND",
    92: "MAV_CMD_NAV_GUIDED_ENABLE",
    176: "MAV_CMD_DO_SET_MODE",
    177: "MAV_CMD_DO_JUMP",
    178: "MAV_CMD_DO_CHANGE_SPEED",
    179: "MAV_CMD_DO_SET_HOME",
    180: "MAV_CMD_DO_SET_PARAMETER",
    181: "MAV_CMD_DO_SET_RELAY",
    182: "MAV_CMD_DO_REPEAT_RELAY",
    183: "MAV_CMD_DO_SET_SERVO",
    184: "MAV_CMD_DO_REPEAT_SERVO",
    192: "MAV_CMD_DO_FLIGHTTERMINATION",
    200: "MAV_CMD_DO_CONTROL_VIDEO",
    201: "MAV_CMD_DO_SET_ROI",
    203: "MAV_CMD_DO_DIGICAM_CONTROL",
    204: "MAV_CMD_DO_MOUNT_CONTROL",
    241: "MAV_CMD_PREFLIGHT_STORAGE",
    242: "MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN",
    246: "MAV_CMD_PREFLIGHT_CALIBRATION",
    400: "MAV_CMD_COMPONENT_ARM_DISARM",
    511: "MAV_CMD_REQUEST_MESSAGE",
    520: "MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES",
    2500: "MAV_CMD_DO_CHANGE_SPEED",
    3000: "MAV_CMD_START_RX_PAIR",
}

COMMAND_MSG_IDS = {76, 75, 73, 39, 77, 11}  # COMMAND_LONG, COMMAND_INT, MISSION_ITEM_INT, MISSION_ITEM, COMMAND_ACK, SET_MODE


def resolve_command_name(command_id: int) -> str:
    """Resolve numerical MAV_CMD ID to human-readable string representation."""
    if command_id <= 0:
        return ""
    if command_id in COMMON_MAV_CMDS:
        return COMMON_MAV_CMDS[command_id]

    try:
        if hasattr(mavutil.mavlink, "enums") and "MAV_CMD" in mavutil.mavlink.enums:
            entry = mavutil.mavlink.enums["MAV_CMD"].get(command_id)
            if entry and hasattr(entry, "name"):
                return entry.name
    except Exception:
        pass

    return f"MAV_CMD_{command_id}"


@dataclass
class ParsedRecord:
    """Standardized parsed MAVLink frame record with engineered features."""
    timestamp: float          # Raw packet timestamp (epoch seconds)
    delta_time_ms: float      # Inter-packet time delta in milliseconds (per stream)
    msg_id: int               # MAVLink message ID
    msg_name: str             # Human-readable message name (e.g., HEARTBEAT, COMMAND_LONG)
    seq: int                  # Packet sequence number (0-255)
    delta_seq: int            # Sequence delta per stream: (seq_t - seq_{t-1}) % 256
    sys_id: int               # Source System ID
    comp_id: int              # Source Component ID
    command_id: int           # Operational command ID (0 if not command)
    command_name: str         # Human-readable command name (e.g., MAV_CMD_NAV_LAND)
    param1: float             # Operational parameter 1
    param2: float             # Operational parameter 2
    param3: float             # Operational parameter 3
    param4: float             # Operational parameter 4
    param5: float             # Operational parameter 5
    param6: float             # Operational parameter 6
    param7: float             # Operational parameter 7
    label: int                # Target ML ground truth class (0: Benign, 1: Injection, 2: Replay)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MAVLinkFeatureExtractor:
    """
    Stateful feature extractor tracking per-stream (sys_id, comp_id) sequence and timing dynamics.
    Supports packet-level ground truth labeling and bad data rejection.
    """

    def __init__(self, target_label: Any = "auto", filter_bad_data: bool = True):
        self.target_label = self._normalize_target_label(target_label)
        self.filter_bad_data = filter_bad_data
        # Separate sequence and timing dynamics per distinct (sys_id, comp_id) stream
        self.stream_last_seq: Dict[Tuple[int, int], int] = {}
        self.stream_last_timestamp: Dict[Tuple[int, int], float] = {}
        self.total_frames_processed = 0

    @staticmethod
    def _normalize_target_label(target_label: Any) -> Any:
        if target_label in ("auto", None, -1, -2, 10, 100) or str(target_label).strip().lower() == "auto":
            return "auto"
        if target_label in (0, 1, 2, 99):
            return target_label
        try:
            v = int(target_label)
            return v if v in (0, 1, 2, 99) else "auto"
        except Exception:
            return "auto"

    def reset(self, target_label: Optional[Any] = None):
        """Reset internal state for a new capture session."""
        if target_label is not None:
            self.target_label = self._normalize_target_label(target_label)
        self.stream_last_seq.clear()
        self.stream_last_timestamp.clear()
        self.total_frames_processed = 0

    @staticmethod
    def is_bad_frame(msg: Any) -> bool:
        """
        Detect corrupt, incomplete, or framing error packets.
        Specifically filters out BAD_DATA and sys_id==0/comp_id==0 garbage frames.
        """
        if msg is None:
            return True
        msg_type = getattr(msg, "get_type", lambda: "")()
        if msg_type == "BAD_DATA":
            return True
        msg_id = getattr(msg, "get_msgId", lambda: -1)()
        if msg_id < 0:
            return True
        sys_id = getattr(msg, "get_srcSystem", lambda: 0)()
        comp_id = getattr(msg, "get_srcComponent", lambda: 0)()
        if sys_id == 0 and comp_id == 0:
            return True
        return False

    def process_message(
        self,
        msg: Any,
        packet_timestamp: float,
        packet_label: Optional[int] = None,
    ) -> Optional[ParsedRecord]:
        """
        Extract attributes, compute per-stream sequence and timing dynamics,
        and construct a ParsedRecord with packet-level ground-truth labeling.
        """
        # Filter bad data if enabled
        if self.filter_bad_data and self.is_bad_frame(msg):
            return None

        # 1. Message IDs & Stream Identification
        msg_id = msg.get_msgId()
        msg_name = msg.get_type()
        sys_id = msg.get_srcSystem()
        comp_id = msg.get_srcComponent()
        stream_key = (sys_id, comp_id)

        # 2. Timing Delta (Δt in ms) per stream
        if stream_key not in self.stream_last_timestamp:
            delta_time_ms = 0.0
        else:
            delta_time_ms = max(0.0, (packet_timestamp - self.stream_last_timestamp[stream_key]) * 1000.0)
        self.stream_last_timestamp[stream_key] = packet_timestamp

        # 3. Sequence info and Per-Stream Delta (ΔSeq % 256)
        seq = getattr(msg, "get_seq", lambda: getattr(msg, "_seq", 0))()
        if stream_key not in self.stream_last_seq:
            delta_seq = 1  # Standard baseline initial delta for new stream
        else:
            delta_seq = (seq - self.stream_last_seq[stream_key]) % 256
        self.stream_last_seq[stream_key] = seq

        # 4. Command Semantics & Parameter Extraction
        command_id = 0
        command_name = ""
        p1, p2, p3, p4, p5, p6, p7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if msg_name == "COMMAND_LONG":
            command_id = int(getattr(msg, "command", 0))
            command_name = resolve_command_name(command_id)
            p1 = float(getattr(msg, "param1", 0.0))
            p2 = float(getattr(msg, "param2", 0.0))
            p3 = float(getattr(msg, "param3", 0.0))
            p4 = float(getattr(msg, "param4", 0.0))
            p5 = float(getattr(msg, "param5", 0.0))
            p6 = float(getattr(msg, "param6", 0.0))
            p7 = float(getattr(msg, "param7", 0.0))

        elif msg_name == "COMMAND_INT":
            command_id = int(getattr(msg, "command", 0))
            command_name = resolve_command_name(command_id)
            p1 = float(getattr(msg, "param1", 0.0))
            p2 = float(getattr(msg, "param2", 0.0))
            p3 = float(getattr(msg, "param3", 0.0))
            p4 = float(getattr(msg, "param4", 0.0))
            p5 = float(getattr(msg, "x", 0.0))
            p6 = float(getattr(msg, "y", 0.0))
            p7 = float(getattr(msg, "z", 0.0))

        elif msg_name in ("MISSION_ITEM", "MISSION_ITEM_INT"):
            command_id = int(getattr(msg, "command", 0))
            command_name = resolve_command_name(command_id)
            p1 = float(getattr(msg, "param1", 0.0))
            p2 = float(getattr(msg, "param2", 0.0))
            p3 = float(getattr(msg, "param3", 0.0))
            p4 = float(getattr(msg, "param4", 0.0))
            p5 = float(getattr(msg, "x", getattr(msg, "param5", 0.0)))
            p6 = float(getattr(msg, "y", getattr(msg, "param6", 0.0)))
            p7 = float(getattr(msg, "z", getattr(msg, "param7", 0.0)))

        elif msg_name == "SET_MODE":
            command_id = 176  # MAV_CMD_DO_SET_MODE
            command_name = "MAV_CMD_DO_SET_MODE"
            p1 = float(getattr(msg, "custom_mode", 0.0))
            p2 = float(getattr(msg, "base_mode", 0.0))

        elif hasattr(msg, "command"):
            command_id = int(getattr(msg, "command", 0))
            command_name = resolve_command_name(command_id)
            p1 = float(getattr(msg, "param1", 0.0)) if hasattr(msg, "param1") else 0.0
            p2 = float(getattr(msg, "param2", 0.0)) if hasattr(msg, "param2") else 0.0

        # 5. Intelligent Packet-Level Ground-Truth Label Assignment
        final_label = 0
        if packet_label is not None and packet_label in (0, 1, 2):
            # Explicit ground-truth provided (from sidecar or embedded packet marker)
            final_label = int(packet_label)
        elif self.target_label in ("auto", None, -1, -2, 10, 100) or str(self.target_label).strip().lower() == "auto":
            # Automatic packet-level detection based on forensic indicators
            is_malicious_cmd = (command_id in (21, 400, 176, 192, 242) and delta_seq > 5) or (
                msg_name in ("COMMAND_LONG", "COMMAND_INT") and delta_seq > 10
            )
            is_replay_burst = (delta_seq > 180 or (delta_seq != 1 and delta_seq != 0 and delta_time_ms < 5.0))
            if is_malicious_cmd:
                final_label = 1
            elif is_replay_burst:
                final_label = 2
            else:
                final_label = 0
        elif self.target_label == 0:
            # Benign capture: all packets 0
            final_label = 0
        elif self.target_label == 1:
            # Command Injection Scenario:
            # Only injected commands or sequence-jumping command bursts are labeled 1.
            # Routine telemetry (HEARTBEAT, ATTITUDE, etc.) remains 0!
            is_injected = (
                command_id > 0
                or msg_id in COMMAND_MSG_IDS
                or (delta_seq > 10 and msg_name not in ("HEARTBEAT", "TIMESYNC"))
            )
            final_label = 1 if is_injected else 0
        elif self.target_label == 2:
            # Replay Scenario:
            # Only packets with sequence rollback or replay anomalies are labeled 2.
            is_replayed = (delta_seq != 1 and delta_seq != 0) or (delta_seq == 0 and delta_time_ms < 2.0)
            final_label = 2 if is_replayed else 0
        elif self.target_label == 99:
            # Explicit whole-file override (for testing only)
            final_label = 1
        else:
            try:
                v = int(self.target_label)
                final_label = v if v in (0, 1, 2) else 0
            except Exception:
                final_label = 0

        self.total_frames_processed += 1

        return ParsedRecord(
            timestamp=round(packet_timestamp, 6),
            delta_time_ms=round(delta_time_ms, 3),
            msg_id=msg_id,
            msg_name=msg_name,
            seq=seq,
            delta_seq=delta_seq,
            sys_id=sys_id,
            comp_id=comp_id,
            command_id=command_id,
            command_name=command_name,
            param1=round(p1, 6),
            param2=round(p2, 6),
            param3=round(p3, 6),
            param4=round(p4, 6),
            param5=round(p5, 6),
            param6=round(p6, 6),
            param7=round(p7, 6),
            label=final_label,
        )


def open_capture_reader(filepath: str):
    """
    Open PCAP or PCAPNG file with automatic magic-byte detection and fallback.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Capture file not found: {filepath}")

    magic = b""
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
    except Exception:
        pass

    is_pcapng = (magic == b"\x0a\x0d\x0d\x0a") or filepath.lower().endswith((".pcapng", ".capng", ".ntar"))

    if is_pcapng:
        try:
            from scapy.utils import PcapNgReader
            return PcapNgReader(filepath)
        except Exception:
            pass

    try:
        return PcapReader(filepath)
    except Exception:
        from scapy.utils import PcapNgReader
        return PcapNgReader(filepath)


def load_sidecar_ground_truth(pcap_filepath: str) -> Optional[Dict[int, int]]:
    """
    Attempt to load ground truth packet labels from co-located sidecar file.
    Checks:
      <name>_labels.json
      <name>_ground_truth.json
      <name>_ground_truth.csv
    """
    base, _ = os.path.splitext(pcap_filepath)
    candidates = [
        f"{base}_labels.json",
        f"{base}_ground_truth.json",
        f"{pcap_filepath}.labels.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert keys to int if necessary
                    return {int(k): int(v) for k, v in data.items()}
            except Exception:
                pass

    # Check CSV candidate
    csv_candidates = [
        f"{base}_ground_truth.csv",
        f"{base}_labels.csv",
    ]
    for c in csv_candidates:
        if os.path.exists(c):
            try:
                df_gt = pd.read_csv(c)
                if "packet_idx" in df_gt.columns and "label" in df_gt.columns:
                    return dict(zip(df_gt["packet_idx"].astype(int), df_gt["label"].astype(int)))
            except Exception:
                pass

    return None


class MAVLinkPCAPParser:
    """
    Decapsulates Ethernet/IP/UDP datagrams from .pcap / .pcapng captures,
    filters MAVLink ports (14550, 14551), eliminates corrupt BAD_DATA / 0/0 frames,
    and applies stateful per-stream feature extraction with packet-level ground-truth labeling.
    """

    def __init__(
        self,
        target_ports: Tuple[int, ...] = (14550, 14551),
        commands_only: bool = False,
        target_label: Any = "auto",
        filter_bad_data: bool = True,
        use_ground_truth: bool = True,
        include_packet_metadata: bool = False,
    ):
        self.target_ports = target_ports
        self.commands_only = commands_only
        self.target_label = target_label
        self.filter_bad_data = filter_bad_data
        self.use_ground_truth = use_ground_truth
        self.include_packet_metadata = include_packet_metadata
        self.total_packets_seen = 0
        self.extractor = MAVLinkFeatureExtractor(
            target_label=target_label,
            filter_bad_data=filter_bad_data,
        )

    @staticmethod
    def get_file_metadata(filepath: str) -> Dict[str, Any]:
        """Calculates file size and SHA-256 checksum for forensic chain of custody."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        file_size = os.path.getsize(filepath)
        sha256_hash = hashlib.sha256()

        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        return {
            "filename": os.path.basename(filepath),
            "filepath": os.path.abspath(filepath),
            "filesize_bytes": file_size,
            "filesize_str": f"{file_size / (1024 * 1024):.2f} MB" if file_size >= 1024 * 1024 else f"{file_size / 1024:.2f} KB",
            "sha256": sha256_hash.hexdigest(),
        }

    def parse_pcap(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> pd.DataFrame:
        """
        Streaming parse of PCAP/PCAPNG file.
        Returns a pandas DataFrame ready for ML export.
        """
        self.extractor.reset(target_label=self.target_label)
        mav_decoder = mavutil.mavlink.MAVLink(None)
        mav_decoder.robust_parsing = True

        # Load sidecar ground truth if available
        sidecar_labels = load_sidecar_ground_truth(filepath) if self.use_ground_truth else None
        if sidecar_labels and log_callback:
            log_callback(f"[INFO] Detected ground-truth sidecar with {len(sidecar_labels):,} packet labels.")

        records: List[Dict[str, Any]] = []
        total_packets = 0
        mavlink_valid_frames = 0
        dropped_non_udp = 0
        corrupted_payloads = 0
        commands_count = 0
        anomalies_count = 0

        if log_callback:
            log_callback(f"[INFO] Ingesting capture: {os.path.basename(filepath)}")
            log_callback(f"[INFO] Filtering UDP ports: {self.target_ports if self.target_ports else 'ALL'}")
            log_callback(f"[INFO] Filter BAD_DATA frames: {self.filter_bad_data}")
            log_callback(f"[INFO] Ground-truth mode: {self.target_label}")

        try:
            reader = open_capture_reader(filepath)
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to open PCAP/PCAPNG reader: {str(e)}")
            raise

        try:
            for pkt in reader:
                if is_cancelled and is_cancelled():
                    if log_callback:
                        log_callback("[WARN] Parsing operation cancelled by user.")
                    break

                total_packets += 1
                pkt_index = total_packets - 1

                # Safe timestamp extraction
                pkt_time = time.time()
                try:
                    if hasattr(pkt, "time") and pkt.time is not None:
                        pkt_time = float(pkt.time)
                except Exception:
                    pass

                # Detect embedded ground truth label from IP TOS / DSCP or sidecar
                embedded_label = None
                if self.use_ground_truth and sidecar_labels and pkt_index in sidecar_labels:
                    embedded_label = sidecar_labels[pkt_index]
                elif self.use_ground_truth and pkt.haslayer(IP):
                    # Check IP.tos (0: benign, 1: injection, 2: replay)
                    tos = getattr(pkt[IP], "tos", 0)
                    if tos in (1, 2):
                        embedded_label = tos

                raw_data = None

                # Extract payload from UDP layer if present
                if pkt.haslayer(UDP):
                    udp_layer = pkt[UDP]
                    sport = getattr(udp_layer, "sport", 0)
                    dport = getattr(udp_layer, "dport", 0)

                    if self.target_ports:
                        if sport not in self.target_ports and dport not in self.target_ports:
                            dropped_non_udp += 1
                            continue

                    try:
                        raw_data = bytes(udp_layer.payload)
                    except Exception:
                        raw_data = None
                else:
                    dropped_non_udp += 1
                    try:
                        p_bytes = bytes(pkt)
                        if b"\xfd" in p_bytes or b"\xfe" in p_bytes:
                            raw_data = p_bytes
                    except Exception:
                        raw_data = None

                if not raw_data:
                    if total_packets % 50 == 0 and progress_callback:
                        stats = {
                            "total_packets": total_packets,
                            "mavlink_valid": mavlink_valid_frames,
                            "commands_count": commands_count,
                            "dropped_non_udp": dropped_non_udp,
                            "corrupted": corrupted_payloads,
                            "anomalies_count": anomalies_count,
                        }
                        progress_callback(total_packets, len(records), stats)
                    continue

                # Decode MAVLink frames from raw payload
                try:
                    msgs = mav_decoder.parse_buffer(raw_data)
                except Exception:
                    corrupted_payloads += 1
                    continue

                if not msgs:
                    if total_packets % 50 == 0 and progress_callback:
                        stats = {
                            "total_packets": total_packets,
                            "mavlink_valid": mavlink_valid_frames,
                            "commands_count": commands_count,
                            "dropped_non_udp": dropped_non_udp,
                            "corrupted": corrupted_payloads,
                            "anomalies_count": anomalies_count,
                        }
                        progress_callback(total_packets, len(records), stats)
                    continue

                for msg in msgs:
                    if msg is None:
                        continue

                    # Filter BAD_DATA frames
                    if self.filter_bad_data and MAVLinkFeatureExtractor.is_bad_frame(msg):
                        corrupted_payloads += 1
                        continue

                    record = self.extractor.process_message(
                        msg=msg,
                        packet_timestamp=pkt_time,
                        packet_label=embedded_label,
                    )

                    if record is None:
                        corrupted_payloads += 1
                        continue

                    mavlink_valid_frames += 1

                    if record.delta_seq != 1:
                        anomalies_count += 1

                    is_cmd = (
                        record.command_id > 0
                        or record.msg_id in COMMAND_MSG_IDS
                    )

                    if is_cmd:
                        commands_count += 1

                    if self.commands_only and not is_cmd:
                        continue

                    record_data = record.to_dict()
                    if self.include_packet_metadata:
                        if pkt.haslayer(IP):
                            source_ip = str(pkt[IP].src)
                            destination_ip = str(pkt[IP].dst)
                        elif pkt.haslayer(IPv6):
                            source_ip = str(pkt[IPv6].src)
                            destination_ip = str(pkt[IPv6].dst)
                        else:
                            source_ip = destination_ip = ""
                        if pkt.haslayer(UDP):
                            source_port = int(pkt[UDP].sport)
                            destination_port = int(pkt[UDP].dport)
                            record_data["_src_addr"] = (
                                f"{source_ip}:{source_port}" if source_ip else f":{source_port}"
                            )
                            record_data["_dst_addr"] = (
                                f"{destination_ip}:{destination_port}" if destination_ip else f":{destination_port}"
                            )
                        else:
                            record_data["_src_addr"] = ""
                            record_data["_dst_addr"] = ""
                        record_data["_packet_index"] = pkt_index
                    records.append(record_data)

                if (total_packets <= 100 or total_packets % 25 == 0) and progress_callback:
                    stats = {
                        "total_packets": total_packets,
                        "mavlink_valid": mavlink_valid_frames,
                        "commands_count": commands_count,
                        "dropped_non_udp": dropped_non_udp,
                        "corrupted": corrupted_payloads,
                        "anomalies_count": anomalies_count,
                    }
                    progress_callback(total_packets, len(records), stats)

        finally:
            try:
                reader.close()
            except Exception:
                pass

        self.total_packets_seen = total_packets
        if log_callback:
            log_callback(
                f"[COMPLETE] Processed {total_packets:,} packets. "
                f"Decoded {mavlink_valid_frames:,} valid frames ({commands_count:,} commands). "
                f"Filtered {corrupted_payloads:,} BAD_DATA / corrupted frames. "
                f"Retained {len(records):,} records for dataset."
            )

        columns = [
            "timestamp", "delta_time_ms", "msg_id", "msg_name",
            "seq", "delta_seq", "sys_id", "comp_id",
            "command_id", "command_name",
            "param1", "param2", "param3", "param4", "param5", "param6", "param7",
            "label"
        ]

        if not records:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(records)
        return df[columns]
