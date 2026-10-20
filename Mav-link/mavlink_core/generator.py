"""
Synthetic UAV PCAP Generator & Real-Flight Attack Injector
Generates realistic UAV traffic and injects authentic cyber-attacks (Command Injection, Replay)
directly into real flight captures with packet-level ground-truth labeling.
"""

import os
import json
import time
import struct
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from scapy.all import Ether, IP, UDP, Raw, wrpcap, Packet, PcapReader
from pymavlink import mavutil


class SyntheticPCAPGenerator:
    """
    Synthesizes multi-stream UAV and GCS network traffic with authentic sensor distributions
    and packet-level ground truth tags.
    """

    @staticmethod
    def create_uav_sample_pcap(
        output_filepath: str,
        num_packets: int = 1200,
        attack_scenario: str = "mixed",  # "benign", "injection", "replay", "mixed"
        base_time: Optional[float] = 1617778618.0,  # Matches authentic flight timeline (April 2021)
    ) -> str:
        """
        Synthesize Ethernet -> IP -> UDP -> MAVLink packet capture with packet-level ground truth.
        """
        mav_uav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        mav_gcs = mavutil.mavlink.MAVLink(None, srcSystem=255, srcComponent=190)
        mav_atk = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)

        packets: List[Packet] = []
        labels_map: Dict[int, int] = {}  # packet_index -> label (0: Benign, 1: Injection, 2: Replay)

        if base_time is None:
            base_time = time.time() - 3600.0
        current_time = base_time

        uav_ip = "192.168.1.200"
        gcs_ip = "192.168.1.100"
        atk_ip = "192.168.1.105"
        port = 14550

        uav_seq = 40
        gcs_seq = 10
        atk_seq = 200

        lat = 377749000
        lon = -1224194000
        alt = 50000
        roll = 0.01
        pitch = -0.02
        yaw = 1.57

        stashed_attitude_packets: List[Tuple[float, Any]] = []

        for i in range(num_packets):
            time_step = 0.020 + (i % 5) * 0.005
            current_time += time_step

            # Determine scenario condition for this individual packet
            is_injection = False
            is_replay = False

            if attack_scenario == "injection":
                is_injection = (i >= 200)
            elif attack_scenario == "replay":
                is_replay = (i >= 300)
            elif attack_scenario == "mixed":
                # Only packets 400..480 are injections, 800..880 are replays; rest are BENIGN!
                if 400 <= i <= 480:
                    is_injection = True
                elif 800 <= i <= 880:
                    is_replay = True

            pkt_label = 0
            if is_injection:
                pkt_label = 1
                atk_seq = (atk_seq + 15) % 256  # Sequence skip anomaly
                time_step = 0.005
                current_time += time_step

                cmd = 21 if i % 2 == 0 else 400  # Land or Disarm
                p1 = 0.0
                msg = mav_atk.command_long_encode(
                    target_system=1,
                    target_component=1,
                    command=cmd,
                    confirmation=0,
                    param1=p1,
                    param2=0.0,
                    param3=0.0,
                    param4=0.0,
                    param5=0.0,
                    param6=0.0,
                    param7=0.0,
                )
                mav_atk.seq = atk_seq
                mav_payload = msg.pack(mav_atk)

                pkt = (
                    Ether(src="00:11:22:33:44:55", dst="66:77:88:99:AA:BB")
                    / IP(src=atk_ip, dst=uav_ip, tos=1)
                    / UDP(sport=port, dport=port)
                    / Raw(load=mav_payload)
                )

            elif is_replay:
                pkt_label = 2
                # Replay attack: rollback sequence and duplicate old attitude payload
                replay_seq = (uav_seq - 50) % 256
                msg = mav_uav.attitude_encode(
                    time_boot_ms=int((current_time - base_time) * 1000),
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    rollspeed=0.001,
                    pitchspeed=-0.001,
                    yawspeed=0.002,
                )
                mav_uav.seq = replay_seq
                mav_payload = msg.pack(mav_uav)

                pkt = (
                    Ether(src="00:11:22:33:44:55", dst="66:77:88:99:AA:BB")
                    / IP(src=uav_ip, dst=gcs_ip, tos=2)
                    / UDP(sport=port, dport=port)
                    / Raw(load=mav_payload)
                )

            else:
                # Normal Benign Telemetry & Legitimate GCS commands
                pkt_label = 0
                mod_cycle = i % 25

                if mod_cycle == 0:
                    # GCS Heartbeat (from sys_id=255, comp_id=190)
                    gcs_seq = (gcs_seq + 1) % 256
                    msg = mav_gcs.heartbeat_encode(
                        type=mavutil.mavlink.MAV_TYPE_GCS,
                        autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        base_mode=0,
                        custom_mode=0,
                        system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
                    )
                    mav_gcs.seq = gcs_seq
                    mav_payload = msg.pack(mav_gcs)
                    pkt = (
                        Ether(src="AA:BB:CC:DD:EE:FF", dst="00:11:22:33:44:55")
                        / IP(src=gcs_ip, dst=uav_ip, tos=0)
                        / UDP(sport=port, dport=port)
                        / Raw(load=mav_payload)
                    )

                elif mod_cycle == 5:
                    # Legitimate GCS Mission Waypoint Command
                    gcs_seq = (gcs_seq + 1) % 256
                    msg = mav_gcs.command_long_encode(
                        target_system=1,
                        target_component=1,
                        command=16,  # MAV_CMD_NAV_WAYPOINT
                        confirmation=0,
                        param1=0.0,
                        param2=2.0,
                        param3=0.0,
                        param4=0.0,
                        param5=37.7750,
                        param6=-122.4190,
                        param7=60.0,
                    )
                    mav_gcs.seq = gcs_seq
                    mav_payload = msg.pack(mav_gcs)
                    pkt = (
                        Ether(src="AA:BB:CC:DD:EE:FF", dst="00:11:22:33:44:55")
                        / IP(src=gcs_ip, dst=uav_ip, tos=0)
                        / UDP(sport=port, dport=port)
                        / Raw(load=mav_payload)
                    )

                else:
                    # UAV Autopilot Telemetry (sys_id=1, comp_id=1)
                    uav_seq = (uav_seq + 1) % 256
                    if mod_cycle in (1, 11, 21):
                        msg = mav_uav.heartbeat_encode(
                            type=mavutil.mavlink.MAV_TYPE_QUADROTOR,
                            autopilot=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                            base_mode=mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED | mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            custom_mode=4,
                            system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
                        )
                    elif mod_cycle in (2, 7, 12, 17, 22):
                        roll += 0.002 * ((i % 5) - 2)
                        pitch += 0.001 * ((i % 3) - 1)
                        msg = mav_uav.attitude_encode(
                            time_boot_ms=int((current_time - base_time) * 1000),
                            roll=roll,
                            pitch=pitch,
                            yaw=yaw,
                            rollspeed=0.001,
                            pitchspeed=-0.001,
                            yawspeed=0.002,
                        )
                    elif mod_cycle in (3, 13, 23):
                        lat += 5
                        lon += 2
                        alt += 5
                        msg = mav_uav.global_position_int_encode(
                            time_boot_ms=int((current_time - base_time) * 1000),
                            lat=lat,
                            lon=lon,
                            alt=alt,
                            relative_alt=alt - 10000,
                            vx=120,
                            vy=50,
                            vz=-10,
                            hdg=15700,
                        )
                    elif mod_cycle in (4, 14, 24):
                        msg = mav_uav.gps_raw_int_encode(
                            time_usec=int((current_time - base_time) * 1e6),
                            fix_type=3,
                            lat=lat,
                            lon=lon,
                            alt=alt,
                            eph=120,
                            epv=140,
                            vel=150,
                            cog=15700,
                            satellites_visible=12,
                        )
                    else:
                        msg = mav_uav.sys_status_encode(
                            onboard_control_sensors_present=1023,
                            onboard_control_sensors_enabled=1023,
                            onboard_control_sensors_health=1023,
                            load=250,
                            voltage_battery=16400,
                            current_battery=1250,
                            battery_remaining=88,
                            drop_rate_comm=0,
                            errors_comm=0,
                            errors_count1=0,
                            errors_count2=0,
                            errors_count3=0,
                            errors_count4=0,
                        )

                    mav_uav.seq = uav_seq
                    mav_payload = msg.pack(mav_uav)
                    pkt = (
                        Ether(src="00:11:22:33:44:55", dst="AA:BB:CC:DD:EE:FF")
                        / IP(src=uav_ip, dst=gcs_ip, tos=0)
                        / UDP(sport=port, dport=port)
                        / Raw(load=mav_payload)
                    )

            pkt.time = current_time
            packets.append(pkt)
            labels_map[i] = pkt_label

        # Write PCAP file
        wrpcap(output_filepath, packets)

        # Write sidecar ground-truth files for 100% scientific verification
        base, _ = os.path.splitext(output_filepath)
        sidecar_json = f"{base}_labels.json"
        with open(sidecar_json, "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)

        sidecar_csv = f"{base}_ground_truth.csv"
        df_gt = pd.DataFrame([{"packet_idx": k, "label": v} for k, v in labels_map.items()])
        df_gt.to_csv(sidecar_csv, index=False)

        return output_filepath


class FlightAttackInjector:
    """
    Forensic Attack Injector for Real Flight Captures.
    Takes an authentic flight capture (.pcap) and injects realistic attacks
    directly into the authentic flight timeline, preserving all real sensors and timestamps.
    """

    @staticmethod
    def inject_attacks_into_pcap(
        input_pcap_path: str,
        output_pcap_path: str,
        injection_count: int = 50,
        replay_count: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Reads authentic flight PCAP, inserts realistic Command Injection and Replay packets
        into the real timeline, and marks packet-level ground truth.
        """
        if not os.path.exists(input_pcap_path):
            raise FileNotFoundError(f"Input PCAP not found: {input_pcap_path}")

        reader = PcapReader(input_pcap_path)
        all_packets: List[Packet] = []
        try:
            for p in reader:
                all_packets.append(p)
        finally:
            reader.close()

        if not all_packets:
            raise ValueError(f"No packets found in {input_pcap_path}")

        mav_atk = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        blended_packets: List[Packet] = []
        labels_map: Dict[int, int] = {}

        total_input = len(all_packets)
        inject_idx_start = int(total_input * 0.35)
        replay_idx_start = int(total_input * 0.70)

        atk_seq = 150
        injected_so_far = 0
        replayed_so_far = 0

        for i, pkt in enumerate(all_packets):
            cur_time = float(pkt.time) if hasattr(pkt, "time") and pkt.time is not None else time.time()

            # Normal authentic flight packet (label=0)
            if pkt.haslayer(IP):
                pkt[IP].tos = 0
            blended_packets.append(pkt)
            labels_map[len(blended_packets) - 1] = 0

            # Inject Command Injection attack packets in the mid-flight segment
            if inject_idx_start <= i < inject_idx_start + injection_count * 2 and injected_so_far < injection_count:
                injected_so_far += 1
                atk_seq = (atk_seq + 17) % 256  # Sequence skip anomaly
                cmd = 21 if injected_so_far % 2 == 0 else 400  # Land or Disarm
                msg = mav_atk.command_long_encode(
                    target_system=1,
                    target_component=1,
                    command=cmd,
                    confirmation=0,
                    param1=0.0,
                    param2=0.0,
                    param3=0.0,
                    param4=0.0,
                    param5=0.0,
                    param6=0.0,
                    param7=0.0,
                )
                mav_atk.seq = atk_seq
                mav_payload = msg.pack(mav_atk)

                atk_pkt = (
                    Ether(src="00:11:22:33:44:55", dst="66:77:88:99:AA:BB")
                    / IP(src="192.168.1.105", dst="192.168.1.200", tos=1)
                    / UDP(sport=14550, dport=14550)
                    / Raw(load=mav_payload)
                )
                atk_pkt.time = cur_time + 0.002
                blended_packets.append(atk_pkt)
                labels_map[len(blended_packets) - 1] = 1

            # Inject Replay attack packets in the late-flight segment
            if replay_idx_start <= i < replay_idx_start + replay_count * 2 and replayed_so_far < replay_count:
                replayed_so_far += 1
                # Grab a packet from 200 steps ago to replay
                stale_idx = max(0, i - 200)
                stale_pkt = all_packets[stale_idx].copy()
                if stale_pkt.haslayer(IP):
                    stale_pkt[IP].tos = 2
                stale_pkt.time = cur_time + 0.001
                blended_packets.append(stale_pkt)
                labels_map[len(blended_packets) - 1] = 2

        # Sort blended packets chronologically by timestamp
        blended_packets.sort(key=lambda p: float(p.time) if hasattr(p, "time") and p.time is not None else 0.0)

        # Re-index labels map after chronological sort
        final_labels: Dict[int, int] = {}
        for idx, p in enumerate(blended_packets):
            lbl = 0
            if p.haslayer(IP) and p[IP].tos in (1, 2):
                lbl = p[IP].tos
            final_labels[idx] = lbl

        wrpcap(output_pcap_path, blended_packets)

        base, _ = os.path.splitext(output_pcap_path)
        with open(f"{base}_labels.json", "w", encoding="utf-8") as f:
            json.dump(final_labels, f, indent=2)

        df_gt = pd.DataFrame([{"packet_idx": k, "label": v} for k, v in final_labels.items()])
        df_gt.to_csv(f"{base}_ground_truth.csv", index=False)

        stats = {
            "total_packets": len(blended_packets),
            "benign_packets": sum(1 for v in final_labels.values() if v == 0),
            "injection_packets": sum(1 for v in final_labels.values() if v == 1),
            "replay_packets": sum(1 for v in final_labels.values() if v == 2),
            "output_path": output_pcap_path,
        }
        return output_pcap_path, stats
