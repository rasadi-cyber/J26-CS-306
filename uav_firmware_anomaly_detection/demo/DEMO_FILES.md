# Demo files (simulated tampering, never flash these)

| File | Made from | What was changed | Expected result |
|---|---|---|---|
| copter_safety_content_removed.apj | Copter 4.7.2 | RTL, failsafe, geofence names zeroed | Anomalous |
| plane_geofence_removed.apj | Plane 4.5.7 CUAVv5 | geofence names zeroed | Anomalous |
| px4_safety_content_removed.px4 | PX4 v1.17.0 cuav_nora | RTL, failsafe, geofence names zeroed | Anomalous |
| copter_small_byte_edit.apj | Copter 4.6.3 CUAVv5 | ONE byte flipped | Normal (known limitation) |
| rover_relabelled.apj | Test file with Rover banner | vehicle name changed to Rover | Refused (not a UAV) |

Clean comparison files are in data\raw (e.g. ardupilot\Plane\4.5.7\CUAVv5__arduplane.apj, px4\v1.17.0\cuav_nora_default.px4) and show Normal.
All edits keep the file size the same. These are controlled, simulated modifications, not real malware.
