from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_routing import UnsupportedVehicleError, detect_vehicle, model_paths


def test_copter_is_multirotor():
    info = detect_vehicle("ArduPilot", b"xx ArduCopter V4.6.3 (abc) xx")
    assert info["vehicle"] == "ArduCopter"
    assert info["vehicle_class"] == "multirotor"


def test_plane_is_fixed_wing():
    info = detect_vehicle("ArduPilot", b"ArduPlane V4.5.7 (abc)")
    assert info["vehicle_class"] == "fixed_wing"


def test_rover_is_refused():
    with pytest.raises(UnsupportedVehicleError):
        detect_vehicle("ArduPilot", b"ArduRover V4.5.7 (abc)")


def test_helicopter_detected_by_parameter():
    info = detect_vehicle("ArduPilot", b"ArduCopter V4.5.7 H_RSC_MODE")
    assert info["vehicle_class"] == "helicopter"


def test_px4_uses_all_scope():
    assert detect_vehicle("PX4", b"anything")["model_scope"] == "all"


def test_model_paths_names():
    main, base = model_paths("PX4", "all", Path("m"))
    assert main.name == "PX4_all_uav_one_class_svm.joblib"
    assert base.name == "PX4_all_uav_isolation_forest.joblib"
