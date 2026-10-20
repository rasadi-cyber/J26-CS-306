"""Choose the right saved models for an uploaded firmware image.

ArduPilot images carry a version banner such as "ArduCopter V4.7.2" or
"ArduPlane V4.0.9", which identifies the vehicle type without running the
firmware. The component is specific to aerial vehicles, so ground, marine
and antenna-tracker builds are refused instead of being scored by a model
that has never seen them.
"""
from pathlib import Path
import re


class UnsupportedVehicleError(ValueError):
    """The firmware is not for an aerial vehicle this component covers."""


BANNER = re.compile(
    rb"Ardu(Copter|Plane|Rover|Sub|Blimp|Tracker) V(\d+\.\d+\.\d+)"
)
HELICOPTER_PARAMETER = re.compile(rb"\bH_RSC_[A-Z0-9_]+\b")

VEHICLE_CLASS = {"Copter": "multirotor", "Plane": "fixed_wing"}
NOT_AERIAL = {"Rover", "Sub", "Tracker"}


def detect_vehicle(family, main_image, external_image=b""):
    """Return the vehicle description and which model scope to use."""
    if family == "PX4":
        return {
            "vehicle": "PX4",
            "vehicle_class": "px4_vehicles",
            "release": None,
            "model_scope": "all",
            "note": "PX4 builds cover several vehicle types; the general "
                    "PX4 model is used.",
        }

    images = [image for image in (main_image, external_image) if image]

    match = None
    for image in images:
        match = BANNER.search(image)
        if match:
            break

    if match is None:
        return {
            "vehicle": "ArduPilot (type not identified)",
            "vehicle_class": "unknown",
            "release": None,
            "model_scope": "all",
            "note": "No ArduCopter/ArduPlane version banner was found; "
                    "the general ArduPilot model is used.",
        }

    name = match.group(1).decode("ascii")
    release = match.group(2).decode("ascii")

    if name in NOT_AERIAL:
        raise UnsupportedVehicleError(
            f"This is ArduPilot {name} firmware, which is not for an aerial "
            "vehicle. This component is specific to UAV firmware "
            "(ArduCopter, ArduPlane and PX4)."
        )

    if name == "Blimp":
        raise UnsupportedVehicleError(
            "ArduPilot Blimp firmware is not covered: no stable Blimp "
            "release builds were available for training."
        )

    if name == "Copter" and any(
        HELICOPTER_PARAMETER.search(image) for image in images
    ):
        return {
            "vehicle": "ArduCopter (helicopter build)",
            "vehicle_class": "helicopter",
            "release": release,
            "model_scope": "all",
            "note": "Helicopter builds had no separate validation data; "
                    "the general ArduPilot model is used.",
        }

    return {
        "vehicle": f"Ardu{name}",
        "vehicle_class": VEHICLE_CLASS[name],
        "release": release,
        "model_scope": VEHICLE_CLASS[name],
        "note": "",
    }


def model_paths(family, model_scope, model_directory):
    """Paths of the main One-Class SVM and the Isolation Forest baseline."""
    directory = Path(model_directory)
    stem = f"{family}_{model_scope}_uav"
    return (
        directory / f"{stem}_one_class_svm.joblib",
        directory / f"{stem}_isolation_forest.joblib",
    )
