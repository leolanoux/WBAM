# -*- coding: utf-8 -*-
"""
===============================================================================
WBAM_tutorial.py
===============================================================================
Author: Leonardo Lanoux
Email: lanoux@usp.br
GitHub: https://github.com/leolanoux
Creation Date: 2025-12-18
Last Update: 2026-08-17
Version: 1.0.0

Description:
------------
Self-contained Whole-Body Angular Momentum (WBAM) analysis pipeline for 3D
motion-capture data (Vicon Nexus, Plug-in Gait full-body model).  For each
dynamic trial of a participant it computes the orbital and local (spin)
components of angular momentum for 15 body segments (Head, Thorax, Pelvis and
six bilateral limb segments):

    L(t) = sum_i [ R_i(t) . (I_i . w_i(t))
                   + m_i * (r_i(t) - r_CoM(t)) x (v_i(t) - v_CoM(t)) ]

and analyses one stride per trial, chosen by a task-specific rule.

Segment angular velocity (w_i) is computed from each segment's own absolute
rotation in the laboratory frame, reconstructed from the Origin/Anterior/
Proximal technical points Nexus exports for every segment.  

WBAM is normalized by subject mass, whole-body CoM speed and mean static CoM
height, and every selected stride is resampled to 0-100% of the gait cycle.

Stride-selection rules:
-----------------------
Every candidate stride is bounded by two consecutive same-side ``Foot Strike``
events read from the trial's ``_events.csv``.  

* UW, BSR and CT: the right-foot stride whose temporal midpoint is closest to
  the midpoint of the complete trial (steady-state locomotion, away from gait
  initiation and termination).
* TOH: the right-foot stride whose two bounding Foot Strike positions lie on
  opposite sides of the progression-axis plane through the obstacle centroid.
  Strides entirely before or entirely after the obstacle are rejected.
* DOVdir and DOVesq: both right- and left-foot strides are equal candidates.
  A candidate is preferred when its own foot CoM crosses the obstacle plane
  during the stride interior; the crossing closest to that stride's temporal
  midpoint wins.  

The obstacle plane is perpendicular to the progression axis (lab Y) and passes
through the four-marker obstacle centroid: an arithmetic mean of the available
``obstaculo1``-``obstaculo4`` positions at each frame, followed by a temporal
median over the trial.  This is the centroid of the MARKER SET, not a measured
centre of mass or a volumetric centroid; it represents the object's horizontal
centre only if marker placement was symmetric around it.

CoM quality gate:
-----------------
Before WBAM is extracted, the prescribed stride is checked against the
original (pre-interpolation) whole-body CoM samples and against the same
filtered CoM speed that forms the normalization denominator.  A stride is
valid only when it spans at least 3 frames, lies inside the trial frame range,
has no missing original CoM samples, has finite CoM speed throughout, and has
a minimum CoM speed of at least ``MIN_COM_SPEED_M_S`` (0.001 m/s).

If the prescribed stride fails, a popup lists the valid alternatives on the
same side and asks the operator to choose one.  Cancelling, or having no valid
alternative, skips the trial.  Supplying a custom ``fallback_selector`` makes
the pipeline fully headless.  Every fallback is recorded in the audit table,
so operator intervention is always visible in the published results.

File organization:
------------------
PART 1  Configuration and model constants
PART 2  WBAM mechanics (loading, filtering, rotations, inertia, momentum)
PART 3  Stride selection (obstacle centroid, task rules, quality gate)
PART 4  Exports and figures
PART 5  Orchestration and entry point

The two layers stay separable in practice: ``compute_trial_wbam()`` holds all
of the mechanics and accepts a ``cycles_override`` argument, while
``process_trial()`` decides which stride to pass it.  To apply a different
stride-selection protocol, change PART 3 only; the mechanics need no edits.

Usage:
------
- Run directly: `python WBAM_tutorial.py`
- Two selection windows appear, in order:
  - **Folder selection**: choose the participant's data folder.
  - **Subject-file selection**: choose that participant's `*_subject.csv`.
- If the selected CSV has no valid `Bodymass`, a third window asks for mass.
- Results are written to `<participant>/<name>_results`.
- The module can also be imported to reuse individual functions without
  triggering any dialog; the dialogs only run inside the `__main__` block.
  When importing, set SUBJECT_MASS_KG, SOURCE_DIR, SUBJECT_FILE and OUTPUT_DIR
  before calling collect_results() or main().  SUBJECT_FILE is read only for
  Bodymass, via subject_mass_from_data().

Outputs:
--------
csv/selected_stride_audit.csv             one auditable row per trial
csv/individual_cycles_normalized.csv      the selected stride, resampled 0-100%
csv/individual_trial_series_normalized.csv
csv/task_medians_normalized.csv           median across trials, per task
csv/angular_velocity_audit.csv            peak |w| per segment/trial
figures/                                  PNG and SVG summary figures

Inertia model:
--------------
Winter/Dempster mass fractions as implemented by the Vicon Plug-in Gait kinetic
model, closing exactly to one body mass over 15 segments.  Segment radii of
gyration follow the PiG ratios.  Each segment's inertia tensor is obtained
from I = mass * (k * length)^2, where k is that segment's dimensionless PiG
radius-of-gyration ratio and length is the model-matched anatomical length
(joint-centre distances, and the specific landmark pairs Winter's ratios are
normalized by).  Segment shape is therefore never integrated, and the subject
CSV contributes only Bodymass to the calculation.

Conventions:
------------
Lab X = mediolateral, Y = progression/anteroposterior, Z = vertical.  WBAM
components are labelled by the axis normal to the plane of rotation:
L_X = sagittal, L_Y = frontal, L_Z = transverse.  Those plane labels apply to
angular momentum only, never to position or linear-velocity components.
Marker CSVs are in millimetres and converted to SI before any mechanics; the
obstacle centroid and foot progression series intentionally remain in mm,
since they are only ever compared with each other.  Left strides are stored
sign-flipped so they pool directly with right strides.

Requirements:
-------------
Python >= 3.10, numpy, pandas, scipy, matplotlib, tkinter.
ezc3d is needed only when C3D files must be auto-exported.

See README.md for input-data requirements, column dictionaries, the full API
reference, and guidance on adapting the task codes, marker names, sampling
rate and progression axis to another laboratory.

License:
--------
WBAM_tutorial.py -- Whole-Body Angular Momentum analysis pipeline
Copyright (C) 2026  Leonardo Lanoux

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
===============================================================================
"""


from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from scipy import signal
from scipy.interpolate import interp1d


# =============================================================================
# PART 1 - CONFIGURATION AND MODEL CONSTANTS
# Paths, sampling, unit conversion, task list, inertia model tables.
# =============================================================================


# SOURCE_DIR (participant folder) and OUTPUT_DIR (results folder) have
# no fixed path - they are only set when the script runs directly,
# through the folder-selection window (see select_participant_folder()
# and the __main__ block at the end of the file). They are left as
# None here just so the names exist in the module before that happens.
SOURCE_DIR: Path | None = None
OUTPUT_DIR: Path | None = None
# Selected explicitly by the operator when the script runs directly.
# It is never inferred from a participant-specific folder or filename.
SUBJECT_FILE: Path | None = None

FREQ_HZ = 200.0
DT = 1.0 / FREQ_HZ

# Instantaneous-speed normalization is numerically unstable when the
# whole-body COM is stationary (or has been made artificially stationary by
# filling a long missing-data interval). This small absolute floor is well
# below the minimum speed in every source-valid P01 cycle (0.043 m/s), while
# still rejecting effectively zero denominators such as CT02 left cycle 1.
MIN_COM_SPEED_M_S = 0.001

# Participant mass (kg) - has no fixed value; it is requested through a
# window when the script runs directly (see ask_subject_mass() and the
# __main__ block at the end of the file). Left as None here just so
# the name exists in the module before that happens.
SUBJECT_MASS_KG: float | None = None

# Vicon Nexus exports all marker/COM positions in millimeters, but the
# rest of the calculation (mass in kg, inertia tensor in kg*m^2,
# angular velocity in rad/s) assumes SI units (meters). Without this
# conversion, the orbital term of the angular momentum
# (L_orbital = delta_r x m*delta_v) comes out ~1000x larger than it
# should, artificially dominating the local/spin term (I*w).
MM_TO_M = 0.001

TASKS = ["UW", "TOH", "DOVdir", "DOVesq", "BSR", "CT"]
TRIALS = ["01", "02", "03", "04", "05"]

TASK_ABBREV = {
    "UW": "HW",
    "TOH": "HOC",
    "DOVdir": "RVOC",
    "DOVesq": "LVOC",
    "BSR": "RBS",
    "CT": "BW",
}

TASK_FULL_NAMES = {
    "UW": "Habitual Walking",
    "TOH": "Horizontal Obstacle Crossing",
    "DOVdir": "Right Vertical Obstacle Circumvention",
    "DOVesq": "Left Vertical Obstacle Circumvention",
    "BSR": "Walking with a Reduced Base of Support",
    "CT": "Beam Walking",
}

# Body mass fractions per segment.
SEGMENT_MASS_RATIO = {
    "Head": .081, "Thorax": .355, "Pelvis": .142,
    "L_UpperArm": .028, "R_UpperArm": .028,
    "L_Forearm": .016, "R_Forearm": .016,
    "L_Hand": .006, "R_Hand": .006,
    "L_Thigh": .100, "R_Thigh": .100,
    "L_Shank": .0465, "R_Shank": .0465,
    "L_Foot": .0145, "R_Foot": .0145,
}
INERTIA_MODEL_NAME = "Vicon Plug-in Gait / Winter-Dempster"
VICON_GYRATION_RATIO = {
    "Head": .495, "Thorax": .310, "Pelvis": .310,
    "UpperArm": .322, "Forearm": .303, "Hand": .223,
    "Thigh": .323, "Shank": .302, "Foot": .475,
}


SEGMENT_COM_PREFIX = {
    "Head": "HeadCOM",
    "Thorax": "ThoraxCOM",
    "Pelvis": "PelvisCOM",
    "L_UpperArm": "LeftHumerusCOM",
    "R_UpperArm": "RightHumerusCOM",
    "L_Forearm": "LeftRadiusCOM",
    "R_Forearm": "RightRadiusCOM",
    "L_Hand": "LeftHandCOM",
    "R_Hand": "RightHandCOM",
    "L_Thigh": "LeftFemurCOM",
    "R_Thigh": "RightFemurCOM",
    "L_Shank": "LeftTibiaCOM",
    "R_Shank": "RightTibiaCOM",
    "L_Foot": "LeftFootCOM",
    "R_Foot": "RightFootCOM",
}

# Prefix of the O/A/L/P technical points (Origin, Anterior, Lateral,
# Proximal) that Nexus exports for each segment. Used in Method B to
# reconstruct the segment's absolute rotation at each frame.
SEGMENT_OALP_PREFIX = {
    "Head": "HED",
    "Thorax": "TRX",
    "Pelvis": "PEL",
    "L_UpperArm": "LHU",
    "R_UpperArm": "RHU",
    "L_Forearm": "LRA",
    "R_Forearm": "RRA",
    "L_Hand": "LHN",
    "R_Hand": "RHN",
    "L_Thigh": "LFE",
    "R_Thigh": "RFE",
    "L_Shank": "LTI",
    "R_Shank": "RTI",
    "L_Foot": "LFO",
    "R_Foot": "RFO",
}

# Anatomical joint-centre columns are present in some exports as LSJC,
# LEJC, etc. In those files the same modeled joint centres can be 
# recovered from the neighbouring segment's O/P technical points. 
# Keeping this mapping explicit prevents the generic
# O->P fallback from silently changing the anatomical definition.
JOINT_CENTER_TECHNICAL_FALLBACK = {
    "LSJC": "LHUP", "RSJC": "RHUP",
    "LEJC": "LHUO", "REJC": "RHUO",
    "LWJC": "LRAO", "RWJC": "RRAO",
    "LHJC": "LFEP", "RHJC": "RFEP",
    "LKJC": "LFEO", "RKJC": "RFEO",
    "LAJC": "LTIO", "RAJC": "RTIO",
}


AXIS_NAMES = ["X", "Y", "Z"]
PERCENT_CYCLE = np.linspace(0.0, 100.0, 100)


PROGRESSION_AXIS = "Y"
OBSTACLE_MARKERS = ("obstaculo1", "obstaculo2", "obstaculo3", "obstaculo4")


# =============================================================================
# PART 2 - WBAM MECHANICS
# Data loading, filtering, segment rotation/inertia, angular momentum.
# =============================================================================


@dataclass
class TrialResult:
    """Consolidated result of one trial after the complete WBAM processing.

    Attributes:
        task: Task code (e.g., "UW", "BSR").
        trial: Trial number (e.g., "01").
        right_cycles: List of (100, 3) arrays with the normalized WBAM,
            interpolated to 0-100% of the gait cycle, one per cycle
            starting with a right footstep.
        left_cycles_flipped: Same, for cycles starting with a left
            footstep, with the sign flipped (*-1) so they can be
            compared directly with the right cycles.
        angular_velocity_max_abs: Largest |w| (rad/s) observed for each
            segment during the trial - used as an audit/QA check to
            see whether the computed angular velocity is in a
            physiologically plausible range.
    """

    task: str
    trial: str
    right_cycles: list[np.ndarray]
    left_cycles_flipped: list[np.ndarray]
    angular_velocity_max_abs: dict[str, float]


def safe_filter(values: np.ndarray, fs: float = FREQ_HZ, cutoff: float = 6.0, order: int = 4) -> np.ndarray:
    """Applies a zero-phase low-pass Butterworth filter (filtfilt).

    Used to smooth positions, velocities and angular velocities before
    they are differentiated/used in the angular momentum calculation,
    removing high-frequency noise without introducing phase lag.

    Args:
        values: Array to filter, shape (n_frames, ...).
        fs: Sampling frequency, in Hz.
        cutoff: Low-pass cutoff frequency, in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered array, same shape as `values`. If the series is too
        short for the pad length required by filtfilt (padlen), the
        original array is returned unfiltered, to avoid the filter
        raising an exception on very short segments.
    """
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        return values
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if values.shape[0] <= padlen:
        return values
    return signal.filtfilt(b, a, values, axis=0)


def numeric_frame(df: pd.DataFrame, columns: list[str], index: pd.Index) -> pd.DataFrame:
    """Extracts a subset of columns/rows from a DataFrame, forcing numeric conversion and filling gaps.

    Args:
        df: Source DataFrame (e.g., the full markers table for one trial).
        columns: Column names to extract.
        index: Row index (frame numbers) to extract.

    Returns:
        DataFrame with the requested columns/rows, coerced to numeric
        (any non-numeric value becomes NaN via errors="coerce"). Small
        tracking gaps (a marker missing for a few frames) are filled by
        linear interpolation, and edges with no valid neighbor on
        either side are filled by forward/backward fill.
    """
    out = df.loc[index, columns].apply(pd.to_numeric, errors="coerce")
    out = out.interpolate(limit_direction="both").ffill().bfill()
    return out


def load_com_height_static(source_dir: Path) -> float:
    """Reads the static trial and computes the mean whole-body COM height.

    Args:
        source_dir: Participant folder, containing a
            `PO1_N_AV1_anatomica` subfolder with `anatomica01_markers.csv`.

    Returns:
        Mean height of the body's center of mass (CentreOfMass_Z), in
        meters. Used as a WBAM normalization factor together with the
        subject's mass and velocity, allowing comparison between
        people of different sizes.

    Raises:
        ValueError: If the CentreOfMass_Z column is missing from the
            static trial file, or has no valid (numeric) values.
    """
    markers_path = source_dir / "PO1_N_AV1_anatomica" / "anatomica01_markers.csv"
    if not markers_path.exists():
        candidates = sorted(source_dir.rglob("*anatomica*_markers.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one static anatomica markers CSV below {source_dir}; found {len(candidates)}"
            )
        markers_path = candidates[0]
    markers = pd.read_csv(markers_path)
    if "CentreOfMass_Z" not in markers.columns:
        raise ValueError(f"Column CentreOfMass_Z not found in {markers_path}")
    com_height = pd.to_numeric(markers["CentreOfMass_Z"], errors="coerce").dropna()
    if com_height.empty:
        raise ValueError(f"No valid CentreOfMass_Z values in {markers_path}")
    return float(com_height.mean()) * MM_TO_M


def load_subject_parameters(subject_file: Path | None = None) -> dict[str, float]:
    """Read anthropometric parameters from the operator-selected CSV.

    Args:
        subject_file: CSV selected by the operator, normally named
            ``anatomica01_subject.csv`` or ``anatomica_subject.csv``.
            When omitted during imported use, the module-level
            ``SUBJECT_FILE`` is used. No participant-specific path is inferred.

    Returns:
        Dictionary mapping parameter names to numeric values.

    Raises:
        FileNotFoundError: If a supplied path does not exist.
        ValueError: If the CSV does not contain ``Parameter`` and ``Value``.
    """
    selected = subject_file if subject_file is not None else SUBJECT_FILE
    if selected is None:
        return {}
    selected = Path(selected)
    if not selected.is_file():
        raise FileNotFoundError(f"Subject parameter CSV not found: {selected}")
    df = pd.read_csv(selected)
    required = {"Parameter", "Value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{selected} must contain columns Parameter and Value; "
            f"found {list(df.columns)}"
        )
    params: dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            params[str(row["Parameter"]).strip()] = float(row["Value"])
        except (ValueError, TypeError):
            continue
    return params


def subject_mass_from_data(subject_file: Path | None = None) -> float | None:
    """Read body mass from the explicitly selected subject CSV."""
    params = load_subject_parameters(subject_file)
    mass = params.get("Bodymass")
    if mass is None or mass <= 0:
        return None
    return float(mass)


def export_discovered_c3d_trial(
    source_dir: Path, task: str, trial: str
) -> tuple[Path, Path, Path] | None:
    """Discover and cache a C3D trial when pre-exported WBAM CSVs are absent.

    Discovery is recursive and participant-name independent. Exact ``task+trial``
    matches are preferred. For trial 01 only, a task-only filename such as
    ``tentativaUW.c3d`` is accepted. Static/anatomical C3Ds are excluded.
    """
    exact = [p for p in source_dir.rglob("*.c3d") if f"{task}{trial}".lower() in p.stem.lower()]
    candidates = exact
    if not candidates and trial == "01":
        candidates = [p for p in source_dir.rglob("*.c3d")
                      if task.lower() in p.stem.lower()
                      and "anatom" not in p.stem.lower()]
    if len(candidates) != 1:
        return None
    c3d_path = candidates[0]
    cache = source_dir / "_wbam_export" / f"{task}{trial}"
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{task}{trial}"
    markers_path = cache / f"{stem}_markers.csv"
    frames_path = cache / f"{stem}_frames.csv"
    events_path = cache / f"{stem}_events.csv"
    if all(path.exists() for path in (markers_path, frames_path, events_path)):
        return markers_path, frames_path, events_path

    import ezc3d
    c3d = ezc3d.c3d(str(c3d_path))
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    points = np.asarray(c3d["data"]["points"][:3], dtype=float).transpose(2, 1, 0)
    marker_columns = [f"{label}_{axis}" for label in labels for axis in AXIS_NAMES]
    pd.DataFrame(points.reshape(points.shape[0], -1), columns=marker_columns).to_csv(markers_path, index=False)
    first = int(c3d["header"]["points"]["first_frame"])
    last = int(c3d["header"]["points"]["last_frame"])
    pd.DataFrame({"first_frame": [first], "last_frame": [last]}).to_csv(frames_path, index=False)

    event = c3d["parameters"].get("EVENT", {})
    contexts = list(event.get("CONTEXTS", {}).get("value", []))
    event_labels = list(event.get("LABELS", {}).get("value", []))
    raw_times = np.asarray(event.get("TIMES", {}).get("value", np.empty((2, 0))), dtype=float)
    times = raw_times[1] if raw_times.ndim == 2 and raw_times.shape[0] > 1 else np.array([])
    names = []
    counts: dict[str, int] = {}
    for context, label in zip(contexts, event_labels):
        base = f"{context}_{label}"
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    pd.DataFrame([times], columns=names).to_csv(events_path, index=False)
    return markers_path, frames_path, events_path


def load_trial(source_dir: Path, task: str, trial: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, int, int]:
    """Locates and reads the 3 files of a trial (markers, frames, events).

    Reindexes the markers DataFrame so the row index matches the
    actual frame number (start_frame..end_frame), which allows using
    `.loc[frame]` instead of relative positions in the rest of the
    script.

    Args:
        source_dir: Participant folder.
        task: Task code (e.g., "UW").
        trial: Trial number (e.g., "01").

    Returns:
        Tuple (markers, frames, events_path, start_frame, end_frame),
        following the `PO1_N_AV1_<task><trial>` folder pattern.

    Raises:
        FileNotFoundError: If any of the 3 expected files does not exist.
    """
    stem = f"PO1_N_AV1_{task}{trial}"
    folder = source_dir / stem
    markers_path = folder / f"{stem}_markers.csv"
    frames_path = folder / f"{stem}_frames.csv"
    events_path = folder / f"{stem}_events.csv"

    if not all(path.exists() for path in (markers_path, frames_path, events_path)):
        discovered = export_discovered_c3d_trial(source_dir, task, trial)
        if discovered is not None:
            markers_path, frames_path, events_path = discovered
    for path in [markers_path, frames_path, events_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    markers = pd.read_csv(markers_path)
    frames = pd.read_csv(frames_path)
    frames.columns = frames.columns.str.strip()
    start_frame = int(frames["first_frame"].iloc[0])
    end_frame = int(frames["last_frame"].iloc[0])

    markers.index = pd.RangeIndex(start_frame, start_frame + len(markers))
    expected_end = start_frame + len(markers) - 1
    if end_frame != expected_end:
        end_frame = expected_end

    return markers, frames, events_path, start_frame, end_frame


def read_event_times(events_path: Path) -> dict[str, float]:
    """Reads events.csv and returns each event's name and time.

    Args:
        events_path: Path to the trial's `_events.csv` file (2 rows:
            event names and their times, in seconds).

    Returns:
        Dictionary {event_name: time_in_seconds}. Events with a
        repeated name (e.g., more than one "Right_Foot Strike") get
        numeric suffixes (_2, _3, ...) so they don't overwrite each
        other. Empty or invalid times are ignored.
    """
    with events_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    names = [name.strip() for name in rows[0]]
    times: dict[str, float] = {}
    for name, raw in zip(names, rows[1]):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if np.isfinite(value):
            # Repeated event names are intentionally overwritten by adding suffixes.
            key = name
            suffix = 2
            while key in times:
                key = f"{name}_{suffix}"
                suffix += 1
            times[key] = value
    return times


def gait_cycles_from_events(events_path: Path, start_frame: int, end_frame: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Converts heel-contact events into complete gait cycle frame ranges.

    Each cycle runs from one "Foot Strike" to the next one on the same
    side. Cycles outside the trial's [start_frame, end_frame] range,
    or with start >= end, are discarded.

    Args:
        events_path: Path to the trial's `_events.csv` file.
        start_frame: First frame of the trial.
        end_frame: Last frame of the trial.

    Returns:
        Tuple (right_cycles, left_cycles), each a list of
        (start_frame, end_frame) tuples.
    """
    events = read_event_times(events_path)

    def strikes(side: str) -> list[float]:
        prefix = f"{side}_Foot Strike"
        return sorted(time for name, time in events.items() if name.startswith(prefix))

    def cycles(side_strikes: list[float]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for a, b in zip(side_strikes[:-1], side_strikes[1:]):
            cycle_start = int(round(a * FREQ_HZ))
            cycle_end = int(round(b * FREQ_HZ))
            if cycle_start < cycle_end and cycle_start >= start_frame and cycle_end <= end_frame:
                out.append((cycle_start, cycle_end))
        return out

    return cycles(strikes("Right")), cycles(strikes("Left"))


def available_segments(markers: pd.DataFrame) -> list[str]:
    """Returns which of the 15 segments have COM data in this trial.

    Allows a trial to be processed even if some segment is missing
    from the CSV (e.g., a marker lost during data collection).

    Args:
        markers: Trial's markers DataFrame.

    Returns:
        List of segment names (out of the 15 in SEGMENT_MASS_RATIO)
        whose 3 COM position columns (_X, _Y, _Z) exist in `markers`.
    """
    segments = []
    for segment, prefix in SEGMENT_COM_PREFIX.items():
        cols = [f"{prefix}_{axis}" for axis in AXIS_NAMES]
        if all(col in markers.columns for col in cols):
            segments.append(segment)
    return segments


def compute_linear_velocity(position: np.ndarray, dt: float = DT) -> np.ndarray:
    """Differentiates position over time to obtain linear velocity.

    Args:
        position: Position array, in meters.
        dt: Time step between frames, in seconds.

    Returns:
        Velocity array (central difference, via np.gradient), same
        shape as `position`, in m/s.
    """
    if position.shape[0] < 2:
        return np.zeros_like(position)
    return np.gradient(position, dt, axis=0)


def rotation_matrices_from_oalp(
    markers: pd.DataFrame,
    index: pd.Index,
    segment: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a segment's rotation matrix at every frame from its O/A/P technical points.

    Local X axis = longitudinal (proximal-distal), in the O->P
        direction. This is the axis segment_inertia_tensor() treats as
        longitudinal, and the one ordinary segments receive zero inertia
        about.
    Local Z axis = perpendicular to the longitudinal/anterior plane.
    Local Y axis = completes the orthonormal basis (right-hand rule).

    Note on units: the O/A/P points are read directly from the CSV
    (in millimeters), without converting to meters. This is
    intentional and introduces no error: since we only use the
    DIRECTIONS (normalized vectors) between these points to build the
    orthonormal basis, the scale (mm vs m) cancels out and the
    resulting rotation matrix is dimensionless.

    Args:
        markers: Trial's markers DataFrame.
        index: Frame index to compute the rotation for.
        segment: Segment name (key of SEGMENT_OALP_PREFIX).

    Returns:
        Tuple (rotations, valid): `rotations` is an (n_frames, 3, 3)
        array (NaN where the rotation could not be computed); `valid`
        is a boolean array flagging which frames have a usable
        rotation matrix.

    Raises:
        KeyError: If the O/A/P columns for this segment are missing
            from `markers`.
    """
    prefix = SEGMENT_OALP_PREFIX[segment]
    cols = {p: [f"{prefix}{p}_{axis}" for axis in AXIS_NAMES] for p in ["O", "A", "P"]}

    missing = [c for group in cols.values() for c in group if c not in markers.columns]
    if missing:
        raise KeyError(f"Missing O/A/P columns for {segment}: {missing}")

    origin = numeric_frame(markers, cols["O"], index).to_numpy(dtype=float)
    anterior = numeric_frame(markers, cols["A"], index).to_numpy(dtype=float)
    proximal = numeric_frame(markers, cols["P"], index).to_numpy(dtype=float)

    n = len(index)
    rotations = np.full((n, 3, 3), np.nan)

    v_long = proximal - origin
    v_ant = anterior - origin
    norm_long = np.linalg.norm(v_long, axis=1)
    valid = norm_long > 1e-9

    for i in np.where(valid)[0]:
        e_x = v_long[i] / norm_long[i]
        e_z = np.cross(e_x, v_ant[i])
        norm_z = np.linalg.norm(e_z)
        if norm_z < 1e-9:
            valid[i] = False
            continue
        e_z = e_z / norm_z
        e_y = np.cross(e_z, e_x)
        rotations[i] = np.column_stack([e_x, e_y, e_z])

    return rotations, valid


def segment_technical_length_series(
    markers: pd.DataFrame,
    index: pd.Index,
    segment: str,
) -> np.ndarray:
    """Per-frame O->P technical length of one segment, in meters.

    Uses the same Origin and Proximal technical points as
    rotation_matrices_from_oalp(), so each segment gets its own measured
    longitudinal length rather than a shared constant.

    This is only a fallback. segment_anatomical_length_series() prefers the
    model-matched anatomical definition (joint-centre distances and the
    landmark pairs Winter's radii of gyration are normalized by), and calls
    this function only for legacy exports that lack the required virtual
    joint centres.

    Args:
        markers: Trial's markers DataFrame.
        index: Frame index to compute for.
        segment: Segment name (key of SEGMENT_OALP_PREFIX).

    Returns:
        (n_frames,) array of lengths, in meters. Frames whose O/P points are
        unavailable fall back to the median length of the valid frames in the
        same trial, or to zero if none are valid - which makes that frame's
        inertia tensor zero rather than raising.
    """
    prefix = SEGMENT_OALP_PREFIX[segment]
    o_cols = [f"{prefix}O_{axis}" for axis in AXIS_NAMES]
    p_cols = [f"{prefix}P_{axis}" for axis in AXIS_NAMES]

    if not all(col in markers.columns for col in o_cols + p_cols):
        return np.zeros(len(index))

    origin = numeric_frame(markers, o_cols, index).to_numpy(dtype=float) * MM_TO_M
    proximal = numeric_frame(markers, p_cols, index).to_numpy(dtype=float) * MM_TO_M
    length = np.linalg.norm(proximal - origin, axis=1)

    valid = length > 1e-6
    if not valid.any():
        return np.zeros(len(index))

    fallback_length = float(np.median(length[valid]))
    return np.where(valid, length, fallback_length)


def _base_segment(segment: str) -> str:
    return segment.removeprefix("L_").removeprefix("R_")


def segment_anatomical_length_series(
    markers: pd.DataFrame, index: pd.Index, segment: str,
    segment_com: np.ndarray, rotations: np.ndarray,
) -> np.ndarray:
    """Return the model-matched rigid segment length for every frame (meters).

    Winter/Vicon gyration ratios are normalized by specific landmark
    distances, not the visible external extent. The foot reference is
    AJC->TOE (ankle joint centre to the second-metatarsal-head TOE marker),
    matching the Plug-in Gait output specification and Winter's lateral-
    malleolus->metatarsal-II definition. It intentionally excludes the heel.

    Explicit joint-centre columns are preferred. If they are absent (as in
    N1), JOINT_CENTER_TECHNICAL_FALLBACK resolves the corresponding Vicon
    O/P technical point.
    """
    n=len(index)
    def point(label: str) -> np.ndarray:
        cols=[f"{label}_{axis}" for axis in AXIS_NAMES]
        if not all(c in markers.columns for c in cols):
            technical_label=JOINT_CENTER_TECHNICAL_FALLBACK.get(label)
            if technical_label is not None:
                cols=[f"{technical_label}_{axis}" for axis in AXIS_NAMES]
        if not all(c in markers.columns for c in cols):
            raise KeyError(f"Missing point {label}")
        return numeric_frame(markers,cols,index).to_numpy(dtype=float)*MM_TO_M
    def median_distance(a: str,b: str) -> float:
        return float(np.nanmedian(np.linalg.norm(point(a)-point(b),axis=1)))
    pairs={
        "L_UpperArm":("LSJC","LEJC"), "R_UpperArm":("RSJC","REJC"),
        "L_Forearm":("LEJC","LWJC"), "R_Forearm":("REJC","RWJC"),
        "L_Thigh":("LHJC","LKJC"), "R_Thigh":("RHJC","RKJC"),
        "L_Shank":("LKJC","LAJC"), "R_Shank":("RKJC","RAJC"),
    }
    try:
        if segment in pairs:
            length=median_distance(*pairs[segment])
        elif segment in ("L_Foot","R_Foot"):
            side=segment[0]; length=median_distance(side+"AJC",side+"TOE")
        elif segment in ("L_Hand","R_Hand"):
            side=segment[0]; length=median_distance(side+"WJC",side+"FIN")/.75
        elif segment=="Thorax":
            relative=point("C7")-point("PELP")
            local=np.einsum("nji,nj->ni",rotations,relative)
            length=float(np.linalg.norm(np.nanmean(local,axis=0)))
        elif segment=="Pelvis":
            length=.925*median_distance("LHJC","RHJC")
        elif segment=="Head":
            local=np.einsum("nji,nj->ni",rotations,point("C7")-segment_com)
            length=float(np.linalg.norm(np.nanmean(local,axis=0)))
        else:
            raise KeyError(segment)
    except (KeyError,ValueError):
        # Explicit fallback for legacy exports lacking virtual joint centers.
        fallback=segment_technical_length_series(markers,index,segment)
        valid=fallback[np.isfinite(fallback)&(fallback>0)]
        length=float(np.nanmedian(valid)) if valid.size else 0.0
    return np.full(n,length,dtype=float)


def segment_inertia_tensor(segment: str, mass: float, length: float) -> np.ndarray:
    """Local, COM-centred inertia tensor of one segment (kg m^2).

    Radius-of-gyration model, not a solid-geometry integration:

        I = mass * (k * length)^2

    with k the segment's dimensionless Plug-in Gait radius-of-gyration ratio
    (VICON_GYRATION_RATIO) and length the model-matched anatomical length
    from segment_anatomical_length_series(). Because mass, k and length are
    all constant within a trial, so is the tensor - the segment is rigid, and
    only its orientation changes from frame to frame.

    The tensor is diagonal in the segment's local basis, where local X is the
    longitudinal (proximal-distal) axis built by rotation_matrices_from_oalp().
    Ordinary segments receive ZERO longitudinal inertia, so rotation about the
    long axis contributes nothing to the spin term. Head and Pelvis receive the
    same value on all three axes, following the Plug-in Gait convention adopted
    in this project.

    Args:
        segment: Segment name, with or without the L_/R_ prefix.
        mass: Segment mass, in kg.
        length: Segment length, in meters.

    Returns:
        3x3 diagonal inertia tensor in the segment's local basis (kg*m^2).

    Raises:
        NotImplementedError: If INERTIA_MODEL_NAME is set to any model other
            than the Vicon/Winter one implemented here.
    """
    if not INERTIA_MODEL_NAME.startswith("Vicon"):
        raise NotImplementedError(
            f"Only the Vicon Plug-in Gait / Winter-Dempster inertia model is "
            f"implemented here; INERTIA_MODEL_NAME is {INERTIA_MODEL_NAME!r}. "
            f"Supporting another model requires replacing BOTH "
            f"VICON_GYRATION_RATIO and the landmark definitions in "
            f"segment_anatomical_length_series(): each radius of gyration is "
            f"normalized by its own model's segment length, so changing this "
            f"constant alone would silently pair one model's lengths with "
            f"another model's ratios."
        )
    base = _base_segment(segment)
    transverse = mass * (VICON_GYRATION_RATIO[base] * length) ** 2
    longitudinal = transverse if base in ("Head", "Pelvis") else 0.0
    return np.diag([longitudinal, transverse, transverse])


def angular_velocity_from_rotation(rotations: np.ndarray, valid: np.ndarray, dt: float = DT) -> np.ndarray:
    """Differentiates a rotation matrix series to obtain angular velocity.

    Steps:
      1. R_dot(t) ~= (R(t+1) - R(t-1)) / (2*dt)  (central difference).
      2. W(t) = R(t)^T @ R_dot(t) is theoretically skew-symmetric;
         this is enforced via (W - W^T)/2 to absorb small numerical
         noise.
      3. The vector w is extracted from the skew-symmetric matrix W
         (shape [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]).

    Args:
        rotations: (n_frames, 3, 3) rotation matrices.
        valid: Boolean array flagging which frames have a usable
            rotation matrix.
        dt: Time step between frames, in seconds.

    Returns:
        (n_frames, 3) angular velocity array, in rad/s, in the
        segment's LOCAL reference frame (the same frame in which the
        diagonal inertia tensor I_i is defined). Frames without a
        valid rotation matrix (edges of the series or missing O/A/P
        points) are filled by linear interpolation, or zero if there
        is no valid neighbor at all.
    """
    n = rotations.shape[0]
    omega = np.full((n, 3), np.nan)

    for i in range(1, n - 1):
        if valid[i - 1] and valid[i] and valid[i + 1]:
            r_dot = (rotations[i + 1] - rotations[i - 1]) / (2.0 * dt)
            skew = rotations[i].T @ r_dot
            skew = (skew - skew.T) / 2.0  # enforce skew-symmetry (reduces numerical noise)
            omega[i] = [skew[2, 1], skew[0, 2], skew[1, 0]]

    omega_df = pd.DataFrame(omega).interpolate(limit_direction="both")
    if omega_df.isna().any().any():
        omega_df = omega_df.fillna(0.0)
    return omega_df.to_numpy(dtype=float)


def angular_velocity_from_marker_angles(
    markers: pd.DataFrame,
    index: pd.Index,
    segment: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Computes a segment's absolute angular velocity (Method B).

    Reconstructs the segment's rotation from the O/A/P technical
    points (rotation_matrices_from_oalp) and differentiates that
    rotation over time (angular_velocity_from_rotation), then filters
    the result (6 Hz Butterworth), same as the other kinematic series
    in this script.

    Kept with the same name/signature as an earlier, angle-based
    implementation so the rest of the pipeline (process_trial)
    requires no further changes.

    Args:
        markers: Trial's markers DataFrame.
        index: Frame index to compute for.
        segment: Segment name.

    Returns:
        Tuple (angular_velocity_df, source):
          - angular_velocity_df: DataFrame (n_frames x 3) with columns
            ANGVEL_<segment>_X/Y/Z, in rad/s.
          - source: List naming the data source used (e.g.,
            ["LFE_OALP"]), or an empty list if the O/A/P points were
            not available/valid for this segment - in that case
            angular_velocity_df is zero everywhere.
    """
    out_cols = [f"ANGVEL_{segment}_{axis}" for axis in AXIS_NAMES]
    try:
        rotations, valid = rotation_matrices_from_oalp(markers, index, segment)
    except KeyError:
        zeros = pd.DataFrame(np.zeros((len(index), 3)), index=index, columns=out_cols)
        return zeros, []

    if valid.sum() < 3:
        zeros = pd.DataFrame(np.zeros((len(index), 3)), index=index, columns=out_cols)
        return zeros, []

    angular_velocity = angular_velocity_from_rotation(rotations, valid)
    angular_velocity = safe_filter(angular_velocity)
    source = [f"{SEGMENT_OALP_PREFIX[segment]}_OALP"]
    return pd.DataFrame(angular_velocity, index=index, columns=out_cols), source


def segment_rotation_series(markers: pd.DataFrame, index: pd.Index, segment: str) -> np.ndarray:
    """Returns a gap-filled rotation matrix series for one segment.

    Used to bring the spin term (I_i @ w_i), computed in the segment's
    local reference frame, back to the global frame before summing
    with the orbital term - see compute_wbam_frame.

    Args:
        markers: Trial's markers DataFrame.
        index: Frame index to compute for.
        segment: Segment name.

    Returns:
        (n_frames, 3, 3) rotation matrices, with gaps filled by
        interpolation (same logic used in numeric_frame for
        positions), or a fallback of identity matrices if there is no
        valid O/A/P point at all.
    """
    n = len(index)
    try:
        rotations, valid = rotation_matrices_from_oalp(markers, index, segment)
    except KeyError:
        return np.tile(np.eye(3), (n, 1, 1))

    if valid.sum() == 0:
        return np.tile(np.eye(3), (n, 1, 1))

    flat = pd.DataFrame(rotations.reshape(n, 9))
    flat = flat.interpolate(limit_direction="both").ffill().bfill()
    filled = flat.to_numpy(dtype=float).reshape(n, 3, 3)
    return filled


def compute_wbam_frame(
    coms: list[np.ndarray],
    vels: list[np.ndarray],
    angvels: list[np.ndarray],
    inertias: list[np.ndarray],
    masses: list[float],
    body_com: np.ndarray,
    body_velocity: np.ndarray,
    rotations: list[np.ndarray],
) -> np.ndarray:
    """Sums the orbital and spin components of WBAM at a single frame.

        L = sum_i [ R_i . (I_i . w_i) + m_i * (r_i - r_COM) x (v_i - v_COM) ]

    The orbital term (r_i, v_i, r_COM, v_COM) is entirely in the
    GLOBAL (laboratory) reference frame. I_i and w_i, however, are
    defined in each segment's LOCAL frame (I_i is diagonal only in
    those axes; w_i comes from Method B in the local frame - see
    angular_velocity_from_rotation). Without rotating I_i @ w_i back
    to the global frame via R_i(t) (the same rotation matrix used to
    compute w_i), we would be summing vectors expressed in different
    bases - an error that, tested on real data, reached >100%
    difference in the direction/magnitude of the spin term.

    Args:
        coms: List of each segment's COM position (m).
        vels: List of each segment's linear velocity (m/s).
        angvels: List of each segment's local angular velocity (rad/s).
        inertias: List of each segment's 3x3 inertia tensor (kg*m^2).
        masses: List of each segment's mass (kg).
        body_com: Whole-body COM position (m).
        body_velocity: Whole-body COM velocity (m/s).
        rotations: List of each segment's rotation matrix (local ->
            global).

    Returns:
        3-element array with the whole-body angular momentum at this
        frame (kg*m^2/s).
    """
    wbam = np.zeros(3)
    for r, v, w, inertia, mass, rotation in zip(coms, vels, angvels, inertias, masses, rotations):
        r_rel = r - body_com
        v_rel = v - body_velocity
        spin_global = rotation @ (inertia @ w)
        wbam += spin_global + mass * np.cross(r_rel, v_rel)
    return wbam


def normalize_wbam(wbam: np.ndarray, body_velocity: np.ndarray, mean_com_height: float) -> np.ndarray:
    """Non-dimensionalizes the WBAM time series.

    Args:
        wbam: Raw WBAM time series, shape (n_frames, 3).
        body_velocity: Whole-body COM velocity time series, shape
            (n_frames, 3).
        mean_com_height: Mean COM height, in meters.

    Returns:
        Normalized (dimensionless) WBAM time series, same shape as
        `wbam` - divided by (subject_mass * |v_COM(t)| *
        mean_com_height), frame by frame. Frames with non-finite or
        near-zero COM speed remain NaN. Cycles containing those frames
        are rejected before interpolation instead of being exported as
        artificial zeros or extreme finite values.
    """
    com_speed = np.linalg.norm(body_velocity, axis=1)
    denominator = SUBJECT_MASS_KG * com_speed * mean_com_height
    invalid = ~np.isfinite(com_speed) | (com_speed < MIN_COM_SPEED_M_S)
    denominator[invalid] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        return wbam / denominator[:, np.newaxis]


def validated_cycles(
    cycles: list[tuple[int, int]],
    body_com_source_valid: np.ndarray,
    body_speed: np.ndarray,
    frame_index: pd.Index,
    task: str,
    trial: str,
    side: str,
) -> list[tuple[int, int]]:
    """Rejects gait cycles that cannot be normalized reliably.

    A cycle is retained only when every original (pre-interpolation)
    whole-body COM sample is present and every filtered COM-speed sample is
    finite and at least ``MIN_COM_SPEED_M_S``. This prevents long leading or
    trailing gaps from being backward/forward filled into a nearly stationary
    COM trajectory and then magnified by framewise normalization.
    """
    accepted: list[tuple[int, int]] = []
    for cycle_number, (start, end) in enumerate(cycles, start=1):
        if end - start < 2:
            print(
                f"Excluding {task}{trial} {side} cycle {cycle_number} "
                f"({start}-{end}): fewer than 3 frames."
            )
            continue

        start_pos = frame_index.get_loc(start)
        end_pos = frame_index.get_loc(end)
        cycle_validity = body_com_source_valid[start_pos : end_pos + 1]
        cycle_speed = body_speed[start_pos : end_pos + 1]
        missing_frames = int(np.count_nonzero(~cycle_validity))
        finite_speed = cycle_speed[np.isfinite(cycle_speed)]
        min_speed = float(np.min(finite_speed)) if finite_speed.size else np.nan

        reasons: list[str] = []
        if missing_frames:
            reasons.append(f"{missing_frames} source COM frame(s) missing")
        if finite_speed.size != cycle_speed.size:
            reasons.append("non-finite COM speed")
        if not np.isfinite(min_speed) or min_speed < MIN_COM_SPEED_M_S:
            reasons.append(
                f"minimum COM speed {min_speed:.6g} m/s is below "
                f"{MIN_COM_SPEED_M_S:g} m/s"
            )

        if reasons:
            print(
                f"Excluding {task}{trial} {side} cycle {cycle_number} "
                f"({start}-{end}): {'; '.join(reasons)}."
            )
            continue
        accepted.append((start, end))

    return accepted


def interpolate_cycle(series: np.ndarray, start_frame: int, end_frame: int, index: pd.Index) -> np.ndarray:
    """Resamples one gait cycle to 100 points spanning 0-100% of the cycle.

    Allows comparing cycles of different duration (different
    trials/tasks) point by point, on the same "percent of cycle"
    scale.

    Args:
        series: Time series to resample.
        start_frame: First frame of the cycle.
        end_frame: Last frame of the cycle.
        index: Frame index of `series`.

    Returns:
        (100, ...) array, linearly interpolated onto PERCENT_CYCLE.
    """
    start_pos = index.get_loc(start_frame)
    end_pos = index.get_loc(end_frame)
    cycle = series[start_pos : end_pos + 1]
    frames = np.arange(start_frame, end_frame + 1)
    percent = (frames - start_frame) / (end_frame - start_frame) * 100.0
    f = interp1d(percent, cycle, axis=0, kind="linear", fill_value="extrapolate")
    return f(PERCENT_CYCLE)


def compute_trial_wbam(
    source_dir: Path,
    task: str,
    trial: str,
    mean_com_height: float,
    cycles_override: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None = None,
) -> TrialResult | None:
    """Processes a single trial end to end.

    Steps:
      1. Loads markers/frames/events (load_trial).
      2. Filters and converts to meters the whole-body COM position and
         the position of each available segment, and differentiates
         the linear velocities.
      3. Computes w_i for each segment using Method B
         (angular_velocity_from_marker_angles).
      4. Builds the WBAM frame by frame (compute_wbam_frame) and
         normalizes it (normalize_wbam).
      5. Extracts and resamples the right and left gait cycles
         (interpolate_cycle) - the left ones are flipped (*-1) to
         match the sign convention of the right ones.

    Args:
        source_dir: Participant folder.
        task: Task code.
        trial: Trial number.
        mean_com_height: Mean COM height, in meters.
        cycles_override: Optional (right_cycles, left_cycles) pair that
            replaces the cycles read from the events file. This is how
            process_trial() restricts the whole calculation to the single
            protocol-defined stride: every downstream step - validation,
            interpolation, cycle storage and all later summaries - then
            operates on exactly that stride. When None (the default) the
            events file is read as usual and every complete cycle is kept.

    Returns:
        A TrialResult on success, or None (printing the reason) if the
        trial cannot be processed: missing files, missing whole-body
        COM columns, or no valid gait cycle found.
    """
    try:
        markers, _frames, events_path, start_frame, end_frame = load_trial(source_dir, task, trial)
    except Exception as exc:
        print(f"Skipping {task}{trial}: {exc}")
        return None

    frame_index = pd.RangeIndex(start_frame, end_frame + 1)
    body_cols = [f"CentreOfMass_{axis}" for axis in AXIS_NAMES]
    if not all(col in markers.columns for col in body_cols):
        print(f"Skipping {task}{trial}: missing body COM columns")
        return None

    body_com_source = markers.loc[frame_index, body_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    body_com_source_valid = np.isfinite(body_com_source.to_numpy(dtype=float)).all(axis=1)
    body_com_raw = numeric_frame(markers, body_cols, frame_index).to_numpy(dtype=float) * MM_TO_M
    body_com = pd.DataFrame(
        safe_filter(body_com_raw),
        index=frame_index,
        columns=body_cols,
    )
    body_velocity = compute_linear_velocity(body_com.to_numpy(dtype=float))
    body_speed = np.linalg.norm(body_velocity, axis=1)

    segment_coms: dict[str, pd.DataFrame] = {}
    segment_velocities: dict[str, np.ndarray] = {}
    segment_angvels: dict[str, pd.DataFrame] = {}
    segment_rotations: dict[str, np.ndarray] = {}
    segment_lengths: dict[str, np.ndarray] = {}
    angular_velocity_max_abs: dict[str, float] = {}

    for segment in available_segments(markers):
        prefix = SEGMENT_COM_PREFIX[segment]
        cols = [f"{prefix}_{axis}" for axis in AXIS_NAMES]
        com_raw = numeric_frame(markers, cols, frame_index).to_numpy(dtype=float) * MM_TO_M
        com_filtered = safe_filter(com_raw)
        segment_coms[segment] = pd.DataFrame(com_filtered, index=frame_index, columns=cols)
        segment_velocities[segment] = compute_linear_velocity(com_filtered)

        angvel_df, angle_sources = angular_velocity_from_marker_angles(markers, frame_index, segment)
        segment_angvels[segment] = angvel_df
        segment_rotations[segment] = segment_rotation_series(markers, frame_index, segment)
        angular_velocity_max_abs[segment] = float(np.nanmax(np.abs(angvel_df.to_numpy(dtype=float))))
        if not angle_sources:
            print(f"Warning {task}{trial}: missing/invalid O/A/P points for {segment}; angular velocity set to zero.")

        mass = SEGMENT_MASS_RATIO[segment] * SUBJECT_MASS_KG
        segment_lengths[segment] = segment_anatomical_length_series(
            markers, frame_index, segment,
            segment_coms[segment].to_numpy(dtype=float), segment_rotations[segment]
        )

    wbam = np.zeros((len(frame_index), 3))
    for row, frame in enumerate(frame_index):
        coms: list[np.ndarray] = []
        vels: list[np.ndarray] = []
        angvels: list[np.ndarray] = []
        inertias: list[np.ndarray] = []
        masses: list[float] = []
        rotations: list[np.ndarray] = []

        for segment in segment_coms:
            mass = SEGMENT_MASS_RATIO[segment] * SUBJECT_MASS_KG
            length_series = segment_lengths[segment]
            coms.append(segment_coms[segment].loc[frame].to_numpy(dtype=float))
            vels.append(segment_velocities[segment][row])
            angvels.append(segment_angvels[segment].loc[frame].to_numpy(dtype=float))
            inertias.append(segment_inertia_tensor(segment, mass, length_series[row]))
            masses.append(mass)
            rotations.append(segment_rotations[segment][row])

        wbam[row] = compute_wbam_frame(
            coms,
            vels,
            angvels,
            inertias,
            masses,
            body_com.loc[frame].to_numpy(dtype=float),
            body_velocity[row],
            rotations,
        )

    wbam_normalized = normalize_wbam(wbam, body_velocity, mean_com_height)
    if cycles_override is not None:
        right_cycles, left_cycles = cycles_override
    else:
        right_cycles, left_cycles = gait_cycles_from_events(events_path, start_frame, end_frame)

    right_cycles = validated_cycles(
        right_cycles,
        body_com_source_valid,
        body_speed,
        frame_index,
        task,
        trial,
        "right",
    )
    left_cycles = validated_cycles(
        left_cycles,
        body_com_source_valid,
        body_speed,
        frame_index,
        task,
        trial,
        "left",
    )

    right_interp = [
        interpolate_cycle(wbam_normalized, start, end, frame_index)
        for start, end in right_cycles
    ]
    left_interp_flipped = [
        -1.0 * interpolate_cycle(wbam_normalized, start, end, frame_index)
        for start, end in left_cycles
    ]

    if not right_interp and not left_interp_flipped:
        print(f"Skipping {task}{trial}: no valid gait cycles")
        return None

    return TrialResult(
        task=task,
        trial=trial,
        right_cycles=right_interp,
        left_cycles_flipped=left_interp_flipped,
        angular_velocity_max_abs=angular_velocity_max_abs,
    )


def median_or_none(cycles: list[np.ndarray]) -> np.ndarray | None:
    """Computes the point-by-point median of a list of resampled cycles.

    Args:
        cycles: List of (100, 3) arrays, already resampled to 0-100%
            of the gait cycle.

    Returns:
        (100, 3) median array, or None if `cycles` is empty (e.g., a
        trial with no cycles on that side).
    """
    if not cycles:
        return None
    return np.nanmedian(np.stack(cycles, axis=0), axis=0)


def representative_trial_series(result: TrialResult) -> tuple[np.ndarray | None, str]:
    """Chooses the series that best represents one trial.

    Rule, depending on the task:
      - "DOVesq" (left vertical obstacle circumvention): uses the
        median of the left cycles (already flipped).
      - "CT" (beam walking): uses the average between the right
        median and the flipped left median, when both exist.
      - Other tasks: uses the right median if it exists, otherwise the
        flipped left one.

    Args:
        result: TrialResult to summarize.

    Returns:
        Tuple (series, source_label): `series` is a (100, 3) array or
        None; `source_label` is one of "right", "left_flipped",
        "right_left_average" or "none" (no valid cycle found).
    """
    right = median_or_none(result.right_cycles)
    left = median_or_none(result.left_cycles_flipped)

    if result.task == "DOVesq" and left is not None:
        return left, "left_flipped"
    if result.task == "CT" and right is not None and left is not None:
        return (right + left) / 2.0, "right_left_average"
    if right is not None:
        return right, "right"
    if left is not None:
        return left, "left_flipped"
    return None, "none"


# =============================================================================
# PART 3 - STRIDE SELECTION
# Obstacle centroid, task-specific rules, CoM quality gate, fallback.
# =============================================================================


@dataclass(frozen=True)
class StrideQuality:
    """Whole-body CoM checks for one candidate stride."""

    side: str
    stride_number: int
    start_frame: int
    end_frame: int
    missing_com_frames: int
    nonfinite_speed_frames: int
    minimum_com_speed_m_s: float
    valid: bool
    reasons: tuple[str, ...]

    def label(self) -> str:
        minimum = (
            f"{self.minimum_com_speed_m_s:.6g} m/s"
            if np.isfinite(self.minimum_com_speed_m_s)
            else "not finite"
        )
        return (
            f"{self.side} stride {self.stride_number}: frames "
            f"{self.start_frame}-{self.end_frame}; minimum CoM speed {minimum}"
        )


@dataclass
class SelectedStrideResult:
    """WBAM result plus an audit trail for the selected stride."""

    task: str
    trial: str
    right_cycles: list[np.ndarray]
    left_cycles_flipped: list[np.ndarray]
    angular_velocity_max_abs: dict[str, float]
    prescribed_side: str
    prescribed_stride_number: int
    selected_side: str
    selected_stride_number: int
    selected_start_frame: int
    selected_end_frame: int
    selection_rule: str
    fallback_used: bool
    obstacle_centroid_y_mm: float | None
    selected_minimum_com_speed_m_s: float


FallbackSelector = Callable[[str, str, StrideQuality, list[StrideQuality]], StrideQuality | None]


def assess_stride_com_quality(
    cycle: tuple[int, int],
    side: str,
    stride_number: int,
    body_com_source_valid: np.ndarray,
    body_speed: np.ndarray,
    frame_index: pd.Index,
) -> StrideQuality:
    """Check whether one indicated stride can be normalized safely.

    The check intentionally uses the original whole-body CoM validity mask,
    before interpolation/filling, and the same filtered CoM speed that enters
    ``mass * speed * mean CoM height`` in the normalization denominator.
    """
    start, end = cycle
    reasons: list[str] = []
    if end - start < 2:
        reasons.append("fewer than 3 frames")

    try:
        start_pos = int(frame_index.get_loc(start))
        end_pos = int(frame_index.get_loc(end))
    except KeyError:
        reasons.append("stride lies outside the trial frame range")
        return StrideQuality(
            side, stride_number, start, end, 0, 0, np.nan, False, tuple(reasons)
        )

    validity = body_com_source_valid[start_pos : end_pos + 1]
    speed = body_speed[start_pos : end_pos + 1]
    missing = int(np.count_nonzero(~validity))
    nonfinite = int(np.count_nonzero(~np.isfinite(speed)))
    finite_speed = speed[np.isfinite(speed)]
    minimum = float(np.min(finite_speed)) if finite_speed.size else np.nan

    if missing:
        reasons.append(f"{missing} original whole-body CoM frame(s) missing")
    if nonfinite:
        reasons.append(f"{nonfinite} non-finite CoM-speed frame(s)")
    if not np.isfinite(minimum) or minimum < MIN_COM_SPEED_M_S:
        reasons.append(
            f"minimum CoM speed {minimum:.6g} m/s is below "
            f"{MIN_COM_SPEED_M_S:g} m/s"
        )

    return StrideQuality(
        side=side,
        stride_number=stride_number,
        start_frame=start,
        end_frame=end,
        missing_com_frames=missing,
        nonfinite_speed_frames=nonfinite,
        minimum_com_speed_m_s=minimum,
        valid=not reasons,
        reasons=tuple(reasons),
    )


def _case_insensitive_column(markers: pd.DataFrame, requested: str) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in markers.columns}
    return lookup.get(requested.casefold())


def obstacle_centroid_mm(markers: pd.DataFrame, frame_index: pd.Index) -> np.ndarray:
    """Return the time-median centroid of the four obstacle markers."""
    marker_series: list[np.ndarray] = []
    for marker in OBSTACLE_MARKERS:
        columns = [
            _case_insensitive_column(markers, f"{marker}_{axis}")
            for axis in AXIS_NAMES
        ]
        if any(column is None for column in columns):
            continue
        values = markers.loc[frame_index, columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        marker_series.append(values)

    if not marker_series:
        raise ValueError(
            "No obstacle markers were found; expected obstaculo1-obstaculo4."
        )

    with np.errstate(invalid="ignore"):
        frame_centroids = np.nanmean(np.stack(marker_series, axis=0), axis=0)
        centroid = np.nanmedian(frame_centroids, axis=0)
    if not np.isfinite(centroid).all():
        raise ValueError("The obstacle-marker centroid is not finite.")
    return centroid


def foot_progression_series_mm(
    markers: pd.DataFrame, frame_index: pd.Index, side: str
) -> np.ndarray:
    """Return the selected foot's CoM progression coordinate in millimetres."""
    prefix = "RightFootCOM" if side == "right" else "LeftFootCOM"
    column = _case_insensitive_column(markers, f"{prefix}_{PROGRESSION_AXIS}")
    if column is None:
        raise ValueError(f"Missing {prefix}_{PROGRESSION_AXIS} for obstacle selection.")
    return pd.to_numeric(
        markers.loc[frame_index, column], errors="coerce"
    ).to_numpy(dtype=float)


def central_stride_number(
    cycles: list[tuple[int, int]], start_frame: int, end_frame: int
) -> int:
    """Return the 1-based stride nearest the temporal centre of the trial."""
    if not cycles:
        raise ValueError("No complete candidate strides were found.")
    trial_midpoint = (start_frame + end_frame) / 2.0
    return min(
        range(len(cycles)),
        key=lambda index: (
            abs((cycles[index][0] + cycles[index][1]) / 2.0 - trial_midpoint),
            index,
        ),
    ) + 1


def obstacle_crossing_stride_number(
    cycles: list[tuple[int, int]],
    foot_progression_mm: np.ndarray,
    obstacle_progression_mm: float,
    frame_index: pd.Index,
) -> int:
    """Return the right stride whose two Foot Strikes bracket the obstacle.

    Each candidate is already bounded by two consecutive Right Foot Strike
    events from the _events.csv file. A TOH stride qualifies only when the
    right-foot CoM at one boundary is before the obstacle-centroid Y plane
    and the right-foot CoM at the other boundary is after it. Strides lying
    completely before or completely after the obstacle are rejected.
    """
    if not cycles:
        raise ValueError("No complete candidate strides were found.")

    qualifying: list[tuple[float, int]] = []
    for index, (start, end) in enumerate(cycles):
        start_pos = int(frame_index.get_loc(start))
        end_pos = int(frame_index.get_loc(end))
        first_strike_y = float(foot_progression_mm[start_pos])
        second_strike_y = float(foot_progression_mm[end_pos])
        if not np.isfinite(first_strike_y) or not np.isfinite(second_strike_y):
            continue
        first_delta = first_strike_y - obstacle_progression_mm
        second_delta = second_strike_y - obstacle_progression_mm
        brackets_object = first_delta == 0.0 or second_delta == 0.0 or (
            np.signbit(first_delta) != np.signbit(second_delta)
        )
        if brackets_object:
            spatial_midpoint_error = abs(
                (first_strike_y + second_strike_y) / 2.0
                - obstacle_progression_mm
            )
            qualifying.append((spatial_midpoint_error, index))

    if not qualifying:
        raise ValueError(
            "No right-foot stride has consecutive Foot Strike positions on "
            "opposite sides of the obstacle-centroid Y plane."
        )
    return min(qualifying)[1] + 1


def side_neutral_obstacle_crossing_stride(
    right_cycles: list[tuple[int, int]],
    left_cycles: list[tuple[int, int]],
    markers: pd.DataFrame,
    frame_index: pd.Index,
    obstacle_progression_mm: float,
) -> tuple[str, int]:
    """Select a DOV stride from both sides using the obstacle-centroid plane.

    Every candidate remains bounded by consecutive same-side Foot Strike
    events. A candidate is preferred when its corresponding foot CoM crosses
    the obstacle-centroid Y plane. If both sides cross it, the crossing
    closest to that stride's temporal midpoint wins; minimum foot-to-plane
    distance and stable right-before-left ordering are deterministic
    tie-breakers.
    """
    candidates: list[tuple[int, float, float, int, str, int]] = []
    for side_order, (side, cycles) in enumerate(
        (("right", right_cycles), ("left", left_cycles))
    ):
        foot_y = foot_progression_series_mm(markers, frame_index, side)
        for stride_index, (start, end) in enumerate(cycles):
            start_pos = int(frame_index.get_loc(start))
            end_pos = int(frame_index.get_loc(end))
            values = foot_y[start_pos + 1 : end_pos]
            delta = values - obstacle_progression_mm
            finite_positions = np.flatnonzero(np.isfinite(delta))
            if finite_positions.size:
                finite_delta = delta[finite_positions]
                nearest_local = int(
                    finite_positions[int(np.argmin(np.abs(finite_delta)))]
                )
                crossing_frame = int(frame_index[start_pos + 1 + nearest_local])
                crosses = bool(
                    np.min(finite_delta) <= 0.0 <= np.max(finite_delta)
                )
                distance = float(np.min(np.abs(finite_delta)))
                midpoint_distance = abs(crossing_frame - (start + end) / 2.0)
            else:
                crosses = False
                distance = np.inf
                midpoint_distance = np.inf
            candidates.append(
                (
                    0 if crosses else 1,
                    midpoint_distance,
                    distance,
                    side_order,
                    side,
                    stride_index + 1,
                )
            )

    if not candidates:
        raise ValueError("No complete right- or left-foot DOV strides were found.")
    selected = min(candidates)
    return selected[4], selected[5]


def prescribed_stride(
    task: str,
    right_cycles: list[tuple[int, int]],
    left_cycles: list[tuple[int, int]],
    markers: pd.DataFrame,
    frame_index: pd.Index,
    start_frame: int,
    end_frame: int,
) -> tuple[str, int, str, float | None]:
    """Apply the task-specific experimental stride-selection protocol."""
    if task in {"UW", "BSR", "CT"}:
        number = central_stride_number(right_cycles, start_frame, end_frame)
        return "right", number, "temporally central right-foot stride", None

    if task == "TOH":
        side = "right"
        cycles = right_cycles
        centroid = obstacle_centroid_mm(markers, frame_index)
        foot_y = foot_progression_series_mm(markers, frame_index, side)
        number = obstacle_crossing_stride_number(
            cycles, foot_y, float(centroid[1]), frame_index
        )
        rule = (
            "right-foot stride whose consecutive Foot Strike positions "
            f"bracket the {PROGRESSION_AXIS} plane through the four-marker "
            "obstacle centroid"
        )
        return side, number, rule, float(centroid[1])

    if task in {"DOVdir", "DOVesq"}:
        centroid = obstacle_centroid_mm(markers, frame_index)
        side, number = side_neutral_obstacle_crossing_stride(
            right_cycles,
            left_cycles,
            markers,
            frame_index,
            float(centroid[1]),
        )
        rule = (
            "right-or-left stride whose corresponding foot CoM crosses the "
            f"{PROGRESSION_AXIS} plane through the four-marker obstacle "
            "centroid closest to the stride midpoint"
        )
        return side, number, rule, float(centroid[1])

    raise ValueError(f"No stride-selection rule is defined for task {task!r}.")


def choose_alternative_stride_popup(
    task: str,
    trial: str,
    prescribed: StrideQuality,
    alternatives: list[StrideQuality],
) -> StrideQuality | None:
    """Ask the operator to choose a valid alternative using a popup."""
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    problem = "; ".join(prescribed.reasons)
    if not alternatives:
        messagebox.showerror(
            "WBAM - No valid alternative stride",
            f"{task}{trial}: the prescribed {prescribed.label()} is invalid:\n"
            f"{problem}.\n\nNo other valid {prescribed.side}-foot stride is available. "
            "This trial will be skipped.",
            parent=root,
        )
        root.destroy()
        return None

    options = "\n".join(
        f"{option}: {quality.label()}"
        for option, quality in enumerate(alternatives, start=1)
    )
    prompt = (
        f"{task}{trial}: the prescribed {prescribed.label()} cannot be used.\n"
        f"Reason: {problem}.\n\nChoose another valid {prescribed.side}-foot stride:\n"
        f"{options}"
    )
    choice = simpledialog.askinteger(
        "WBAM - Choose an alternative stride",
        prompt,
        minvalue=1,
        maxvalue=len(alternatives),
        parent=root,
    )
    root.destroy()
    if choice is None:
        return None
    return alternatives[choice - 1]


def resolve_stride_with_quality_check(
    task: str,
    trial: str,
    prescribed_side: str,
    prescribed_number: int,
    right_cycles: list[tuple[int, int]],
    left_cycles: list[tuple[int, int]],
    body_com_source_valid: np.ndarray,
    body_speed: np.ndarray,
    frame_index: pd.Index,
    fallback_selector: FallbackSelector | None = None,
) -> tuple[StrideQuality | None, bool]:
    """Validate the prescribed stride and obtain an operator fallback if needed."""
    cycles = left_cycles if prescribed_side == "left" else right_cycles
    qualities = [
        assess_stride_com_quality(
            cycle,
            prescribed_side,
            number,
            body_com_source_valid,
            body_speed,
            frame_index,
        )
        for number, cycle in enumerate(cycles, start=1)
    ]
    if not 1 <= prescribed_number <= len(qualities):
        raise ValueError(
            f"Prescribed {prescribed_side} stride {prescribed_number} does not exist."
        )

    prescribed = qualities[prescribed_number - 1]
    if prescribed.valid:
        return prescribed, False

    alternatives = [
        quality
        for quality in qualities
        if quality.stride_number != prescribed_number and quality.valid
    ]
    selector = fallback_selector or choose_alternative_stride_popup
    selected = selector(task, trial, prescribed, alternatives)
    return selected, selected is not None


def process_trial(
    source_dir: Path,
    task: str,
    trial: str,
    mean_com_height: float,
    fallback_selector: FallbackSelector | None = None,
) -> SelectedStrideResult | None:
    """Process only the protocol-defined stride for one trial.

    Reads the trial, applies the task-specific selection rule, validates the
    resulting stride against the original whole-body CoM samples and the CoM
    speed used in the normalization denominator, then computes WBAM for that
    stride alone via compute_trial_wbam().

    Returns:
        A SelectedStrideResult, or None (printing the reason) when the trial
        cannot be processed or no valid stride could be selected.
    """
    try:
        markers, _frames, events_path, start_frame, end_frame = load_trial(
            source_dir, task, trial
        )
    except Exception as exc:
        print(f"Skipping {task}{trial}: {exc}")
        return None

    frame_index = pd.RangeIndex(start_frame, end_frame + 1)
    body_cols = [f"CentreOfMass_{axis}" for axis in AXIS_NAMES]
    if not all(column in markers.columns for column in body_cols):
        print(f"Skipping {task}{trial}: missing body COM columns")
        return None

    body_source = markers.loc[frame_index, body_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    body_source_valid = np.isfinite(body_source.to_numpy(dtype=float)).all(axis=1)
    body_raw = (
        numeric_frame(markers, body_cols, frame_index).to_numpy(dtype=float)
        * MM_TO_M
    )
    body_filtered = safe_filter(body_raw)
    body_velocity = compute_linear_velocity(body_filtered)
    body_speed = np.linalg.norm(body_velocity, axis=1)

    right_cycles, left_cycles = gait_cycles_from_events(
        events_path, start_frame, end_frame
    )
    try:
        side, prescribed_number, rule, obstacle_y = prescribed_stride(
            task,
            right_cycles,
            left_cycles,
            markers,
            frame_index,
            start_frame,
            end_frame,
        )
        selected, fallback_used = resolve_stride_with_quality_check(
            task,
            trial,
            side,
            prescribed_number,
            right_cycles,
            left_cycles,
            body_source_valid,
            body_speed,
            frame_index,
            fallback_selector,
        )
    except Exception as exc:
        print(f"Skipping {task}{trial}: stride selection failed: {exc}")
        return None

    if selected is None:
        print(f"Skipping {task}{trial}: no alternative stride was selected")
        return None

    chosen_cycle = (selected.start_frame, selected.end_frame)
    print(
        f"Selected {task}{trial} {selected.label()} using rule: {rule}"
        + (" (operator fallback)" if fallback_used else "")
    )

    # Hand the chosen stride to the mechanics explicitly.  Everything
    # downstream - cycle validation, interpolation, cycle storage and all
    # later summaries - then operates on exactly one protocol-defined stride.
    cycles_override = (
        ([chosen_cycle], []) if selected.side == "right" else ([], [chosen_cycle])
    )
    calculated = compute_trial_wbam(
        source_dir, task, trial, mean_com_height, cycles_override
    )

    if calculated is None:
        return None

    return SelectedStrideResult(
        task=calculated.task,
        trial=calculated.trial,
        right_cycles=calculated.right_cycles,
        left_cycles_flipped=calculated.left_cycles_flipped,
        angular_velocity_max_abs=calculated.angular_velocity_max_abs,
        prescribed_side=side,
        prescribed_stride_number=prescribed_number,
        selected_side=selected.side,
        selected_stride_number=selected.stride_number,
        selected_start_frame=selected.start_frame,
        selected_end_frame=selected.end_frame,
        selection_rule=rule,
        fallback_used=fallback_used,
        obstacle_centroid_y_mm=obstacle_y,
        selected_minimum_com_speed_m_s=selected.minimum_com_speed_m_s,
    )


# =============================================================================
# PART 4 - EXPORTS AND FIGURES
# Summary CSV tables, figures, and the operator selection dialogs.
# =============================================================================


def save_csv_outputs(results: list[TrialResult], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Consolidates every trial's results into summary CSV tables.

    Writes 4 tables to output_dir/csv/:
      - individual_cycles_normalized.csv: one cycle per row (the most
        granular level - every right and flipped-left cycle from every
        trial, point by point along the gait cycle).
      - individual_trial_series_normalized.csv: one representative
        series per trial (see representative_trial_series).
      - angular_velocity_audit.csv: the peak |w| per segment/trial, to
        check whether the computed angular velocity is in a plausible
        range.
      - task_medians_normalized.csv: the median across trials, per
        task (used in the final figures).

    Args:
        results: List of TrialResult objects to consolidate.
        output_dir: Results folder to write the CSVs into.

    Returns:
        Tuple (cycles_df, trials_df, task_medians_df) - the angular
        velocity audit table is not returned, only saved to disk.
    """
    cycle_rows = []
    trial_rows = []
    angular_rows = []

    for result in results:
        for segment, max_abs in result.angular_velocity_max_abs.items():
            angular_rows.append(
                {
                    "task": result.task,
                    "trial": result.trial,
                    "segment": segment,
                    "max_abs_angular_velocity_rad_s": max_abs,
                }
            )

        side_cycles = [
            ("right", result.right_cycles),
            ("left_flipped", result.left_cycles_flipped),
        ]
        for side, cycles in side_cycles:
            for cycle_number, cycle in enumerate(cycles, start=1):
                for point, percent in enumerate(PERCENT_CYCLE):
                    cycle_rows.append(
                        {
                            "task": result.task,
                            "task_label": TASK_ABBREV[result.task],
                            "trial": result.trial,
                            "side": side,
                            "cycle": cycle_number,
                            "gait_cycle_percent": percent,
                            "L_X_sagittal": cycle[point, 0],
                            "L_Y_frontal": cycle[point, 1],
                            "L_Z_transverse": cycle[point, 2],
                        }
                    )

        representative, source = representative_trial_series(result)
        if representative is not None:
            for point, percent in enumerate(PERCENT_CYCLE):
                trial_rows.append(
                    {
                        "task": result.task,
                        "task_label": TASK_ABBREV[result.task],
                        "task_name": TASK_FULL_NAMES[result.task],
                        "trial": result.trial,
                        "series_source": source,
                        "gait_cycle_percent": percent,
                        "L_X_sagittal": representative[point, 0],
                        "L_Y_frontal": representative[point, 1],
                        "L_Z_transverse": representative[point, 2],
                    }
                )

    cycles_df = pd.DataFrame(cycle_rows)
    trials_df = pd.DataFrame(trial_rows)
    angular_df = pd.DataFrame(angular_rows)
    for output_frame in (cycles_df, trials_df, angular_df):
        output_frame.insert(0, "inertia_model", INERTIA_MODEL_NAME)

    csv_dir = output_dir / "csv"
    cycles_df.to_csv(csv_dir / "individual_cycles_normalized.csv", index=False)
    trials_df.to_csv(csv_dir / "individual_trial_series_normalized.csv", index=False)
    angular_df.to_csv(csv_dir / "angular_velocity_audit.csv", index=False)

    task_rows = []
    for task in TASKS:
        task_trial_values = []
        for trial in TRIALS:
            trial_data = trials_df[(trials_df["task"] == task) & (trials_df["trial"] == trial)]
            if trial_data.empty:
                continue
            task_trial_values.append(trial_data[["L_X_sagittal", "L_Y_frontal", "L_Z_transverse"]].to_numpy())
        if not task_trial_values:
            continue
        task_median = np.nanmedian(np.stack(task_trial_values, axis=0), axis=0)
        for point, percent in enumerate(PERCENT_CYCLE):
            task_rows.append(
                {
                    "task": task,
                    "task_label": TASK_ABBREV[task],
                    "task_name": TASK_FULL_NAMES[task],
                    "gait_cycle_percent": percent,
                    "median_L_X_sagittal": task_median[point, 0],
                    "median_L_Y_frontal": task_median[point, 1],
                    "median_L_Z_transverse": task_median[point, 2],
                }
            )

    task_medians_df = pd.DataFrame(task_rows)
    task_medians_df.insert(0, "inertia_model", INERTIA_MODEL_NAME)
    task_medians_df.to_csv(csv_dir / "task_medians_normalized.csv", index=False)

    return cycles_df, trials_df, task_medians_df


def task_median_array(task_medians_df: pd.DataFrame, task: str) -> np.ndarray:
    """Extracts one task's median WBAM series.

    Args:
        task_medians_df: Table produced by save_csv_outputs.
        task: Task code.

    Returns:
        (100, 3) array sorted by percent of cycle, or zeros if the
        task has no data (no trial processed successfully).
    """
    data = task_medians_df[task_medians_df["task"] == task].sort_values("gait_cycle_percent")
    if data.empty:
        return np.zeros((100, 3))
    return data[["median_L_X_sagittal", "median_L_Y_frontal", "median_L_Z_transverse"]].to_numpy()


def y_formatter(x: float, _pos: int) -> str:
    """Y-axis tick formatter used in the plots.

    Converts the normalized WBAM (dimensionless, order of magnitude
    ~1e-2) to "x100", so the plot shows values multiplied by 100
    (e.g., 0.03 -> "3"), with the "x10^-2" notation shown separately
    in the axis annotation.

    Args:
        x: Tick value.
        _pos: Tick position (unused; required by matplotlib's
            FuncFormatter signature).

    Returns:
        Formatted tick label.
    """
    return f"{x * 100:.0f}"


def style_axis(ax, plane: str, max_y: float, title: str | None = None) -> None:
    """Applies the standard formatting used in every WBAM-per-gait-cycle plot.

    Adds reference lines at y=0 and x=60% (typical stance/swing
    transition), sets the Y-axis limits and ticks, adds the scale
    annotation (x10^-2), and the "Stance"/"Swing" labels.

    Args:
        ax: Matplotlib Axes to format.
        plane: "X", "Y" or "Z" - controls the Y-axis tick spacing.
        max_y: Y-axis limit (the axis spans -max_y to +max_y).
        title: Optional subplot title.
    """
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(60, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(-max_y, max_y)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    tick_spacing = 0.03 if plane in {"Y", "X"} else 0.01
    if max_y / tick_spacing > 20:
        # Near-zero COM speed can create very large normalized values.
        # A fixed small tick interval would then generate hundreds of
        # thousands of ticks and can exhaust memory while saving figures.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, symmetric=True))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.annotate(r"$\times 10^{-2}$", xy=(-0.15, 1.02), xycoords="axes fraction", ha="left", va="bottom", fontsize=10)
    # y in axes-fraction coordinates (0-1), x in data units - this keeps
    # the label a fixed distance from the top border regardless of the
    # y-scale (which is now computed dynamically from the data).
    label_transform = ax.get_xaxis_transform()
    ax.text(30, 0.97, "Stance", transform=label_transform, ha="center", va="top", fontsize=11, fontstyle="italic")
    ax.text(80, 0.97, "Swing", transform=label_transform, ha="center", va="top", fontsize=11, fontstyle="italic")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=12)


def save_figure2(task_medians_df: pd.DataFrame, output_dir: Path) -> None:
    """Generates the main results figure: median WBAM curves plus amplitude bars.

    A 2x3 grid where the top row shows the median (normalized) WBAM
    series along the gait cycle for each plane (frontal/sagittal/
    transverse), one line per task; and the bottom row shows bars with
    the peak-to-peak amplitude of each task in that plane. The
    "DOVesq" task is plotted with the original signal dashed and the
    flipped one (used in the calculation) as a solid line, to make
    clear which version was used where.

    Args:
        task_medians_df: Table produced by save_csv_outputs.
        output_dir: Results folder to save the figure (PNG/SVG, in
            output_dir/figures/) and the amplitude table (in
            output_dir/csv/figure2_amplitudes.csv) into.
    """
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))
    plane_labels = ["Frontal Plane", "Sagittal Plane", "Transversal Plane"]
    planes = ["Y", "X", "Z"]
    axis_indices = {"X": 0, "Y": 1, "Z": 2}

    task_arrays = {task: task_median_array(task_medians_df, task) for task in TASKS}

    max_ys = []
    for plane in planes:
        axis = axis_indices[plane]
        max_y = max(float(np.nanmax(np.abs(task_arrays[task][:, axis]))) for task in TASKS)
        max_ys.append(max(max_y * 1.2, 0.0001))

    amplitudes = np.zeros((3, len(TASKS)))
    for task_idx, task in enumerate(TASKS):
        median_cycle = task_arrays[task]
        for axis in range(3):
            amplitudes[axis, task_idx] = np.nanmax(median_cycle[:, axis]) - np.nanmin(median_cycle[:, axis])

    amplitude_rows = []
    for task_idx, task in enumerate(TASKS):
        amplitude_rows.append(
            {
                "task": task,
                "task_label": TASK_ABBREV[task],
                "frontal_Y_peak_to_peak": amplitudes[1, task_idx],
                "sagittal_X_peak_to_peak": amplitudes[0, task_idx],
                "transverse_Z_peak_to_peak": amplitudes[2, task_idx],
            }
        )
    amplitude_df = pd.DataFrame(amplitude_rows)
    amplitude_df.insert(0, "inertia_model", INERTIA_MODEL_NAME)
    amplitude_df.to_csv(output_dir / "csv" / "figure2_amplitudes.csv", index=False)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for col, plane in enumerate(planes):
        ax = axs[0, col]
        axis = axis_indices[plane]
        for task_idx, task in enumerate(TASKS):
            plotted = task_arrays[task][:, axis]
            if task == "DOVesq":
                original_sign = -1.0 * plotted
                if plane == "X":
                    ax.plot(PERCENT_CYCLE, original_sign, color=colors[task_idx], linewidth=2, label=TASK_ABBREV[task])
                    ax.plot(PERCENT_CYCLE, plotted, color=colors[task_idx], linestyle="--", linewidth=2)
                else:
                    ax.plot(PERCENT_CYCLE, original_sign, color=colors[task_idx], linestyle="--", linewidth=2)
                    ax.plot(PERCENT_CYCLE, plotted, color=colors[task_idx], linewidth=2, label=TASK_ABBREV[task])
            else:
                ax.plot(
                    PERCENT_CYCLE,
                    plotted,
                    color=colors[task_idx],
                    linewidth=2,
                    label=TASK_ABBREV[task],
                )
        style_axis(ax, plane, max_ys[col], plane_labels[col])
        ax.set_xlabel("Gait Cycle (%)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Angular momentum (L)", fontweight="bold", fontsize=12)

    bar_planes = [("Y", 1), ("X", 0), ("Z", 2)]
    for col, (plane, axis) in enumerate(bar_planes):
        ax = axs[1, col]
        ax.bar([TASK_ABBREV[t] for t in TASKS], amplitudes[axis], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xlabel("Tasks", fontweight="bold", fontsize=11)
        if col == 0:
            ax.set_ylabel("L range (peak-to-peak)", fontweight="bold", fontsize=12)
        ax.annotate(r"$\times 10^{-2}$", xy=(-0.15, 1.02), xycoords="axes fraction", ha="left", va="bottom", fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(TASKS))]
    fig.legend(
        handles,
        [TASK_FULL_NAMES[task] for task in TASKS],
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    figure_dir = output_dir / "figures"
    fig.savefig(figure_dir / "figure2_wbam_median_normalized_and_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "figure2_wbam_median_normalized_and_bars.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def save_all_tasks_grid(task_medians_df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a tasks x planes grid of median WBAM curves.

    One subplot per task (rows) and plane (columns), using equal
    Y-axis scales for all tasks within the same plane - makes it
    easier to visually compare magnitude across tasks. The limit of
    each Y axis is computed from the largest absolute value observed
    across ALL tasks in that plane (with a 20% margin), rather than a
    fixed value, so tasks with a larger WBAM amplitude (e.g., Beam
    Walking, a more challenging balance task) don't get their curve
    clipped by a limit set from other tasks.

    Args:
        task_medians_df: Table produced by save_csv_outputs.
        output_dir: Results folder to save the figure (PNG/SVG, in
            output_dir/figures/) into.
    """
    colors = {"line": "blue"}
    planes = [("Frontal Plane", "Y", 1), ("Sagittal Plane", "X", 0), ("Transverse Plane", "Z", 2)]
    task_arrays = {task: task_median_array(task_medians_df, task) for task in TASKS}

    max_ys = {}
    for _title, plane, axis in planes:
        max_abs = max(float(np.nanmax(np.abs(task_arrays[task][:, axis]))) for task in TASKS)
        max_ys[plane] = max(max_abs * 1.2, 0.0001)

    fig, axs = plt.subplots(len(TASKS), 3, figsize=(10, 16), sharex=True)
    for row, task in enumerate(TASKS):
        arr = task_arrays[task]
        for col, (title, plane, axis) in enumerate(planes):
            ax = axs[row, col]
            ax.plot(PERCENT_CYCLE, arr[:, axis], color=colors["line"], linewidth=1.6)
            style_axis(ax, plane, max_ys[plane], title if row == 0 else None)
            if col == 0:
                ax.set_ylabel(TASK_FULL_NAMES[task], fontweight="bold", fontsize=9)
            if row == len(TASKS) - 1:
                ax.set_xlabel("Gait Cycle (%)", fontsize=9)
    fig.tight_layout()
    figure_dir = output_dir / "figures"
    fig.savefig(figure_dir / "figure2_all_tasks_normalized_medians.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "figure2_all_tasks_normalized_medians.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def save_individual_trial_figures(trials_df: pd.DataFrame, output_dir: Path) -> None:
    """Generates, per task, a figure with every individual trial's WBAM curve.

    One figure per task, with 3 subplots (one per plane), showing the
    representative series of EVERY individual trial (not the median) -
    useful for inspecting variability across trials and identifying
    trials with atypical behavior. Since each task produces its own
    figure (no need to share scale with other tasks), the Y-axis limit
    is computed dynamically from the largest absolute value among that
    task's own trials (with a 20% margin), instead of a fixed value -
    this prevents clipping tasks with a larger WBAM amplitude, such as
    Beam Walking.

    Args:
        trials_df: Table produced by save_csv_outputs.
        output_dir: Results folder to save the figures (PNG/SVG, in
            output_dir/figures/individual_trials_by_task/) into.
    """
    planes = [("Frontal Plane", "Y", 1), ("Sagittal Plane", "X", 0), ("Transverse Plane", "Z", 2)]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(TRIALS)))
    figure_dir = output_dir / "figures" / "individual_trials_by_task"

    for task in TASKS:
        task_data = trials_df[trials_df["task"] == task]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        for col, (title, plane, axis) in enumerate(planes):
            ax = axs[col]
            col_name = {"Y": "L_Y_frontal", "X": "L_X_sagittal", "Z": "L_Z_transverse"}[plane]
            if task_data.empty:
                max_y = 0.0001
            else:
                max_y = max(float(np.nanmax(np.abs(task_data[col_name].to_numpy()))) * 1.2, 0.0001)
            for trial_idx, trial in enumerate(TRIALS):
                data = trials_df[(trials_df["task"] == task) & (trials_df["trial"] == trial)].sort_values("gait_cycle_percent")
                if data.empty:
                    continue
                values = data[["L_X_sagittal", "L_Y_frontal", "L_Z_transverse"]].to_numpy()
                ax.plot(PERCENT_CYCLE, values[:, axis], color=colors[trial_idx], linewidth=1.5, label=f"{task}{trial}")
            style_axis(ax, plane, max_y, title)
            ax.set_xlabel("Gait Cycle (%)", fontsize=9)
            if col == 0:
                ax.set_ylabel("Angular momentum (L)", fontweight="bold", fontsize=10)
        axs[-1].legend(loc="upper right", fontsize=8, frameon=True)
        fig.suptitle(f"{TASK_FULL_NAMES[task]} - individual trials", fontweight="bold", fontsize=12)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_individual_trials_normalized.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{task}_individual_trials_normalized.svg", format="svg", bbox_inches="tight")
        plt.close(fig)


def select_participant_folder() -> Path:
    """Prompts the user to select the participant's folder.

    Shows a message explaining what to select, then opens the
    system's native folder-selection dialog (uses tkinter, which
    already ships with standard Python on Windows/Mac/Linux - nothing
    else to install). Starts browsing from the user's home folder
    (Path.home()).

    Only called when the script runs directly (see the
    `if __name__ == "__main__"` block at the end of the file) - never
    when this file is imported as a module, so it doesn't pop up
    unexpected windows in other scripts that reuse these functions.

    Returns:
        Selected folder path.

    Raises:
        RuntimeError: If the person cancels without choosing a folder.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # brings the window/message to the front

    message = "Select the folder with the participant's files"
    messagebox.showinfo("WBAM - Folder selection", message)

    chosen = filedialog.askdirectory(title=message, initialdir=str(Path.home()))
    root.destroy()

    if not chosen:
        raise RuntimeError("No folder was selected. Run the script again and choose the participant's folder.")
    return Path(chosen)


def select_subject_file(initial_dir: Path) -> Path:
    """Ask the operator to select the participant's subject-parameter CSV.

    The file is not inferred from a hard-coded participant folder. The dialog
    accepts either ``anatomica01_subject.csv`` or any equivalent CSV containing
    the required ``Parameter`` and ``Value`` columns.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo(
        "WBAM - Subject parameters",
        "Select the subject-parameter CSV for this participant "
        "(for example, anatomica01_subject.csv).",
        parent=root,
    )
    chosen = filedialog.askopenfilename(
        title="Select the participant subject-parameter CSV",
        initialdir=str(initial_dir),
        filetypes=[("Subject parameter CSV", "*_subject.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    if not chosen:
        raise RuntimeError("No subject-parameter CSV was selected.")
    selected = Path(chosen)
    # Validate immediately so an accidentally selected markers/events file is
    # reported before the time-consuming trial processing starts.
    load_subject_parameters(selected)
    return selected


def ask_subject_mass() -> float:
    """Prompts the user for the participant's mass.

    Opens a window asking for the participant's mass (kg), with
    validation (must be a number between 1 and 400). Re-asks if the
    entered value is invalid.

    Only called when the script runs directly (see the
    `if __name__ == "__main__"` block at the end of the file) - never
    when this file is imported as a module.

    Returns:
        Participant mass, in kg.

    Raises:
        RuntimeError: If the person cancels the window without
            entering a value.
    """
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    mass = simpledialog.askfloat(
        "WBAM - Participant mass",
        "Enter the participant's mass in kg.\n"
        "Use a period (.) as the decimal separator - example: 65.5",
        minvalue=1.0,
        maxvalue=400.0,
        parent=root,
    )
    root.destroy()

    if mass is None:
        raise RuntimeError("No mass was entered. Run the script again and enter the participant's mass.")
    return float(mass)


# =============================================================================
# PART 5 - ORCHESTRATION
# Batch loop over tasks/trials, selection audit, entry point.
# =============================================================================


def collect_results(
    source_dir: Path,
    output_dir: Path,
    fallback_selector: FallbackSelector | None = None,
) -> tuple[list[SelectedStrideResult], float]:
    """Process the single selected stride from every available trial.

    Reads the participant's mean static CoM height, then loops over
    TASKS x TRIALS. Body mass is taken from the module-level
    SUBJECT_MASS_KG, which must be set before calling this.

    Args:
        source_dir: Participant folder.
        output_dir: Results folder; its csv/ and figures/ subfolders are
            created here.
        fallback_selector: Optional replacement for the Tk popup used when a
            prescribed stride fails the CoM quality gate. Supply one to run
            headlessly; returning None skips the trial.

    Returns:
        Tuple (results, mean_com_height).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "csv").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "figures" / "individual_trials_by_task").mkdir(exist_ok=True)

    mean_com_height = load_com_height_static(source_dir)
    print(f"Mean COM height: {mean_com_height:.6f}")

    results: list[SelectedStrideResult] = []
    for task in TASKS:
        for trial in TRIALS:
            print(f"Processing {task}{trial}")
            result = process_trial(
                source_dir,
                task,
                trial,
                mean_com_height,
                fallback_selector,
            )
            if result is not None:
                results.append(result)
    return results, mean_com_height


def save_selected_stride_audit(
    results: list[SelectedStrideResult], output_dir: Path
) -> pd.DataFrame:
    """Save one auditable row describing the selected stride per trial."""
    rows = [
        {
            "inertia_model": INERTIA_MODEL_NAME,
            "task": result.task,
            "trial": result.trial,
            "selection_rule": result.selection_rule,
            "prescribed_side": result.prescribed_side,
            "prescribed_stride_number": result.prescribed_stride_number,
            "selected_side": result.selected_side,
            "selected_stride_number": result.selected_stride_number,
            "selected_start_frame": result.selected_start_frame,
            "selected_end_frame": result.selected_end_frame,
            "fallback_used": result.fallback_used,
            "obstacle_centroid_y_mm": result.obstacle_centroid_y_mm,
            "selected_minimum_com_speed_m_s": result.selected_minimum_com_speed_m_s,
        }
        for result in results
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "csv" / "selected_stride_audit.csv", index=False)
    return audit


def output_dir_for_participant(source_dir: Path) -> Path:
    """Keep selected-stride results separate from both earlier pipelines."""
    return source_dir / f"{source_dir.name}_results"


def main() -> None:
    """Run selected-stride WBAM, exports, figures and selection audit."""
    if SOURCE_DIR is None or OUTPUT_DIR is None:
        raise RuntimeError("SOURCE_DIR and OUTPUT_DIR must be selected first.")
    if SUBJECT_MASS_KG is None:
        raise RuntimeError("SUBJECT_MASS_KG must be set before running main().")
    results, _mean_com_height = collect_results(SOURCE_DIR, OUTPUT_DIR)
    if not results:
        raise RuntimeError("No trials were processed.")
    save_selected_stride_audit(results, OUTPUT_DIR)
    _cycles_df, trials_df, task_medians_df = save_csv_outputs(results, OUTPUT_DIR)
    save_figure2(task_medians_df, OUTPUT_DIR)
    save_all_tasks_grid(task_medians_df, OUTPUT_DIR)
    save_individual_trial_figures(trials_df, OUTPUT_DIR)
    print(f"Done. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    SOURCE_DIR = select_participant_folder()
    SUBJECT_FILE = select_subject_file(SOURCE_DIR)
    OUTPUT_DIR = output_dir_for_participant(SOURCE_DIR)

    mass_from_data = subject_mass_from_data(SUBJECT_FILE)
    SUBJECT_MASS_KG = (
        mass_from_data if mass_from_data is not None else ask_subject_mass()
    )

    print(f"Participant folder: {SOURCE_DIR}")
    print(f"Subject CSV:        {SUBJECT_FILE}")
    print(f"Results folder:     {OUTPUT_DIR}")
    print(f"Participant mass:   {SUBJECT_MASS_KG} kg")
    print(f"Inertia model:      {INERTIA_MODEL_NAME}")
    main()