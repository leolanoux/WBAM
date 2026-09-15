# WBAM — Protocol-Defined Stride Selection

`WBAM_tutorial.py`

Whole-Body Angular Momentum (WBAM) analysis of Vicon Plug-in Gait data in which
**exactly one stride per dynamic trial** is analysed, chosen by an explicit,
reproducible, task-specific rule rather than by averaging every available gait
cycle.

> **Single self-contained file.** `WBAM_tutorial.py` needs no companion module —
> only the packages listed in [§3](#3-requirements).

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Architecture](#2-architecture)
3. [Requirements](#3-requirements)
4. [Input data](#4-input-data)
5. [Running the analysis](#5-running-the-analysis)
6. [The stride-selection protocol](#6-the-stride-selection-protocol)
7. [CoM quality gate and operator fallback](#7-com-quality-gate-and-operator-fallback)
8. [Outputs](#8-outputs)
9. [Conventions and model definitions](#9-conventions-and-model-definitions)
10. [API reference](#10-api-reference)
11. [Adapting the module to another laboratory or protocol](#11-adapting-the-module-to-another-laboratory-or-protocol)
12. [Design notes and known limitations](#12-design-notes-and-known-limitations)
13. [Worked example](#13-worked-example)
14. [References](#14-references)

---

## 1. Motivation

In whole-body angular momentum studies of obstacle crossing, obstacle
circumvention, beam walking and reduced-base-of-support walking, **not every
gait cycle in a trial answers the research question.** A trial typically
contains approach strides, one or two strides in which the participant actually
interacts with the obstacle or the constrained surface, and recovery strides.
Pooling all of them dilutes the effect of interest and mixes mechanically
distinct behaviours into a single median.

Averaging every cycle also creates a numerical hazard. Whole-body centre-of-mass
(CoM) trajectories exported from Vicon Nexus can contain gaps at the beginning
or end of a trial. When those gaps are boundary-filled, the CoM becomes
artificially stationary, the instantaneous CoM speed collapses towards zero, and
because WBAM is normalized frame by frame by

```
m · |v_CoM(t)| · h̄_CoM
```

the normalized values explode. In this project's validation dataset one beam-walking
trial produced a minimum CoM speed of ≈ 4.8 × 10⁻⁸ m/s through exactly this
mechanism.

This module addresses both problems at once:

* a **deterministic, documented rule** selects the single stride that the
  experimental protocol is actually about;
* a **quality gate** verifies that the selected stride can be normalized safely
  before any angular momentum is extracted, and records the decision in an audit
  table.

Every selection decision is exported, so the analysis is reproducible and
reviewable without re-running the code.

---

## 2. Architecture

Everything lives in one file, organised in five parts:

| Part | Contents |
| --- | --- |
| 1 — Configuration | Paths, sampling rate, unit conversion, task list, Winter/PiG inertia tables |
| 2 — WBAM mechanics | Trial loading, filtering, segment rotation reconstruction, inertia tensors, angular momentum, normalization, cycle resampling |
| 3 — Stride selection | Obstacle centroid, the task-specific rules, the CoM quality gate, the operator fallback |
| 4 — Exports and figures | Summary CSV tables, PNG/SVG figures, folder/file dialogs |
| 5 — Orchestration | Batch loop over tasks × trials, selection audit, entry point |

**The mechanics and the selection rules remain cleanly separable even though
they share a file.** The seam is one argument:

```python
compute_trial_wbam(source_dir, task, trial, mean_com_height,
                   cycles_override)                # all the physics
process_trial(source_dir, task, trial, mean_com_height,
              fallback_selector)                   # decides which stride
```

`compute_trial_wbam` computes WBAM for whatever cycles it is given. Passing
`cycles_override=None` makes it read the events file and keep every complete
cycle — the classic all-cycles analysis. `process_trial` applies the selection
rule, validates the result, and passes exactly one stride. **To implement a
different stride-selection protocol, edit Part 3 only; the mechanics need no
changes.**

The underlying angular momentum model is

```
L(t) = Σᵢ [ Rᵢ(t) · Iᵢ · ωᵢ(t)  +  mᵢ · (rᵢ(t) − r_CoM(t)) × (vᵢ(t) − v_CoM(t)) ]
```

summed over 15 segments (head, thorax including abdomen, pelvis, and bilateral
upper arm, forearm, hand, thigh, shank, foot), using Winter/Dempster mass
fractions as implemented by the Vicon Plug-in Gait kinetic model. Segment
angular velocity ωᵢ is derived from each segment's **absolute** rotation in the
laboratory frame, reconstructed from the Origin/Anterior/Lateral/Proximal
technical points Nexus exports per segment — not from clinical joint angles,
which describe relative rotation between adjacent segments and are a different
physical quantity.

---

## 3. Requirements

* Python ≥ 3.10 (the module uses PEP 604 `X | None` annotations and
  `from __future__ import annotations`)
* `numpy`
* `pandas`
* `scipy` (Butterworth filtering, cycle interpolation)
* `matplotlib` (figures)
* `ezc3d` (only when C3D files must be auto-exported; see §4.3)
* `tkinter` (folder/file dialogs and the fallback popup; ships with most CPython
  builds, but on some Linux distributions it must be installed separately, e.g.
  `sudo apt install python3-tk`)

```bash
pip install numpy pandas scipy matplotlib ezc3d
```

`tkinter` is imported lazily inside `choose_alternative_stride_popup`, so the
module can be imported and used headlessly as long as you supply your own
`fallback_selector` (see §10.4).

---

## 4. Input data

### 4.1 Directory layout

One participant folder contains one static trial and the dynamic trials. The
default layout expected by `load_trial` is:

```
<participant_folder>/
├── PO1_N_AV1_anatomica/
│   └── anatomica01_markers.csv          # static trial (mean CoM height)
├── PO1_N_AV1_UW01/
│   ├── PO1_N_AV1_UW01_markers.csv       # per-frame marker & model outputs
│   ├── PO1_N_AV1_UW01_frames.csv        # first_frame, last_frame
│   └── PO1_N_AV1_UW01_events.csv        # gait event times, in seconds
├── PO1_N_AV1_TOH01/
│   └── ...
└── anatomica01_subject.csv              # Parameter/Value anthropometrics
```

The filename stem pattern is `PO1_N_AV1_<task><trial>`. If those three CSVs are
absent, the pipeline falls back to recursive C3D discovery (§4.3).

### 4.2 Required columns

**`*_markers.csv`** — one row per frame, columns named `<point>_<axis>` with
axis ∈ {X, Y, Z}, in **millimetres**:

| Group | Columns | Used for |
| --- | --- | --- |
| Whole-body CoM | `CentreOfMass_X/Y/Z` | normalization denominator, quality gate |
| Segment CoM | `HeadCOM_*`, `ThoraxCOM_*`, `PelvisCOM_*`, `Left/RightHumerusCOM_*`, `Left/RightRadiusCOM_*`, `Left/RightHandCOM_*`, `Left/RightFemurCOM_*`, `Left/RightTibiaCOM_*`, `Left/RightFootCOM_*` | orbital term; DOV/TOH foot-crossing test |
| Segment technical points | `HED*`, `TRX*`, `PEL*`, `LHU*`/`RHU*`, `LRA*`/`RRA*`, `LHN*`/`RHN*`, `LFE*`/`RFE*`, `LTI*`/`RTI*`, `LFO*`/`RFO*` with O/A/L/P suffixes | absolute segment rotation → ωᵢ |
| Obstacle markers | `obstaculo1_*` … `obstaculo4_*` | **TOH and DOV only** — obstacle centroid |

Obstacle-marker lookup is **case-insensitive**. Segment column names follow the
Nexus Plug-in Gait export convention.

**`*_frames.csv`** — a single row with `first_frame` and `last_frame`. If the
marker table length disagrees with `last_frame`, the marker table wins and
`end_frame` is corrected silently.

**`*_events.csv`** — two rows: event names and event times in seconds. Names
follow `Right_Foot Strike`, `Left_Foot Strike`, `Right_Foot Off`, …; repeated
names are disambiguated with `_2`, `_3`, … suffixes. Event times are converted
to frames with `round(t · FREQ_HZ)`, where `FREQ_HZ = 200.0`.

**`*_subject.csv`** — two columns, `Parameter` and `Value`. Only `Bodymass`
is used; it is read automatically into `SUBJECT_MASS_KG`. The other Plug-in
Gait parameters (`LKneeWidth`, `RAnkleWidth`, …) are ignored, because the
radius-of-gyration model needs no measured segment widths — see
[§9](#9-conventions-and-model-definitions).

### 4.3 Automatic C3D export

When the three per-trial CSVs are missing, `export_discovered_c3d_trial`
searches the participant folder recursively for a `.c3d` whose stem contains
`<task><trial>` (case-insensitive), converts it with `ezc3d`, and caches the
result under `<participant_folder>/_wbam_export/<task><trial>/`. Discovery is
participant-name independent. For trial `01` only, a task-only filename such as
`tentativaUW.c3d` is also accepted; static/anatomical C3Ds are excluded. If
zero or more than one candidate matches, discovery is abandoned and the trial is
skipped with a printed message.

---

## 5. Running the analysis

### 5.1 Interactive (default)

```bash
python WBAM_tutorial.py
```

Two dialogs open, in order:

1. **participant folder** — the directory containing the trial subfolders;
2. **subject CSV** — that participant's `*_subject.csv`.

If the CSV contains no valid `Bodymass`, a third dialog asks for body mass in kg.
The output folder is created automatically as
`<participant_folder>/<folder_name>_results`.

The console reports each trial as it is processed:

```
Processing TOH01
Selected TOH01 right stride 1: frames 1077-1346; minimum CoM speed 0.734 m/s using rule: right-foot stride whose consecutive Foot Strike positions bracket the Y plane through the four-marker obstacle centroid
```

### 5.2 Programmatic / batch

```python
import matplotlib
matplotlib.use("Agg")                      # no display needed

from pathlib import Path
import WBAM_tutorial as w

source = Path("data/P01")
subject = source / "anatomica01_subject.csv"

# Participant state lives in module globals; set it before calling anything.
w.SOURCE_DIR = source
w.SUBJECT_FILE = subject
w.OUTPUT_DIR = w.output_dir_for_participant(source)
w.SUBJECT_MASS_KG = w.subject_mass_from_data(subject)

def skip_on_failure(task, trial, prescribed, alternatives):
    """Headless fallback: never substitute a stride, just skip the trial."""
    print(f"{task}{trial}: prescribed stride rejected ({'; '.join(prescribed.reasons)})")
    return None

results, mean_com_height = w.collect_results(
    source, w.OUTPUT_DIR, fallback_selector=skip_on_failure
)
audit = w.save_selected_stride_audit(results, w.OUTPUT_DIR)
cycles_df, trials_df, task_medians_df = w.save_csv_outputs(results, w.OUTPUT_DIR)
```

Supplying a `fallback_selector` is what makes the pipeline non-interactive: the
Tk popup is only reached when no selector is given.

> **Note on module-level state.** `SUBJECT_MASS_KG`, `SOURCE_DIR`,
> `SUBJECT_FILE` and `OUTPUT_DIR` are module globals. They must be set before
> `collect_results` or `main` is called, and a single Python process can
> therefore process only one participant at a time.

### 5.3 Analysing every cycle instead of one stride

The mechanics are not tied to the selection rules. Call `compute_trial_wbam`
directly, leaving `cycles_override` at its default, to recover the classic
all-cycles analysis for one trial:

```python
result = w.compute_trial_wbam(source, "UW", "01", mean_com_height)
# result.right_cycles / result.left_cycles_flipped hold every valid cycle,
# each resampled to 100 points, after the same CoM validity screening.
```

---

## 6. The stride-selection protocol

Every candidate stride is bounded by **two consecutive same-side `Foot Strike`
events** taken from the trial's `_events.csv`. Strides that fall outside
`[first_frame, last_frame]` are discarded upstream by
`gait_cycles_from_events`. Stride numbers are **1-based, in temporal
order, within the side**.

### 6.1 Rule summary

| Task | Code | Side | Rule |
| --- | --- | --- | --- |
| Habitual walking | `UW` | right (fixed) | stride whose temporal midpoint is closest to the trial midpoint |
| Reduced base of support | `BSR` | right (fixed) | same |
| Beam walking | `CT` | right (fixed) | same |
| Horizontal obstacle crossing | `TOH` | right (fixed) | stride whose two Foot Strikes **bracket** the obstacle plane |
| Right vertical obstacle circumvention | `DOVdir` | **determined from data** | stride whose own foot CoM **crosses** the obstacle plane nearest its own midpoint |
| Left vertical obstacle circumvention | `DOVesq` | **determined from data** | same |

An unrecognised task raises `ValueError`.

### 6.2 UW, BSR, CT — temporally central stride

`central_stride_number` minimises the distance between the stride midpoint and
the trial midpoint:

```
argminₖ | (startₖ + endₖ)/2 − (first_frame + last_frame)/2 |
```

Ties break towards the earlier stride (stable ordering by index). This selects
the stride recorded during steady-state locomotion, away from gait initiation
and termination. Only right strides are considered.

### 6.3 Obstacle centroid (TOH and DOV)

Both obstacle rules are defined relative to a plane perpendicular to the
progression axis, passing through the obstacle centroid.

`obstacle_centroid_mm` computes it in two steps:

1. **Spatial mean per frame** — the arithmetic mean of the available
   `obstaculo1`–`obstaculo4` positions at each frame (`np.nanmean`, so markers
   momentarily occluded are ignored for that frame):

   ```
   c(t) = (1/N) Σᵢ rᵢ(t)
   ```

2. **Temporal median** — across all frames of the trial, giving one stable
   centroid per trial and rejecting transient marker noise:

   ```
   C = medianₜ [ c(t) ]
   ```

The result is returned as a 3-vector in millimetres. A non-finite centroid
raises `ValueError`. **Only the progression component `C[1]` (lab Y) is used**;
the X and Z components are computed but discarded.

> **Interpretation.** This is the centroid of the *marker set*, not a measured
> centre of mass or a volumetric centroid of the physical object. It coincides
> with the object's horizontal centre only when the four markers were placed
> symmetrically around it. Verify this for your own setup. Note in particular
> that markers mounted on top of an obstacle make `C[2]` the height of the top
> surface, not the object's vertical centre.

### 6.4 TOH — bracketing rule

`obstacle_crossing_stride_number` evaluates the **right-foot CoM progression
coordinate at the two Foot Strike frames** that bound each stride:

```
Δ₁ = y_foot(start) − C_y
Δ₂ = y_foot(end)   − C_y
```

A stride qualifies when the two strikes lie on **opposite sides** of the plane
(`signbit(Δ₁) ≠ signbit(Δ₂)`), or when either strike falls exactly on it.
Strides entirely before or entirely after the obstacle are rejected. Among the
qualifying strides, the one whose two strikes are most symmetric about the
obstacle wins:

```
argmin | (y_foot(start) + y_foot(end))/2 − C_y |
```

If no stride brackets the plane, `ValueError` is raised and the trial is skipped
with a printed message.

### 6.5 DOV — side-neutral crossing rule

`side_neutral_obstacle_crossing_stride` evaluates **both right and left strides
as equal candidates**. The side is a *result* of the geometry, never inferred
from the `DOVdir` / `DOVesq` task label — the task name describes the side the
obstacle was placed on in the protocol, not necessarily the limb whose stride
best captures the circumvention.

For each candidate stride, the foot CoM progression series is taken over the
**interior** of the stride, `frames (start, end)` exclusive of both boundaries,
and:

* `crosses` — true when `min(Δ) ≤ 0 ≤ max(Δ)`, i.e. the foot CoM passes through
  the obstacle plane at some point inside the stride;
* `crossing_frame` — the interior frame where `|Δ|` is smallest;
* `midpoint_distance` — `|crossing_frame − (start + end)/2|`, in frames;
* `distance` — `min |Δ|`, in millimetres.

Candidates are ranked lexicographically by

```
(0 if crosses else 1,  midpoint_distance,  distance,  side_order)
```

so: crossing strides always beat non-crossing ones; among those, the stride
whose crossing sits closest to its own temporal midpoint wins; the smallest
foot-to-plane distance is the next tie-break; and `side_order` (right before
left) makes the outcome fully deterministic. Strides with no finite foot data
receive `crosses = False` and infinite distances, so they rank last but are never
silently dropped.

> **Asymmetry worth knowing.** TOH tests the two *boundary* frames; DOV tests the
> stride *interior*. This is intentional — TOH asks "did the participant step
> over the obstacle during this stride?", which is a property of where the feet
> landed, while DOV asks "did the foot pass the obstacle plane during this
> stride?", which is a property of the swing trajectory. It does mean the two
> rules are not interchangeable.

---

## 7. CoM quality gate and operator fallback

Before any angular momentum is extracted, `assess_stride_com_quality` checks the
prescribed stride — **and every other stride on the same side**, so that valid
alternatives are known in advance.

A stride is **valid** only when all of the following hold:

| Check | Threshold |
| --- | --- |
| Stride length | ≥ 3 frames |
| Stride lies within the trial frame index | required |
| Original, **pre-interpolation** `CentreOfMass_X/Y/Z` samples | zero missing frames |
| Filtered CoM speed | all frames finite |
| Minimum filtered CoM speed | ≥ `MIN_COM_SPEED_M_S` = 0.001 m/s |

The two decisive points are that the gate reads the **original CoM validity
mask, before any gap filling**, and that it tests **the very same filtered CoM
speed that forms the normalization denominator**. A stride that passes cannot
produce a near-zero denominator.

If the prescribed stride passes, it is used and `fallback_used = False`.

If it fails, `resolve_stride_with_quality_check` collects the valid
alternatives on the same side and calls the `fallback_selector`. The default
selector, `choose_alternative_stride_popup`, opens a Tk dialog listing each
alternative with its frame range and minimum CoM speed:

* the operator picks one → that stride is used, `fallback_used = True`;
* the operator cancels → the trial is **skipped**;
* no valid alternative exists → an error dialog is shown and the trial is
  **skipped**.

The fallback never crosses to the other side: substitutions stay on the
prescribed side so that the reported side remains meaningful.

Any custom selector must match:

```python
Callable[[str, str, StrideQuality, list[StrideQuality]], StrideQuality | None]
#         task  trial  prescribed   alternatives         chosen or None
```

Returning `None` skips the trial. **Every fallback is recorded in the audit
table**, so operator intervention is always visible in the published results.

---

## 8. Outputs

All outputs are written under
`<participant_folder>/<folder_name>_results/`:

```
csv/
├── selected_stride_audit.csv               # one row per trial: which stride, and why
├── individual_cycles_normalized.csv        # the selected stride, 0–100%
├── individual_trial_series_normalized.csv  # representative series per trial
├── task_medians_normalized.csv             # median across trials, per task
├── figure2_amplitudes.csv                  # peak-to-peak per task and plane
└── angular_velocity_audit.csv              # peak |ω| per segment/trial
figures/
├── figure2_wbam_median_normalized_and_bars.png / .svg   # main multi-task figure
├── figure2_all_tasks_normalized_medians.png / .svg
└── individual_trials_by_task/*.png / .svg
```

### 8.1 `selected_stride_audit.csv` — the key provenance table

One row per successfully processed trial.

| Column | Type | Meaning |
| --- | --- | --- |
| `inertia_model` | str | Always `Vicon Plug-in Gait / Winter-Dempster` |
| `task` | str | `UW`, `TOH`, `DOVdir`, `DOVesq`, `BSR`, `CT` |
| `trial` | str | `01`–`05` |
| `selection_rule` | str | Human-readable statement of the rule applied |
| `prescribed_side` | str | `right` / `left`, from the protocol rule |
| `prescribed_stride_number` | int | 1-based stride the rule originally chose |
| `selected_side` | str | Side actually analysed |
| `selected_stride_number` | int | 1-based stride actually analysed |
| `selected_start_frame` | int | First frame (a Foot Strike) |
| `selected_end_frame` | int | Last frame (the next same-side Foot Strike) |
| `fallback_used` | bool | `True` when an operator substituted the stride |
| `obstacle_centroid_y_mm` | float | Progression coordinate of the obstacle plane; **empty for UW, BSR, CT** — those rules do not use an obstacle |
| `selected_minimum_com_speed_m_s` | float | Smallest filtered CoM speed inside the selected stride |

`prescribed_*` differing from `selected_*` is the signature of an operator
fallback. Empty `obstacle_centroid_y_mm` cells for UW/BSR/CT are intentional,
not missing data.

### 8.2 WBAM tables

`individual_cycles_normalized.csv` contains, per row, one point of one cycle:

| Column | Meaning |
| --- | --- |
| `inertia_model`, `task`, `task_label`, `trial` | identifiers |
| `side` | `right` or `left_flipped` |
| `cycle` | always `1` in this pipeline — one stride per trial |
| `gait_cycle_percent` | 0–100, 100 points |
| `L_X_sagittal`, `L_Y_frontal`, `L_Z_transverse` | normalized WBAM (dimensionless) |

`individual_trial_series_normalized.csv` adds `task_name` and `series_source`
(which cycles the representative series came from).
`task_medians_normalized.csv` holds `median_L_X_sagittal`,
`median_L_Y_frontal`, `median_L_Z_transverse` — the median across trials of a
task, and the series plotted in the figures.

`figure2_amplitudes.csv` is written by `save_figure2` alongside the main
figure, one row per task:

| Column | Meaning |
| --- | --- |
| `inertia_model`, `task`, `task_label` | identifiers |
| `frontal_Y_peak_to_peak` | range of the task-median `L_Y` over the cycle |
| `sagittal_X_peak_to_peak` | range of the task-median `L_X` |
| `transverse_Z_peak_to_peak` | range of the task-median `L_Z` |

These are the values plotted in the bar panels of the main figure: the
peak-to-peak excursion of each task's median normalized WBAM, per plane.

`angular_velocity_audit.csv` reports `max_abs_angular_velocity_rad_s` per
segment and trial. Use it as a plausibility check: implausibly large peaks
usually indicate a mislabelled or gap-filled technical marker rather than a real
movement.

---

## 9. Conventions and model definitions

**Laboratory axes** (as calibrated for the datasets this module was developed
on — verify before reusing):

| Axis | Direction |
| --- | --- |
| X | mediolateral |
| Y | progression / anteroposterior |
| Z | vertical |

`PROGRESSION_AXIS = "Y"` encodes this and is used by every obstacle rule.

**Anatomical plane labels.** A plane is spanned by two axes and named by the
axis normal to it:

| Plane | Spanned by | Normal axis | WBAM component |
| --- | --- | --- | --- |
| Sagittal | Y–Z | X | `L_X_sagittal` |
| Frontal | X–Z | Y | `L_Y_frontal` |
| Transverse | X–Y | Z | `L_Z_transverse` |

These labels are appropriate for angular momentum, which is a rotation about the
normal axis. They are **not** appropriate for position or linear-velocity
components: a CoM position `[X, Y, Z]` holds mediolateral, anteroposterior and
vertical coordinates, and should never be described as sagittal, frontal and
transverse coordinates.

**Units.** Marker CSVs are in millimetres; `MM_TO_M = 0.001` converts to SI
before any mechanics. The obstacle centroid and foot progression series are the
exception — they stay in **millimetres**, because they are only ever compared
with each other. Angular momentum output is dimensionless after normalization.

**Sign convention for the left side.** Left strides are stored as
`left_cycles_flipped`, multiplied by −1, so that left and right strides can be
pooled and compared directly. This matters for DOV, where the selected side is
data-determined: a DOV trial analysed on the left contributes a sign-flipped
series.

**Inertia model.** Each segment's inertia tensor comes from the
radius-of-gyration model. Segment shape is never integrated:

```
I = m · (k · L)²
```

| Symbol | Source |
| --- | --- |
| `m` | `SEGMENT_MASS_RATIO[segment] × SUBJECT_MASS_KG` — Winter/Dempster body-mass fractions as implemented by the Plug-in Gait kinetic model, closing exactly to one body mass over 15 segments |
| `k` | `VICON_GYRATION_RATIO[segment]` — the dimensionless PiG radius-of-gyration ratio |
| `L` | `segment_anatomical_length_series()` — one trial median per segment, in metres |

Radius-of-gyration ratios:

| Segment | k | Segment | k |
| --- | ---: | --- | ---: |
| Head | 0.495 | Thigh | 0.323 |
| Thorax | 0.310 | Shank | 0.302 |
| Pelvis | 0.310 | Foot | 0.475 |
| UpperArm | 0.322 | Forearm | 0.303 |
| Hand | 0.223 | | |

`L` is the **model-matched** anatomical length, not the visible external extent,
because each `k` is normalized by a specific landmark pair: joint-centre
distances for the limbs (SJC→EJC, EJC→WJC, HJC→KJC, KJC→AJC), AJC→TOE for the
foot, WJC→FIN ÷ 0.75 for the hand, 0.925 × inter-HJC for the pelvis, and
locally-projected C7 distances for thorax and head. Using heel-to-toe for the
foot with the same `k = 0.475`, for instance, inflates foot inertia.

The tensor is diagonal in the segment's **local** basis, whose X axis is the
longitudinal (proximal-distal) direction built by
`rotation_matrices_from_oalp()`:

```python
np.diag([longitudinal, transverse, transverse])
```

Ordinary segments receive **zero longitudinal inertia**, so rotation about the
long axis contributes nothing to the spin term. Head and Pelvis receive the same
value on all three axes, following the PiG convention adopted here.

Because `m`, `k` and `L` are all constant within a trial, so is the tensor — the
segment is rigid, and only its orientation changes frame to frame. The spin term
is rotated into the laboratory frame as `R · (I · ω)` before being summed with
the orbital term.

Only this model is implemented. `segment_inertia_tensor()` raises
`NotImplementedError` for any other `INERTIA_MODEL_NAME`, because segment lengths
and radii of gyration must come from the same source — silently pairing one
model's lengths with another's ratios would produce plausible-looking but wrong
inertia — so `VICON_GYRATION_RATIO` and the landmark definitions in
`segment_anatomical_length_series()` must be replaced together. Consequently
the subject CSV contributes only `Bodymass`.

**Normalization.**

```
L̂(t) = L(t) / ( m · |v_CoM(t)| · h̄_CoM )
```

with `m` the body mass in kg, `|v_CoM(t)|` the instantaneous filtered whole-body
CoM speed in m/s, and `h̄_CoM` the mean whole-body CoM height during the static
trial in m. Normalization is **frame by frame**, which is why the CoM-speed
floor in §7 is essential. Marker positions are low-pass filtered with a
zero-phase 4th-order Butterworth at 6 Hz before differentiation.

**Cycle resampling.** Each selected stride is interpolated to 100 points
spanning 0–100 % of the stride, so strides of different duration are directly
comparable.

---

## 10. API reference

### 10.1 Module constants

```python
PROGRESSION_AXIS = "Y"
OBSTACLE_MARKERS = ("obstaculo1", "obstaculo2", "obstaculo3", "obstaculo4")
SOURCE_DIR: Path | None      # set by the __main__ block
OUTPUT_DIR: Path | None
SUBJECT_FILE: Path | None
```

### 10.2 Data classes

**`StrideQuality`** (frozen) — the outcome of the CoM check for one candidate
stride.

| Field | Meaning |
| --- | --- |
| `side`, `stride_number` | identity of the candidate |
| `start_frame`, `end_frame` | inclusive frame bounds |
| `missing_com_frames` | count of frames with missing original whole-body CoM |
| `nonfinite_speed_frames` | count of non-finite filtered CoM-speed frames |
| `minimum_com_speed_m_s` | smallest finite CoM speed, or `NaN` |
| `valid` | `True` only when `reasons` is empty |
| `reasons` | tuple of human-readable rejection reasons |

`.label()` returns a one-line description used in console output and dialogs.

**`SelectedStrideResult`** — a `TrialResult` (task, trial, `right_cycles`,
`left_cycles_flipped`, `angular_velocity_max_abs`) extended with the full
selection audit trail (`prescribed_side`, `selected_stride_number`,
`selection_rule`, `fallback_used`, `obstacle_centroid_y_mm`, …).

### 10.3 Selection functions

| Function | Returns |
| --- | --- |
| `obstacle_centroid_mm(markers, frame_index)` | `np.ndarray` (3,), mm — time-median four-marker centroid. Raises if no marker found or result non-finite. |
| `foot_progression_series_mm(markers, frame_index, side)` | `np.ndarray`, mm — `Right/LeftFootCOM_Y` over the trial. Raises if the column is absent. |
| `central_stride_number(cycles, start_frame, end_frame)` | `int`, 1-based — temporally central stride. |
| `obstacle_crossing_stride_number(cycles, foot_y, obstacle_y, frame_index)` | `int`, 1-based — TOH bracketing rule. Raises if none qualifies. |
| `side_neutral_obstacle_crossing_stride(right_cycles, left_cycles, markers, frame_index, obstacle_y)` | `(side, stride_number)` — DOV rule. |
| `prescribed_stride(task, right_cycles, left_cycles, markers, frame_index, start_frame, end_frame)` | `(side, stride_number, rule_text, obstacle_y_or_None)` — dispatches on task. |

### 10.4 Validation and orchestration

| Function | Purpose |
| --- | --- |
| `assess_stride_com_quality(cycle, side, stride_number, body_com_source_valid, body_speed, frame_index)` | Build a `StrideQuality` for one candidate. |
| `choose_alternative_stride_popup(task, trial, prescribed, alternatives)` | Default Tk fallback selector. |
| `resolve_stride_with_quality_check(...)` | Returns `(StrideQuality | None, fallback_used)`. Accepts a custom `fallback_selector`. |
| `compute_trial_wbam(source_dir, task, trial, mean_com_height, cycles_override=None)` | **All the mechanics.** Returns a `TrialResult` or `None`. With `cycles_override=None` it reads the events file and keeps every valid cycle; pass `(right_cycles, left_cycles)` to restrict it to chosen strides. |
| `process_trial(source_dir, task, trial, mean_com_height, fallback_selector=None)` | Selection + quality gate + `compute_trial_wbam`; returns `SelectedStrideResult` or `None`. |
| `collect_results(source_dir, output_dir, fallback_selector=None)` | Loops over `TASKS` × `TRIALS`; returns `(results, mean_com_height)`. Body mass comes from the module-level `SUBJECT_MASS_KG`. |
| `save_selected_stride_audit(results, output_dir)` | Writes `selected_stride_audit.csv`; returns the DataFrame. |
| `output_dir_for_participant(source_dir)` | `<source_dir>/<name>_results`. |
| `main()` | Runs the whole pipeline: results, audit, CSVs, figures. |

`process_trial` and `collect_results` never raise on a bad trial. They print the
reason and return `None` / omit the trial, so a batch run always completes.

---

## 11. Adapting the module to another laboratory or protocol

This code was written for one specific acquisition protocol. The following are
hard-coded and **must be reviewed** before reuse.

| What | Where | Note |
| --- | --- | --- |
| Task codes `UW, TOH, DOVdir, DOVesq, BSR, CT` | `TASKS` (Part 1) | Portuguese-derived codes; `prescribed_stride` dispatches on them and raises on anything else |
| Trial numbers `01`–`05` | `TRIALS` (Part 1) | Extend for more repetitions |
| Filename stem `PO1_N_AV1_<task><trial>` | `load_trial` (Part 2) | Change here for another naming scheme |
| Static file `*anatomica*_markers.csv` | `load_com_height_static` (Part 2) | Falls back to a recursive search that requires exactly one match |
| Obstacle marker names | `OBSTACLE_MARKERS` (Part 1) | Portuguese (`obstaculo…`); rename for your marker set |
| Progression axis | `PROGRESSION_AXIS` (Part 1) | Change if your lab's forward direction is not +Y |
| Sampling rate 200 Hz | `FREQ_HZ` (Part 1) | Used to convert event times to frames — **must** match your acquisition |
| Filter cutoff 6 Hz, order 4 | `safe_filter` (Part 2) | Standard for gait; reconsider for faster tasks |
| CoM speed floor 0.001 m/s | `MIN_COM_SPEED_M_S` (Part 1) | Chosen well below the minimum speed of any valid stride in the reference dataset (0.043 m/s) |
| Winter/Dempster mass fractions and PiG gyration ratios | `SEGMENT_MASS_RATIO`, `VICON_GYRATION_RATIO` (Part 1) | Replace for another inertia model. Segment lengths must be replaced together with them — see the guard in `segment_inertia_tensor` |
| Segment column prefixes | `SEGMENT_COM_PREFIX`, `SEGMENT_OALP_PREFIX` (Part 1) | Nexus Plug-in Gait export names |

**Adding a new task rule.** Add a branch to `prescribed_stride` returning
`(side, stride_number, rule_text, obstacle_y_or_None)`. Write `rule_text` as a
complete sentence — it is exported verbatim to the audit CSV and is what a
reader of your results will see.

---

## 12. Design notes and known limitations

**How one stride reaches the mechanics.** `process_trial` passes the chosen
stride to `compute_trial_wbam` as an explicit `cycles_override` argument. Cycle
validation, interpolation, cycle storage and every downstream summary therefore
operate on exactly that stride, with no global state involved. The two layers
are independently testable, and the pipeline is safe to run concurrently as long
as each participant gets its own process — the participant globals
(`SUBJECT_MASS_KG`, `SOURCE_DIR`, …) remain the only shared state.

**Double CoM validation.** The stride passes the quality gate in `process_trial`,
and then `validated_cycles` re-checks it inside `compute_trial_wbam` against the
same criteria. This is redundant by design: the gate exists to *choose* a stride
and report why, the downstream check exists to *protect* the mechanics, including
when `compute_trial_wbam` is called directly with no selection layer at all. A
stride accepted by the gate always passes downstream.

**Obstacle marker dropout is silent.** `obstacle_centroid_mm` skips any of the
four markers whose columns are entirely absent and averages the rest, without
warning. `ValueError` is raised only when none of the four is found. Confirm
marker completeness in your own data before trusting the centroid; the temporal
median makes per-frame occlusions harmless, but a wholly missing marker biases
the centroid.

**Only the progression component of the centroid is used.** X and Z are
computed and discarded. If markers are mounted on top of an obstacle, `C[2]` is
the top surface height, not the object's vertical centre — do not reuse it as
obstacle height.

**One stride per trial is a deliberate statistical trade-off.** Within-trial
variability is not captured, and the analysis assumes the protocol-defined
stride is the mechanically relevant one. Report the audit table alongside the
results so readers can see exactly which strides entered the analysis.

**Cross-side comparability.** Because left strides are sign-flipped, a DOV task
in which the rule selects left strides for some trials and right strides for
others produces a task median that pools sign-corrected series from both limbs.
Check `selected_side` in the audit before interpreting DOV medians as
side-specific.

**Trial-level failures are silent in the outputs.** Skipped trials are reported
on stdout but simply do not appear in the CSVs. Compare the number of audit rows
against the number of trials you expect.

---

## 13. Worked example

A complete run on a 5-trial-per-task dataset (82 kg, 1.780 m, 200 Hz,
30 dynamic trials) produced 30 audit rows — exactly one stride per trial — with
no operator fallback (`fallback_used = False` throughout) and all WBAM outputs
finite.

| Task | Selected side | Selected stride | Comment |
| --- | --- | --- | --- |
| `UW` ×5 | right | 1 | temporally central stride |
| `BSR` ×5 | right | 1 | temporally central stride |
| `CT` ×5 | right | 1–3 | varies with trial length; one trial starts at frame 4094 |
| `TOH` ×5 | right | 1 | consecutive Right Foot Strikes on opposite sides of the obstacle plane in every trial |
| `DOVdir` ×5 | right | 2 | side chosen geometrically, not from the task name |
| `DOVesq` ×5 | right | 2 | idem — the label says the obstacle was on the left, the data selected the right limb |

The `DOVesq` result is the clearest illustration of why the DOV rule is
side-neutral: had the side been inferred from the task name, five trials would
have been analysed on the limb that did not produce the crossing nearest its own
stride midpoint.

This same dataset also motivated the quality gate. In an earlier all-cycle run,
one `CT` trial had missing whole-body CoM over frames 1111–1294; boundary
filling drove the minimum CoM speed to ≈ 4.83 × 10⁻⁸ m/s and produced extreme
normalized values. The selected-stride pipeline instead used the valid
temporally central right stride (frames 1355–1716), whose maximum absolute
normalized WBAM is ≈ 0.146.

---

## 14. References

* Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement*
  (4th ed.). Wiley. — segment mass fractions (Table 4.1) and radii of gyration.
* Dempster, W. T. (1955). *Space Requirements of the Seated Operator*. WADC
  Technical Report 55-159. — cadaver data underlying the Winter tables.
* Vicon Motion Systems. *Plug-in Gait kinetic modelling*, Nexus documentation. —
  the 15-segment representation and gyration ratios implemented here.
* Herr, H., & Popovic, M. (2008). Angular momentum in human walking.
  *Journal of Experimental Biology*, 211(4), 467–481. — the normalization
  convention `L / (m · v · h)`.

---

## Licence and citation

This project is licensed under the GNU General Public License v3.0 (GPLv3) --
see [LICENSE](LICENSE) for the full text. State how you would like the code
cited. If you use this module in a publication, please also cite the sources
of the inertia model you selected.
