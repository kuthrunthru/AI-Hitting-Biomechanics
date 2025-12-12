import os
import math
import uuid
import statistics

import cv2
import mediapipe as mp
import imageio.v2 as imageio
import streamlit as st

import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "pose_landmarker_full.task"

@st.cache_resource
def get_pose_landmarker():
    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    RunningMode = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)

def tasks_landmarks_to_px(lm_list, w, h):
    return [(lm.x * w, lm.y * h) for lm in lm_list]

# ----------------- CONSTANTS ----------------- #

GIF_FPS = 4
MAX_FRAME_HEIGHT = 480
ASSUMED_HEIGHT_INCHES = 60.0  # for inch-estimates of displacement


# ----------------- BASIC GEOMETRY HELPERS ----------------- #

def get_point(lm_list, idx, w, h):
    lm = lm_list[idx]
    return (lm.x * w, lm.y * h)


def angle_between(p1, p2):
    # angle of vector p1->p2 in radians
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def normalize_angle(a):
    # normalize to [-pi, pi]
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def rad_to_deg(r):
    return r * 180.0 / math.pi


def point_line_distance(px, py, x1, y1, x2, y2):
    """
    Distance from point (px, py) to the infinite line through (x1, y1) - (x2, y2).
    Legacy helper (no longer used in this version, but kept for possible future use).
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# ----------------- SCORING HELPERS ----------------- #

def color_dot(score):
    if score is None:
        return "⚪", "N/A"
    if score >= 80:
        return "🟢", "Excellent"
    if score >= 60:
        return "🟡", "Acceptable"
    return "🔴", "Needs Work"


def score_negative_move_inches(back_inches: float) -> int:
    """
    Negative move during load (backward movement, opposite pitcher, in inches).

    Ranges:
      Green (Excellent):   0–8"
      Yellow (Acceptable): 8–14"
      Red (Needs Work):    >14"
    """
    v = max(0.0, back_inches)

    # GREEN: 0–8"
    if v <= 8.0:
        score = 100 - (v / 8.0) * 10.0  # 100 → 90
        return int(round(score))

    # YELLOW: 8–14"
    if v <= 14.0:
        ratio = (v - 8.0) / 6.0         # 8→0, 14→1
        score = 90 - 25.0 * ratio       # 90 → 65
        return int(round(score))

    # RED: >14"
    capped = min(v, 24.0)
    ratio = (capped - 14.0) / 10.0      # 14→0, 24→1
    score = 65 - 35.0 * ratio           # 65 → 30
    return int(round(score))


def score_hip_coil_deg(coil_deg: float) -> int:
    """
    Hip coil (pelvis load) in degrees.

    Updated low-end ranges:
      Red:    0–2°
      Yellow: 2–5°
      Green:  5°+ (on the low side)

    High-end (too much coil) remains coached:
      Great:      20–35°
      Acceptable: 12–20° or 35–45°
      Needs work: >45°
    """
    v = max(0.0, coil_deg)

    # RED: 0–2
    if v < 2.0:
        ratio = v / 2.0  # 0→0, 2→1
        return int(round(35 + 20 * ratio))  # 35→55

    # YELLOW: 2–5
    if v < 5.0:
        ratio = (v - 2.0) / 3.0  # 2→0, 5→1
        return int(round(60 + 19 * ratio))  # 60→79

    # GREEN (low-side): 5–12
    if v < 12.0:
        ratio = (v - 5.0) / 7.0  # 5→0, 12→1
        return int(round(80 + 7 * ratio))   # 80→87

    # Great band
    if 20.0 <= v <= 35.0:
        mid = 27.5
        dist = abs(v - mid)
        score = 100 - dist * 1.2
        return int(round(max(90, min(score, 100))))

    # Still GREEN/usable: 12–20
    if 12.0 <= v < 20.0:
        mid = 16.0
        dist = abs(v - mid)
        score = 90 - dist * 2.0
        return int(round(max(80, min(score, 90))))

    # Yellow-ish: 35–45
    if 35.0 < v <= 45.0:
        mid = 40.0
        dist = abs(v - mid)
        score = 85 - dist * 2.0
        return int(round(max(70, min(score, 85))))

    # Red: >45
    capped = min(v, 60.0)
    ratio = (capped - 45.0) / 15.0      # 45→0, 60→1
    return int(round(70 - 30 * ratio))  # 70→40


def score_upper_torso_coil_deg(coil_deg: float) -> int:
    """
    Upper torso (shoulder) coil in degrees.

    Updated:
      Green:  12–40°
      Yellow: 8–12° or 40–50°
      Red:    <8° or >50°
    """
    v = max(0.0, coil_deg)

    # GREEN: 12–40
    if 12.0 <= v <= 40.0:
        mid = 26.0
        dist = abs(v - mid)
        score = 100 - dist * 0.8
        return int(round(max(85, min(score, 100))))

    # YELLOW: 8–12
    if 8.0 <= v < 12.0:
        ratio = (v - 8.0) / 4.0  # 8→0, 12→1
        return int(round(65 + 14 * ratio))  # 65→79

    # YELLOW: 40–50
    if 40.0 < v <= 50.0:
        ratio = (v - 40.0) / 10.0  # 40→0, 50→1
        return int(round(79 - 14 * ratio))  # 79→65

    # RED: <8
    if v < 8.0:
        ratio = min(v, 8.0) / 8.0
        return int(round(35 + 30 * ratio))  # 35→65

    # RED: >50
    capped = min(v, 65.0)
    ratio = (capped - 50.0) / 15.0  # 50→0, 65→1
    return int(round(65 - 35 * ratio))  # 65→30


def score_hip_shoulder_sep_foot_plant(sep_deg: float) -> int:
    """
    Hip–shoulder separation at foot plant (degrees).

    Updated to reflect elite HS testing:
      Green:  13–31°
      Yellow: 8–13° or 31–36°
      Red:    <8° or >36°
    """
    v = abs(sep_deg)

    # GREEN: 13–31
    if 13.0 <= v <= 31.0:
        mid = 22.0
        dist = abs(v - mid)
        score = 100 - dist * 0.9
        return int(round(max(85, min(score, 100))))

    # YELLOW: 8–13
    if 8.0 <= v < 13.0:
        ratio = (v - 8.0) / 5.0  # 8→0, 13→1
        return int(round(65 + 14 * ratio))  # 65→79

    # YELLOW: 31–36
    if 31.0 < v <= 36.0:
        ratio = (v - 31.0) / 5.0  # 31→0, 36→1
        return int(round(79 - 14 * ratio))  # 79→65

    # RED: <8
    if v < 8.0:
        ratio = min(v, 8.0) / 8.0
        return int(round(35 + 30 * ratio))  # 35→65

    # RED: >36
    capped = min(v, 50.0)
    ratio = (capped - 36.0) / 14.0  # 36→0, 50→1
    return int(round(65 - 35 * ratio))  # 65→30


def score_lead_shoulder_early_open_deg(sep_drop_deg: float) -> int | None:
    """
    Lead-Shoulder Early Open (deg): how much hip–shoulder separation is lost shortly AFTER foot plant.

    Updated to reflect elite range (0–6°):
      Green:  0–6°
      Yellow: 6–10°
      Red:    >10°
    """
    if sep_drop_deg is None:
        return None

    v = max(0.0, sep_drop_deg)

    # GREEN: 0–6
    if v <= 6.0:
        ratio = v / 6.0
        score = 100 - 12 * ratio  # 100→88
        return int(round(max(85, min(score, 100))))

    # YELLOW: 6–10
    if v <= 10.0:
        ratio = (v - 6.0) / 4.0
        score = 79 - 14 * ratio   # 79→65
        return int(round(max(60, min(score, 79))))

    # RED: >10
    capped = min(v, 18.0)
    ratio = (capped - 10.0) / 8.0
    score = 65 - 35 * ratio      # 65→30
    return int(round(max(20, score)))


def score_extension_through_contact_inches(ext_in: float) -> int | None:
    """
    Extension Through Contact (inches)
    Measures how far the lead hand continues moving FORWARD (toward pitcher) after a contact-proxy moment.

    Typical ranges (coachable, 2D-friendly):
      Excellent: 8–12"
      Acceptable: 5–8" or 12–15"
      Needs work: <5" or >15"
    """
    if ext_in is None:
        return None

    v = max(0.0, ext_in)

    if 8.0 <= v <= 12.0:
        mid = 10.0
        dist = abs(v - mid)
        score = 100 - dist * 2.0
        return int(round(max(90, min(score, 100))))

    if (5.0 <= v < 8.0) or (12.0 < v <= 15.0):
        mid = 6.5 if v < 8.0 else 13.5
        dist = abs(v - mid)
        score = 85 - dist * 6.0
        return int(round(max(70, min(score, 85))))

    if v < 5.0:
        ratio = min(v, 5.0) / 5.0
        return int(round(40 + 30 * ratio))

    capped = min(v, 22.0)
    ratio = (capped - 15.0) / 7.0
    return int(round(70 - 30 * ratio))  # 70→40


def score_rear_elbow_connection_inches(dist_in: float) -> int | None:
    """
    Rear Elbow Connection (inches from torso center).

    Updated to reflect elite range (approx 8–22) and your constraint:
      Green should go up to 15.

    Scoring:
      Green:  8–15"
      Yellow: 6–8" or 15–20"
      Red:    <6" or >20"
    """
    if dist_in is None:
        return None

    v = max(0.0, dist_in)

    # GREEN: 8–15
    if 8.0 <= v <= 15.0:
        mid = 11.5
        dist = abs(v - mid)
        score = 100 - dist * 1.8
        return int(round(max(85, min(score, 100))))

    # YELLOW: 6–8
    if 6.0 <= v < 8.0:
        ratio = (v - 6.0) / 2.0
        return int(round(65 + 14 * ratio))  # 65→79

    # YELLOW: 15–20
    if 15.0 < v <= 20.0:
        ratio = (v - 15.0) / 5.0
        return int(round(79 - 14 * ratio))  # 79→65

    # RED: <6 (too pinned / very tight)
    if v < 6.0:
        ratio = min(v, 6.0) / 6.0
        return int(round(35 + 30 * ratio))  # 35→65

    # RED: >20 (disconnected)
    capped = min(v, 28.0)
    ratio = (capped - 20.0) / 8.0
    return int(round(65 - 35 * ratio))  # 65→30


def score_kinematic_sequence(times_dict) -> int | None:
    """
    Score how well the sequence of peak angular speeds matches:
    pelvis → torso → arms → bat

    times_dict: {'pelvis': t_ms, 'torso': t_ms, 'arms': t_ms, 'bat': t_ms}
    """
    if not isinstance(times_dict, dict):
        return None
    required = ["pelvis", "torso", "arms", "bat"]
    if any(k not in times_dict for k in required):
        return None

    t = times_dict
    ideal = required[:]

    score = 100

    pair_penalty = 18
    close_penalty = 8
    min_separation_ms = 15.0

    for i in range(len(ideal)):
        for j in range(i + 1, len(ideal)):
            a = ideal[i]
            b = ideal[j]
            ta = t[a]
            tb = t[b]
            if ta >= tb:
                score -= pair_penalty
            else:
                if (tb - ta) < min_separation_ms:
                    score -= close_penalty

    score = max(40, min(score, 100))
    return int(round(score))


def score_head_off_ball_line(rise_in: float) -> int:
    """
    Head Rise After Foot Plant (inches).

    Excellent: 0–1"
    Acceptable: 1–3"
    Needs work: >3"
    """
    v = max(0.0, rise_in)

    if v <= 1.0:
        return 100

    if v <= 3.0:
        ratio = (v - 1.0) / 2.0
        return int(round(90 - 25 * ratio))  # 90→65

    capped = min(v, 8.0)
    ratio = (capped - 3.0) / 5.0  # 3→0, 8→1
    return int(round(65 - 35 * ratio))  # 65→30


# ----------------- METRIC EXPLANATIONS & TIPS ----------------- #

def metric_explanation(metric_id, raw_value):
    v = raw_value
    mid = metric_id

    # Negative Move
    if mid == "negative_move_in":
        base = (
            "This measures how far your body (using your hips as a reference) drifts backward away from the pitcher during your load. "
            "A small gather is fine, but big backward moves can hurt timing and balance."
        )
        if v is None:
            return base

        if v <= 8.0:
            detail = f" In this swing, your negative move was about {v:.2f}\", which is in the strong (green) range."
        elif v <= 14.0:
            detail = f" In this swing, your negative move was about {v:.2f}\", which is a bit large (yellow)."
        else:
            detail = f" In this swing, your negative move was about {v:.2f}\", which is very large (red) and can make timing much harder."
        return base + detail

    # Hip Coil
    if mid == "hip_coil_deg":
        base = (
            "This measures how much your hips coil or turn away from the pitcher during your load, before they start to fire forward. "
            "Enough coil helps store energy and create stretch."
        )
        if v is None:
            return base

        if v < 2.0:
            detail = f" In this swing, your hip coil was about {v:.1f}°, which is very low (red)."
        elif v < 5.0:
            detail = f" In this swing, your hip coil was about {v:.1f}°, which is modest (yellow)."
        elif v < 12.0:
            detail = f" In this swing, your hip coil was about {v:.1f}°, which clears the minimum coil target (green)."
        elif 20.0 <= v <= 35.0:
            detail = f" In this swing, your hip coil was about {v:.1f}°, which is a strong, healthy power range."
        elif v > 45.0:
            detail = f" In this swing, your hip coil was about {v:.1f}°, which is very large and can create timing/balance issues."
        else:
            detail = f" In this swing, your hip coil was about {v:.1f}°."
        return base + detail

    # Upper Torso Coil
    if mid == "upper_torso_coil_deg":
        base = (
            "This measures how much your upper body (shoulders/torso) coils away from the pitcher during your load. "
            "A good coil helps create stretch without over-wrapping."
        )
        if v is None:
            return base

        if v < 8.0:
            detail = f" In this swing, your upper torso coil was about {v:.1f}°, which is low (red)."
        elif v < 12.0:
            detail = f" In this swing, your upper torso coil was about {v:.1f}°, which is a bit light (yellow)."
        elif v <= 40.0:
            detail = f" In this swing, your upper torso coil was about {v:.1f}°, which is in the green range."
        elif v <= 50.0:
            detail = f" In this swing, your upper torso coil was about {v:.1f}°, which is getting high (yellow)."
        else:
            detail = f" In this swing, your upper torso coil was about {v:.1f}°, which is very high (red) and may indicate wrapping."
        return base + detail

    # Hip–Shoulder Separation at Foot Plant
    if mid == "hip_shoulder_sep_foot_plant_deg":
        base = (
            "This measures how much your hips and shoulders are twisted apart at foot plant. "
            "That 'stretch' can help produce bat speed. (2D estimate — focus on the band, not the exact digit.)"
        )
        if v is None:
            return base

        av = abs(v)
        if av < 8.0:
            detail = f" In this swing, your separation was about {v:.1f}°, which is low (red)."
        elif av < 13.0:
            detail = f" In this swing, your separation was about {v:.1f}°, which is slightly under the green target (yellow)."
        elif av <= 31.0:
            detail = f" In this swing, your separation was about {v:.1f}°, which is in the green target band."
        elif av <= 36.0:
            detail = f" In this swing, your separation was about {v:.1f}°, which is a bit high (yellow)."
        else:
            detail = f" In this swing, your separation was about {v:.1f}°, which is very high (red)."
        return base + detail

    # Lead-Shoulder Early Open
    if mid == "lead_shoulder_early_open_deg":
        base = (
            "This measures whether your lead shoulder opens too early after foot plant by checking how quickly separation collapses. "
            "Lower is generally better here."
        )
        if v is None:
            return base

        if v <= 6.0:
            detail = f" In this swing, you only lost about {v:.1f}° of separation right after landing (green)."
        elif v <= 10.0:
            detail = f" In this swing, you lost about {v:.1f}° shortly after landing (yellow)."
        else:
            detail = f" In this swing, you lost about {v:.1f}° right after landing (red), a sign of early opening/pull-off."
        return base + detail

    # Rear Elbow Connection
    if mid == "rear_elbow_connection_in":
        base = (
            "This measures how far your rear elbow moves away from your torso as the swing starts (just after foot plant). "
            "We’re aiming for a connected move without being overly pinned."
        )
        if v is None:
            return base

        if v < 6.0:
            detail = f" In this swing, the rear elbow stayed about {v:.2f}\" from the torso (red: very tight/pinned)."
        elif v < 8.0:
            detail = f" In this swing, the rear elbow was about {v:.2f}\" from the torso (yellow: a bit tight)."
        elif v <= 15.0:
            detail = f" In this swing, the rear elbow was about {v:.2f}\" from the torso (green)."
        elif v <= 20.0:
            detail = f" In this swing, the rear elbow was about {v:.2f}\" from the torso (yellow: starting to get away)."
        else:
            detail = f" In this swing, the rear elbow was about {v:.2f}\" from the torso (red: disconnected/flying)."
        return base + detail

    # Head Rise After Foot Plant
    if mid == "head_off_ball_line_in":
        base = (
            "This measures whether your head rises (pops up) after your front foot lands. "
            "We compare head height at foot plant to head height at (estimated) contact. Less rise is better."
        )
        if v is None:
            return base

        if v <= 1.0:
            detail = f" In this swing, your head rose about {v:.2f}\", which is excellent."
        elif v <= 3.0:
            detail = f" In this swing, your head rose about {v:.2f}\", which is workable but could be quieter."
        else:
            detail = f" In this swing, your head rose about {v:.2f}\", which is a lot and can hurt vision and consistency."
        return base + detail

    # Extension Through Contact
    if mid == "extension_through_contact_in":
        base = (
            "This measures how far your lead hand continues traveling forward (toward the pitcher) right after a contact-proxy moment. "
            "Good hitters drive through the ball and keep the barrel in the zone."
        )
        if v is None:
            return base

        if v < 5.0:
            detail = f" In this swing, your extension was about {v:.2f}\", which is short."
        elif 5.0 <= v < 8.0:
            detail = f" In this swing, your extension was about {v:.2f}\", which is workable."
        elif 8.0 <= v <= 12.0:
            detail = f" In this swing, your extension was about {v:.2f}\", which is strong."
        elif 12.0 < v <= 15.0:
            detail = f" In this swing, your extension was about {v:.2f}\", which is on the longer side but still okay."
        else:
            detail = f" In this swing, your extension was about {v:.2f}\", which is very long and may indicate casting/reaching."
        return base + detail

    # Kinematic Sequence
    if mid == "kinematic_sequence":
        base = (
            "This checks whether the hips, chest, arms, and bat reach their top turning speed in the right order. "
            "Ideal: hips → chest → arms → bat."
        )
        if not isinstance(v, dict):
            return base

        order = sorted(v.items(), key=lambda kv: kv[1])
        order_labels = " → ".join(name for name, _ in order)
        detail = f" In this swing, peak speeds happened in this order: **{order_labels}**."
        return base + detail

    return ""


def metric_improvement_tip(metric_id, raw_value):
    v = raw_value
    mid = metric_id
    if v is None:
        return None

    if mid == "negative_move_in":
        if v <= 8.0:
            return None
        return (
            "To reduce negative move, coil around your back hip instead of swaying your whole body backward. "
            "Try toe-tap or no-stride swings and freeze at the top of the load—your head/hips should still be centered."
        )

    if mid == "hip_coil_deg":
        if v >= 5.0 and v <= 45.0:
            return None
        if v < 5.0:
            return (
                "To add hip coil, feel the back hip turn slightly away as you load (without drifting back). "
                "Cue: 'close the front pocket a little' while keeping the head quiet."
            )
        return (
            "If hip coil is very large, shorten the load so you can land on time. "
            "Cue: 'coil small and strong'—power without extra sway."
        )

    if mid == "upper_torso_coil_deg":
        if 12.0 <= v <= 40.0:
            return None
        if v < 12.0:
            return (
                "To increase upper torso coil, let the lead shoulder turn in slightly during the load while keeping the head still. "
                "Cue: 'turn the chest a little into the catcher.'"
            )
        return (
            "If upper coil is big, reduce wrap so vision/timing stay stable. "
            "Cue: 'coil but keep the chest more toward the plate.'"
        )

    if mid == "hip_shoulder_sep_foot_plant_deg":
        if 13.0 <= abs(v) <= 31.0:
            return None
        if abs(v) < 13.0:
            return (
                "To build more separation at landing, practice coil-and-stride drills: land closed, then turn. "
                "Feel hips start opening while the shoulders stay quiet for a beat."
            )
        return (
            "If separation is very high, simplify the load so you can land balanced and on time. "
            "Avoid yanking the lead shoulder too far closed."
        )

    if mid == "lead_shoulder_early_open_deg":
        if v <= 6.0:
            return None
        return (
            "To prevent early opening, focus on 'land closed, then turn.' "
            "Try tee/front toss aiming middle-oppo gap while keeping the lead shoulder closed a fraction longer."
        )

    if mid == "rear_elbow_connection_in":
        if 8.0 <= v <= 15.0:
            return None
        if v < 8.0:
            return (
                "If the rear elbow is very tight, make sure you aren’t 'pinning' it—let it work around your side as you turn. "
                "Cue: 'connected, not stuck.'"
            )
        return (
            "If the rear elbow flies out, try a towel/ball-under-arm connection drill and start the swing with the turn (hips/chest), not a push."
        )

    if mid == "head_off_ball_line_in":
        if v <= 1.0:
            return None
        return (
            "To reduce head rise after foot plant, stay in your legs as you turn. "
            "Cue: 'nose under the same ceiling' from landing through contact."
        )

    if mid == "extension_through_contact_in":
        if 8.0 <= v <= 12.0:
            return None
        if v < 8.0:
            return (
                "To improve extension, feel 'long through the zone'—hands continue forward briefly after contact instead of cutting off."
            )
        return (
            "If extension is very long, make sure it’s from rotation, not reaching/casting. "
            "Keep the knob closer to the body early, then let extension happen out front."
        )

    if mid == "kinematic_sequence":
        if not isinstance(v, dict):
            return None
        ideal = ["pelvis", "torso", "arms", "bat"]
        actual = [name for name, _ in sorted(v.items(), key=lambda kv: kv[1])]
        if actual == ideal:
            return None
        return (
            "Use drills that let the lower body start the swing (step-behind / walk-through). "
            "Cue: 'hips start it, barrel finishes it.'"
        )

    return None


# ----------------- METRIC COMPUTATION ----------------- #

def _compute_peak_time_from_angles(angle_list, start_idx, end_idx, fps):
    """
    Helper: from a list of angles (radians), compute the time in ms when the
    absolute angular velocity is highest between start_idx..end_idx.
    """
    n = len(angle_list)
    if n < 2 or fps <= 0:
        return None

    start_idx = max(1, start_idx)
    end_idx = min(end_idx, n - 1)
    if end_idx <= start_idx:
        return None

    best_idx = None
    best_val = -1.0
    for i in range(start_idx, end_idx + 1):
        d = normalize_angle(angle_list[i] - angle_list[i - 1])
        vel = abs(d * fps)
        if vel > best_val:
            best_val = vel
            best_idx = i

    if best_idx is None:
        return None

    t_ms = (best_idx / fps) * 1000.0
    return t_ms


def _compute_overall_weighted_with_guardrails(metric_scores: dict, weights: dict) -> int | None:
    """
    Overall score = weighted average with guardrails:
      - Consider only numeric scores (int/float).
      - If we have at least 6 valid metrics AND the worst score <= 55,
        drop that single worst metric from the overall calculation.
    """
    if not isinstance(metric_scores, dict):
        return None

    valid = []
    for mid, sc in metric_scores.items():
        if isinstance(sc, (int, float)):
            w = float(weights.get(mid, 1.0))
            valid.append((mid, float(sc), w))

    if not valid:
        return None

    # Guardrail: optionally drop the single worst metric (overall only)
    if len(valid) >= 6:
        worst_mid, worst_score, worst_w = min(valid, key=lambda t: t[1])
        if worst_score <= 55.0:
            valid = [(mid, sc, w) for (mid, sc, w) in valid if mid != worst_mid]

    denom = sum(w for _, _, w in valid)
    if denom <= 1e-9:
        return None

    overall = sum(sc * w for _, sc, w in valid) / denom
    return int(round(overall))


def compute_hitting_metrics(frames_landmarks, handedness="R", fps=30.0):
    """
    frames_landmarks: list of dicts with:
      'hip_R', 'hip_L', 'sho_R', 'sho_L',
      'wrist_R', 'wrist_L', 'elbow_R', 'elbow_L',
      'ankle_R', 'ankle_L', 'head',
      'frame_w', 'frame_h'
    """
    n = len(frames_landmarks)
    if n < 5:
        return {}

    fps = fps if fps > 0 else 30.0

    pelvis_x = []
    torso_x = []
    torso_y = []
    head_x = []
    head_y = []
    pelvis_angle = []
    torso_angle = []
    arms_angle = []
    bat_angle = []
    sep_abs = []
    lead_ankle_y = []
    lead_wrist_x = []
    rear_elbow_to_torso_dist = []

    mp_pose = mp.solutions.pose
    PL = mp_pose.PoseLandmark

    lead = "L" if handedness == "R" else "R"

    for fr in frames_landmarks:
        hip_R = fr["hip_R"]
        hip_L = fr["hip_L"]
        sho_R = fr["sho_R"]
        sho_L = fr["sho_L"]
        wrist_R = fr["wrist_R"]
        wrist_L = fr["wrist_L"]
        elbow_R = fr["elbow_R"]
        elbow_L = fr["elbow_L"]
        ankle_R = fr["ankle_R"]
        ankle_L = fr["ankle_L"]
        head = fr["head"]

        pelvis_cx = (hip_R[0] + hip_L[0]) / 2.0
        pelvis_cy = (hip_R[1] + hip_L[1]) / 2.0
        torso_cx = (sho_R[0] + sho_L[0]) / 2.0
        torso_cy = (sho_R[1] + sho_L[1]) / 2.0

        pelvis_x.append(pelvis_cx)
        torso_x.append(torso_cx)
        torso_y.append(torso_cy)
        head_x.append(head[0])
        head_y.append(head[1])

        pa = normalize_angle(angle_between(hip_R, hip_L))
        ta = normalize_angle(angle_between(sho_R, sho_L))
        pelvis_angle.append(pa)
        torso_angle.append(ta)
        sep_abs.append(abs(normalize_angle(ta - pa)))

        if lead == "L":
            la = ankle_L
            lw = wrist_L
            lead_shoulder = sho_L
            rear_elbow = elbow_R
        else:
            la = ankle_R
            lw = wrist_R
            lead_shoulder = sho_R
            rear_elbow = elbow_L

        arms_angle.append(normalize_angle(angle_between(lead_shoulder, lw)))
        bat_angle.append(normalize_angle(angle_between((pelvis_cx, pelvis_cy), lw)))

        lead_ankle_y.append(la[1])
        lead_wrist_x.append(lw[0])

        rear_elbow_to_torso_dist.append(distance(rear_elbow, (torso_cx, torso_cy)))

    # Body height estimate (median head-to-ankle distance)
    heights = []
    for fr in frames_landmarks:
        head = fr["head"]
        aR = fr["ankle_R"]
        aL = fr["ankle_L"]
        mid_ank = ((aR[0] + aL[0]) / 2.0, (aR[1] + aL[1]) / 2.0)
        h = distance(head, mid_ank)
        if h > 0:
            heights.append(h)
    if heights:
        heights.sort()
        body_height = heights[len(heights) // 2]
    else:
        body_height = 200.0

    # Forward direction based on pelvis motion
    idx_a = int(n * 0.2)
    idx_b = int(n * 0.8)
    dx = pelvis_x[idx_b] - pelvis_x[idx_a]
    forward_dir = 1.0 if dx >= 0 else -1.0

    pelvis_forward = [x * forward_dir for x in pelvis_x]
    wrist_forward = [x * forward_dir for x in lead_wrist_x]

    # Foot plant: lead ankle maximum y in first 2/3rds of frames
    fd_end = max(1, int(n * 2 / 3))
    foot_down_idx = max(range(fd_end), key=lambda i: lead_ankle_y[i])

    # Contact (legacy bound): lead wrist most forward
    contact_idx = max(range(n), key=lambda i: wrist_forward[i])

    # --- Negative Move During Load (Backward Only) ---
    if foot_down_idx >= 0:
        segment = pelvis_forward[0: foot_down_idx + 1]
        if segment:
            baseline = segment[0]
            min_val = min(segment)
            back_px = max(0.0, baseline - min_val)
            negative_move_in = (back_px / max(body_height, 1e-6)) * ASSUMED_HEIGHT_INCHES
        else:
            negative_move_in = 0.0
    else:
        negative_move_in = 0.0

    # --- Hip Coil (Pelvis Load) ---
    if pelvis_angle:
        load_end = max(0, min(foot_down_idx, n - 1))
        baseline_end = min(load_end, 4)
        baseline_samples = pelvis_angle[0: baseline_end + 1] if baseline_end >= 0 else [pelvis_angle[0]]
        baseline_pa = sum(baseline_samples) / len(baseline_samples) if baseline_samples else pelvis_angle[0]
        coil_rads = [abs(normalize_angle(pa - baseline_pa)) for pa in pelvis_angle[0: load_end + 1]]
        hip_coil_deg = rad_to_deg(max(coil_rads)) if coil_rads else 0.0
    else:
        hip_coil_deg = 0.0

    # --- Upper Torso Coil (Upper Body Load) ---
    if torso_angle:
        load_end = max(0, min(foot_down_idx, n - 1))
        baseline_end = min(load_end, 4)
        baseline_samples_t = torso_angle[0: baseline_end + 1] if baseline_end >= 0 else [torso_angle[0]]
        baseline_ta = sum(baseline_samples_t) / len(baseline_samples_t) if baseline_samples_t else torso_angle[0]
        coil_rads_t = [abs(normalize_angle(ta - baseline_ta)) for ta in torso_angle[0: load_end + 1]]
        upper_torso_coil_deg = rad_to_deg(max(coil_rads_t)) if coil_rads_t else 0.0
    else:
        upper_torso_coil_deg = 0.0

    # --- Hip–Shoulder Separation at Foot Plant ---
    if sep_abs:
        start_idx = max(0, foot_down_idx - 1)
        end_idx = min(len(sep_abs) - 1, foot_down_idx + 2)
        window = sep_abs[start_idx: end_idx + 1]
        sep_fd_rad = max(window) if window else sep_abs[foot_down_idx]
    else:
        sep_fd_rad = 0.0
    sep_fd_deg = rad_to_deg(sep_fd_rad)

    # --- Lead-Shoulder Early Open ---
    lead_shoulder_early_open_deg = None
    if sep_abs and 0 <= foot_down_idx < n:
        sep_fp_abs_deg = rad_to_deg(abs(sep_abs[foot_down_idx]))
        window_ms = 150.0
        w_frames = max(1, int(round((window_ms / 1000.0) * fps)))
        w_start = foot_down_idx
        w_end = min(n - 1, foot_down_idx + w_frames)

        if w_end > w_start:
            min_sep_after = min(sep_abs[w_start: w_end + 1])
            min_sep_after_deg = rad_to_deg(abs(min_sep_after))
            lead_shoulder_early_open_deg = max(0.0, sep_fp_abs_deg - min_sep_after_deg)
        else:
            lead_shoulder_early_open_deg = 0.0

    # --- Rear Elbow Connection ---
    rear_elbow_connection_in = None
    if rear_elbow_to_torso_dist and 0 <= foot_down_idx < n:
        window_ms = 150.0
        w_frames = max(2, int(round((window_ms / 1000.0) * fps)))
        w_end = min(n - 1, foot_down_idx + w_frames)

        max_dist_px = max(rear_elbow_to_torso_dist[foot_down_idx: w_end + 1])
        rear_elbow_connection_in = (max_dist_px / max(body_height, 1e-6)) * ASSUMED_HEIGHT_INCHES

    # --- Pseudo-contact index (used for extension window end) ---
    if n >= 2:
        search_start = max(1, foot_down_idx)
        search_end = min(contact_idx, n - 1)
        if search_end <= search_start:
            pseudo_contact_idx = contact_idx
        else:
            best_vel = None
            best_idx = None
            for i in range(search_start, search_end + 1):
                dv = (wrist_forward[i] - wrist_forward[i - 1]) * fps
                if best_vel is None or dv > best_vel:
                    best_vel = dv
                    best_idx = i
            pseudo_contact_idx = best_idx if best_idx is not None else contact_idx
    else:
        pseudo_contact_idx = contact_idx

    # --- Head Rise After Foot Plant (inches, upward only; foot plant -> contact) ---
    if head_y and 0 <= foot_down_idx < n and 0 <= contact_idx < n:
        head_fd_y = head_y[foot_down_idx]
        head_contact_y = head_y[contact_idx]
        rise_px = max(0.0, head_fd_y - head_contact_y)  # y decreases when head rises
        head_rise_in = (rise_px / max(body_height, 1e-6)) * ASSUMED_HEIGHT_INCHES
    else:
        head_rise_in = 0.0

    # --- Extension Through Contact ---
    extension_through_contact_in = None
    if n >= 3 and 0 <= pseudo_contact_idx < n - 1:
        window_ms = 150.0
        w_frames = max(2, int(round((window_ms / 1000.0) * fps)))
        w_end = min(n - 1, pseudo_contact_idx + w_frames)

        base_fwd = wrist_forward[pseudo_contact_idx]
        max_forward_px = 0.0
        for i in range(pseudo_contact_idx + 1, w_end + 1):
            forward_px = wrist_forward[i] - base_fwd
            if forward_px > max_forward_px:
                max_forward_px = forward_px

        extension_through_contact_in = (max_forward_px / max(body_height, 1e-6)) * ASSUMED_HEIGHT_INCHES

    # --- Kinematic Sequence ---
    seq_end_idx = max(1, contact_idx)
    pelvis_peak_t = _compute_peak_time_from_angles(pelvis_angle, 0, seq_end_idx, fps)
    torso_peak_t = _compute_peak_time_from_angles(torso_angle, 0, seq_end_idx, fps)
    arms_peak_t = _compute_peak_time_from_angles(arms_angle, 0, seq_end_idx, fps)
    bat_peak_t = _compute_peak_time_from_angles(bat_angle, 0, seq_end_idx, fps)

    if None in (pelvis_peak_t, torso_peak_t, arms_peak_t, bat_peak_t):
        kinematic_sequence_raw = None
        kinematic_sequence_score = None
    else:
        kinematic_sequence_raw = {
            "pelvis": round(pelvis_peak_t, 1),
            "torso": round(torso_peak_t, 1),
            "arms": round(arms_peak_t, 1),
            "bat": round(bat_peak_t, 1),
        }
        kinematic_sequence_score = score_kinematic_sequence(kinematic_sequence_raw)

    # --- RAW METRICS ---
    metrics = {}
    metrics["negative_move_in"] = round(negative_move_in, 2)
    metrics["hip_coil_deg"] = round(hip_coil_deg, 1)
    metrics["upper_torso_coil_deg"] = round(upper_torso_coil_deg, 1)
    metrics["hip_shoulder_sep_foot_plant_deg"] = round(sep_fd_deg, 1)
    metrics["lead_shoulder_early_open_deg"] = None if lead_shoulder_early_open_deg is None else round(lead_shoulder_early_open_deg, 1)
    metrics["rear_elbow_connection_in"] = None if rear_elbow_connection_in is None else round(rear_elbow_connection_in, 2)
    metrics["head_off_ball_line_in"] = round(head_rise_in, 2)
    metrics["extension_through_contact_in"] = None if extension_through_contact_in is None else round(extension_through_contact_in, 2)
    metrics["kinematic_sequence"] = kinematic_sequence_raw

    # --- SCORES ---
    metric_score_values = {}
    metric_score_values["negative_move_in"] = score_negative_move_inches(negative_move_in)
    metric_score_values["hip_coil_deg"] = score_hip_coil_deg(hip_coil_deg)
    metric_score_values["upper_torso_coil_deg"] = score_upper_torso_coil_deg(upper_torso_coil_deg)
    metric_score_values["hip_shoulder_sep_foot_plant_deg"] = score_hip_shoulder_sep_foot_plant(sep_fd_deg)
    metric_score_values["lead_shoulder_early_open_deg"] = score_lead_shoulder_early_open_deg(lead_shoulder_early_open_deg)
    metric_score_values["rear_elbow_connection_in"] = score_rear_elbow_connection_inches(rear_elbow_connection_in)
    metric_score_values["head_off_ball_line_in"] = score_head_off_ball_line(head_rise_in)
    metric_score_values["extension_through_contact_in"] = score_extension_through_contact_inches(extension_through_contact_in)
    metric_score_values["kinematic_sequence"] = kinematic_sequence_score

    metrics["scores"] = metric_score_values

    # Overall weights (performance realism) — per your guidance:
    # lowest weight: negative move, upper torso coil, lead-shoulder early open, head rise
    OVERALL_WEIGHTS = {
        "kinematic_sequence": 1.8,
        "hip_shoulder_sep_foot_plant_deg": 1.6,
        "hip_coil_deg": 1.4,
        "rear_elbow_connection_in": 1.2,
        "extension_through_contact_in": 1.1,

        "negative_move_in": 0.6,
        "upper_torso_coil_deg": 0.6,
        "lead_shoulder_early_open_deg": 0.6,
        "head_off_ball_line_in": 0.6,
    }

    overall = _compute_overall_weighted_with_guardrails(metric_score_values, OVERALL_WEIGHTS)
    metrics["overall_score"] = overall

    return metrics


# ----------------- SUMMARY / TOP 3 ISSUES ----------------- #

METRIC_LABELS = {
    "negative_move_in": "Negative Move (Backward During Load)",
    "hip_coil_deg": "Hip Coil (Pelvis Load)",
    "upper_torso_coil_deg": "Upper Torso Coil (Upper Body Load)",
    "hip_shoulder_sep_foot_plant_deg": "Hip–Shoulder Separation at Foot Plant",
    "lead_shoulder_early_open_deg": "Lead-Shoulder Early Open (Separation Loss After Plant)",
    "rear_elbow_connection_in": "Rear Elbow Connection",
    "head_off_ball_line_in": "Head Rise After Foot Plant",
    "extension_through_contact_in": "Extension Through Contact",
    "kinematic_sequence": "Kinematic Sequence (Peak Speeds Order)",
}

METRIC_UNITS = {
    "negative_move_in": "inches",
    "hip_coil_deg": "degrees",
    "upper_torso_coil_deg": "degrees",
    "hip_shoulder_sep_foot_plant_deg": "degrees",
    "lead_shoulder_early_open_deg": "degrees",
    "rear_elbow_connection_in": "inches",
    "head_off_ball_line_in": "inches",
    "extension_through_contact_in": "inches",
    "kinematic_sequence": "ms timings",
}

# These weights are used ONLY for Top-3 focus priority (not overall).
METRIC_WEIGHTS = {
    "negative_move_in": 1.1,
    "hip_coil_deg": 1.2,
    "upper_torso_coil_deg": 1.2,
    "hip_shoulder_sep_foot_plant_deg": 1.1,
    "lead_shoulder_early_open_deg": 1.3,
    "rear_elbow_connection_in": 1.25,
    "head_off_ball_line_in": 1.1,
    "extension_through_contact_in": 1.2,
    "kinematic_sequence": 1.3,
}


def top_issues_text(metrics):
    score_map = metrics["scores"]
    issues = []
    for mid, score in score_map.items():
        if score is None:
            continue
        if score >= 80:
            continue
        weight = METRIC_WEIGHTS.get(mid, 1.0)
        priority = (80 - score) * weight
        issues.append((priority, mid, score))

    if not issues:
        return []

    issues.sort(reverse=True)
    top = issues[:3]

    out_sentences = []
    for _, mid, score in top:
        v = metrics.get(mid, None)

        if mid == "negative_move_in":
            sent = "Reduce how far you drift backward during the load so you stay centered and can move forward on time."
        elif mid == "hip_coil_deg":
            if v is not None and v < 5.0:
                sent = "Create a bit more hip coil during the load so you store energy before you fire."
            elif v is not None and v > 45.0:
                sent = "Shorten the hip coil to keep timing and balance repeatable."
            else:
                sent = "Tune hip coil so it stays strong and repeatable."
        elif mid == "upper_torso_coil_deg":
            if v is not None and v < 12.0:
                sent = "Add a little more upper torso coil during the load to help create stretch."
            else:
                sent = "Reduce upper torso wrapping so vision and timing stay stable."
        elif mid == "hip_shoulder_sep_foot_plant_deg":
            if v is not None and abs(v) < 13.0:
                sent = "Land with a bit more hip–shoulder separation so you have usable stretch at foot plant."
            else:
                sent = "Keep separation controlled at landing so timing stays simple."
        elif mid == "lead_shoulder_early_open_deg":
            sent = "Keep your lead shoulder closed longer after foot plant so you don’t pull off early."
        elif mid == "rear_elbow_connection_in":
            sent = "Keep the rear elbow connected (without pinning) right after foot plant so the swing stays turn-driven."
        elif mid == "head_off_ball_line_in":
            sent = "Stop the head from rising after foot plant—stay in your legs so your eyes stay steady."
        elif mid == "extension_through_contact_in":
            sent = "Drive through contact longer so the hands continue forward briefly after the contact moment."
        elif mid == "kinematic_sequence":
            sent = "Improve the order of peak speeds (hips → chest → arms → bat) so energy transfers cleanly."
        else:
            sent = f"Work on improving {METRIC_LABELS.get(mid, mid)}."

        out_sentences.append(sent)

    return out_sentences


# ----------------- ORIENTATION & TRACKING ----------------- #

def detect_best_rotation(input_path):
    """
    Tries multiple rotations (including 180°) and picks the one most likely to be "heads up".
    Uses MediaPipe Tasks (local model) to avoid runtime downloads on Streamlit Cloud.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    best_rot = None
    best_score = -1e18

    landmarker = get_pose_landmarker()

    for rot in [None, "cw", "ccw", "180"]:
        test = frame.copy()
        if rot == "cw":
            test = cv2.rotate(test, cv2.ROTATE_90_CLOCKWISE)
        elif rot == "ccw":
            test = cv2.rotate(test, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot == "180":
            test = cv2.rotate(test, cv2.ROTATE_180)

        h, w = test.shape[:2]
        frame_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            continue

        lm_norm = result.pose_landmarks[0]  # 33 landmarks
        pts = tasks_landmarks_to_px(lm_norm, w, h)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        vertical_span = max(ys) - min(ys)
        horizontal_span = max(xs) - min(xs)

        # Standard indices: 0=nose, 27=left ankle, 28=right ankle
        nose_y = pts[0][1]
        ankles_y = (pts[27][1] + pts[28][1]) / 2.0

        heads_up_bonus = 1500.0 if nose_y < ankles_y else -1500.0
        score = (vertical_span - horizontal_span) + heads_up_bonus

        if score > best_score:
            best_score = score
            best_rot = rot

    return best_rot

def create_tracked_frames_and_landmarks(input_path, rotation=None, max_frames=200):
    """
    Returns:
      tracked_frames: list of RGB images (sampled for GIF)
      frames_landmarks: list of dicts for metric computation
      all_frames_rgb: list of RGB frames aligned with frames_landmarks
      pose_landmarks_list: raw Mediapipe pose landmarks for each frame
      fps: video frames per second
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    step = max(1, total_frames // max_frames)
    targets = set(range(0, total_frames, step))

    tracked_frames = []
    frames_landmarks = []
    all_frames_rgb = []
    pose_landmarks_list = []

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    PL = mp_pose.PoseLandmark

    with mp_pose.Pose(model_complexity=2) as pose:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation == "cw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == "ccw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == "180":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            h, w = frame.shape[:2]
            if h > MAX_FRAME_HEIGHT:
                scale = MAX_FRAME_HEIGHT / float(h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                h, w = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                lm_list = result.pose_landmarks.landmark

                hip_R = get_point(lm_list, PL.RIGHT_HIP.value, w, h)
                hip_L = get_point(lm_list, PL.LEFT_HIP.value, w, h)
                sho_R = get_point(lm_list, PL.RIGHT_SHOULDER.value, w, h)
                sho_L = get_point(lm_list, PL.LEFT_SHOULDER.value, w, h)
                elbow_R = get_point(lm_list, PL.RIGHT_ELBOW.value, w, h)
                elbow_L = get_point(lm_list, PL.LEFT_ELBOW.value, w, h)
                wrist_R = get_point(lm_list, PL.RIGHT_WRIST.value, w, h)
                wrist_L = get_point(lm_list, PL.LEFT_WRIST.value, w, h)
                ankle_R = get_point(lm_list, PL.RIGHT_ANKLE.value, w, h)
                ankle_L = get_point(lm_list, PL.LEFT_ANKLE.value, w, h)
                head = get_point(lm_list, PL.NOSE.value, w, h)

                frames_landmarks.append(
                    {
                        "hip_R": hip_R,
                        "hip_L": hip_L,
                        "sho_R": sho_R,
                        "sho_L": sho_L,
                        "elbow_R": elbow_R,
                        "elbow_L": elbow_L,
                        "wrist_R": wrist_R,
                        "wrist_L": wrist_L,
                        "ankle_R": ankle_R,
                        "ankle_L": ankle_L,
                        "head": head,
                        "frame_w": w,
                        "frame_h": h,
                    }
                )
                all_frames_rgb.append(frame_rgb.copy())
                pose_landmarks_list.append(result.pose_landmarks)

                if i in targets:
                    frame_for_gif = frame_rgb.copy()
                    mp_draw.draw_landmarks(frame_for_gif, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    tracked_frames.append(frame_for_gif)

            i += 1

    cap.release()
    return (
        tracked_frames,
        frames_landmarks,
        all_frames_rgb,
        pose_landmarks_list,
        fps,
    )


def save_tracked_gif(frames, output_path="tracked.gif", fps=GIF_FPS):
    if not frames:
        raise RuntimeError("No frames for GIF")
    duration = 1.0 / float(fps)
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
    return output_path


# ----------------- SKELETON DRAWING ----------------- #

def draw_skeleton_frame(pose_landmarks, base_image):
    img = base_image.copy()
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_draw.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return img


# ----------------- STREAMLIT UI ----------------- #

st.set_page_config(page_title="AI Hitting Biomechanics", layout="wide")
st.title("⚾ AI Hitting Biomechanics – 2D Swing Report")

uploaded_file = st.file_uploader("Upload a side-view swing video", type=["mp4", "mov", "avi"])
handedness_label = st.selectbox("Hitter", ["Right-handed", "Left-handed"])
handedness = "R" if handedness_label.startswith("Right") else "L"

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if uploaded_file is not None:
    run_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(uploaded_file.name)[1]
    tmp_video = f"uploaded_swing_{run_id}{ext}"
    with open(tmp_video, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Analyze Swing"):
        with st.spinner("Analyzing swing..."):
            rotation = detect_best_rotation(tmp_video)
            (
                tracked_frames,
                frames_landmarks,
                frames_rgb,
                pose_landmarks_list,
                fps,
            ) = create_tracked_frames_and_landmarks(
                tmp_video, rotation=rotation, max_frames=200
            )
            metrics = compute_hitting_metrics(frames_landmarks, handedness, fps=fps)
            gif_path = save_tracked_gif(tracked_frames, output_path=f"tracked_{run_id}.gif")

            st.session_state.analysis = {
                "metrics": metrics,
                "gif_path": gif_path,
                "frames_landmarks": frames_landmarks,
                "frames_rgb": frames_rgb,
                "pose_landmarks": pose_landmarks_list,
            }

if st.session_state.analysis is not None:
    metrics = st.session_state.analysis["metrics"]
    gif_path = st.session_state.analysis["gif_path"]
    frames_landmarks = st.session_state.analysis["frames_landmarks"]
    frames_rgb = st.session_state.analysis["frames_rgb"]
    pose_landmarks_list = st.session_state.analysis["pose_landmarks"]

    left, right = st.columns([1.6, 1])

    with left:
        overall = metrics.get("overall_score", None)

        st.markdown("## Final Score")
        if overall is None:
            st.markdown("⚪ **N/A**")
        else:
            # Make it prominent
            dot, grade = color_dot(overall)
            st.markdown(
                f"""
                <div style="padding: 18px 18px; border-radius: 14px; border: 1px solid #e6e6e6; background: #ffffff;">
                    <div style="font-size: 18px; opacity: 0.85;">Overall Swing Score</div>
                    <div style="font-size: 56px; font-weight: 800; line-height: 1.0; margin-top: 6px;">
                        {overall} <span style="font-size: 24px; font-weight: 600; margin-left: 10px;">{dot} {grade}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Per your request: no extra language here — go straight to Top 3.
        issues = top_issues_text(metrics)
        st.markdown("#### Top 3 things to focus on next:")
        if issues:
            for i, sent in enumerate(issues, start=1):
                st.markdown(f"{i}. {sent}")
        else:
            st.markdown("Your main movement patterns look solid. Keep training these moves and monitor them over time.")

        st.markdown("---")
        st.markdown("## Detailed Metrics")

        def show_metric(mid):
            raw = metrics.get(mid, None)
            score = metrics["scores"].get(mid, None)
            label = METRIC_LABELS.get(mid, mid)
            units = METRIC_UNITS.get(mid, "")

            if raw is not None and score is None:
                if isinstance(raw, float):
                    raw_str = f"{raw:.2f}"
                else:
                    raw_str = str(raw)
                unit_str = f" {units}" if units and not isinstance(raw, dict) else ""
                st.markdown(f"**{label}** ⚪ — N/A | Raw: {raw_str}{unit_str}")
                explanation = metric_explanation(mid, raw)
                if explanation:
                    st.markdown(f"**What this means:** {explanation}")
                st.markdown("---")
                return

            if raw is None or score is None:
                st.markdown(f"**{label}** ⚪ — N/A")
                return

            dot, grade = color_dot(score)

            if isinstance(raw, float):
                raw_str = f"{raw:.2f}"
            else:
                raw_str = str(raw)
            unit_str = f" {units}" if units and not isinstance(raw, dict) else ""

            st.markdown(
                f"**{label}** {dot} — {grade} | Raw: {raw_str}{unit_str} | Score: {score}"
            )

            explanation = metric_explanation(mid, raw)
            if explanation:
                st.markdown(f"**What this means:** {explanation}")

            tip = metric_improvement_tip(mid, raw)
            if tip and score < 80:
                st.markdown(f"**How to improve:** {tip}")

            st.markdown("---")

        st.markdown("### Load")
        show_metric("negative_move_in")
        show_metric("hip_coil_deg")
        show_metric("upper_torso_coil_deg")

        st.markdown("### Rotation, Separation, and Sequencing")
        show_metric("hip_shoulder_sep_foot_plant_deg")
        show_metric("lead_shoulder_early_open_deg")
        show_metric("rear_elbow_connection_in")
        show_metric("kinematic_sequence")

        st.markdown("### Bat Path & Contact")
        show_metric("head_off_ball_line_in")
        show_metric("extension_through_contact_in")

    with right:
        st.header("Tracked Swing")
        st.image(gif_path, width=480)

        st.header("Skeleton Explorer")
        if frames_landmarks and pose_landmarks_list:
            idx = st.slider(
                "Frame",
                min_value=0,
                max_value=len(frames_landmarks) - 1,
                value=0,
                step=1,
            )
            base = frames_rgb[idx]
            skel_img = draw_skeleton_frame(pose_landmarks_list[idx], base)
            st.image(
                skel_img,
                caption=f"Mediapipe Skeleton – Frame {idx + 1}/{len(frames_landmarks)}",
                width=480,
            )

