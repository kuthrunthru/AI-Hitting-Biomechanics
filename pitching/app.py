import os
import cv2
import math
import hashlib
import tempfile
import numpy as np
import streamlit as st
import mediapipe as mp

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Pitch Mechanics Analyzer (v2 report layout)")
st.title("Pitch Mechanics Analyzer")

# =========================================================
# GLOBAL SETTINGS / DEFAULTS
# =========================================================
CONF_THRESH = 0.4
SAMPLE_ORIENT_FRAMES = 12

CLIP_QUALITY_GOOD = 0.55
CLIP_QUALITY_OK = 0.40

# Release picker defaults
REL_WIN_START_OFF = 5     # frames after FFP
REL_WIN_END_OFF = 45      # frames after FFP
REL_WARN_DIFF_FRAMES = 8  # warn if user release differs from auto by more than this

# Torso-open defaults
TORSO_OPEN_DEG = 25.0
TORSO_OPEN_CONSEC = 3
TORSO_SEARCH_MAX_OFF = 45  # search window after FFP

DEFAULT_MAX_PROCESS_WIDTH = 960

# Pelvis coil bands (degrees of pelvis–shoulder angular separation @ MKL)
PELVIS_COIL_GREEN_MIN = 5.0
PELVIS_COIL_YELLOW_MIN = 2.0

# Posture bands (degrees trunk tilt away from vertical @ MKL; absolute)
POSTURE_GREEN_MAX = 10.0
POSTURE_YELLOW_MAX = 18.0

# Head-behind-hip bands (normalized by hip width) @ MKL
HEAD_BEHIND_HIP_GREEN_MIN = 0.15
HEAD_BEHIND_HIP_YELLOW_MIN = 0.00

# Pelvis drift timing bands (WHEN pelvis begins moving toward plate, as % of MKL→FFP window)
# pct = (onset_frame - MKL) / (FFP - MKL)
PELVIS_DRIFT_GREEN_LO = 0.35
PELVIS_DRIFT_GREEN_HI = 0.70
PELVIS_DRIFT_YELLOW_LO = 0.20
PELVIS_DRIFT_YELLOW_HI = 0.85

# ---------------------------------------------------------
# Hip–Shoulder Separation bands (degrees) @ Foot Strike (FFP)
# ---------------------------------------------------------
HIP_SHO_SEP_GREEN_MIN = 35.0
HIP_SHO_SEP_YELLOW_MIN = 25.0

# ---------------------------------------------------------
# Stride Length bands (normalized by body height proxy) @ Foot Strike (FFP)
# ---------------------------------------------------------
# stride_norm = (lead_ankle_x - back_ankle_x) * plate_dir_sign / height_proxy
# Height proxy ~ shoulder-mid to ankle-mid distance at/near FFP (scale normalization).
STRIDE_GREEN_LO = 0.80
STRIDE_GREEN_HI = 1.00
STRIDE_YELLOW_LO = 0.70
STRIDE_YELLOW_HI = 1.10

# ---------------------------------------------------------
# Trunk Tilt @ Foot Strike (FFP) — signed forward-lean proxy
# ---------------------------------------------------------
# We measure how much the trunk (pelvis-mid -> shoulder-mid) leans TOWARD home plate at FFP.
# Positive = forward lean toward plate, Negative = leaning back / away from plate.
TRUNK_FWD_GREEN_LO = 8.0
TRUNK_FWD_GREEN_HI = 22.0
TRUNK_FWD_YELLOW_LO = 2.0
TRUNK_FWD_YELLOW_HI = 30.0


# =========================================================
# GEOMETRY / SIGNAL HELPERS
# =========================================================
def norm(v):
    return float(np.linalg.norm(v))


def wrap180(a):
    if a is None:
        return None
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


def moving_median(series, k=7):
    arr = np.array(series, dtype=np.float32)
    out = np.copy(arr)
    half = k // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        window = arr[lo:hi]
        window = window[~np.isnan(window)]
        out[i] = np.nan if window.size == 0 else np.median(window)
    return out


def finite_diff(series):
    s = np.array(series, dtype=np.float32)
    d = np.full_like(s, np.nan)
    d[1:] = s[1:] - s[:-1]
    return d


def forward_fill_nan(arr):
    a = np.array(arr, dtype=np.float32)
    last = np.nan
    for i in range(len(a)):
        if np.isnan(a[i]) and not np.isnan(last):
            a[i] = last
        elif not np.isnan(a[i]):
            last = a[i]
    return a


def line_angle_deg(p1, p2):
    d = p2 - p1
    if norm(d) < 1e-9:
        return None
    return float(math.degrees(math.atan2(d[1], d[0])))


def trunk_tilt_from_vertical_deg(hip_mid_xy, shoulder_mid_xy):
    """
    Returns absolute tilt of trunk away from vertical (degrees).
    0° = perfectly vertical (stacked). Higher = more lean/tilt.
    (Used for Posture at MKL.)
    """
    if hip_mid_xy is None or shoulder_mid_xy is None:
        return None
    v = shoulder_mid_xy - hip_mid_xy  # points "up" the body
    if norm(v) < 1e-9:
        return None

    # With image coords, +y is down. Vertical "up" is (0, -1).
    # Angle from vertical: atan2(x, -y)
    ang = math.degrees(math.atan2(float(v[0]), float(-v[1])))
    return abs(float(ang))


def trunk_forward_tilt_signed_deg(hip_mid_xy, shoulder_mid_xy, plate_dir_sign):
    """
    Returns SIGNED forward trunk tilt from vertical (degrees), where:
      + = trunk leans toward home plate
      - = trunk leans back / away from home plate
      0 = stacked/vertical

    Computed from pelvis-mid -> shoulder-mid vector.
    """
    if hip_mid_xy is None or shoulder_mid_xy is None:
        return None
    v = shoulder_mid_xy - hip_mid_xy
    if norm(v) < 1e-9:
        return None

    # Convert x into "toward plate" axis by multiplying plate_dir_sign.
    vx_plate = float(v[0]) * float(plate_dir_sign)

    # Angle from vertical: atan2(x_plate, -y)
    ang = float(math.degrees(math.atan2(vx_plate, float(-v[1]))))
    return ang  # signed


# =========================================================
# ORIENTATION HELPERS
# =========================================================
def apply_orientation(frame_bgr, mode: str):
    if mode == "None":
        return frame_bgr
    if mode == "90° CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if mode == "180°":
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if mode == "270° CW":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


def _resize_keep_aspect(frame_bgr, max_w=None, max_h=None):
    if max_w is None and max_h is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = 1.0
    if max_w is not None:
        scale = min(scale, max_w / max(w, 1))
    if max_h is not None:
        scale = min(scale, max_h / max(h, 1))
    if scale >= 0.999:
        return frame_bgr
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def read_frame_at(video_path, idx, chosen_mode, max_display_width=1280):
    cap = cv2.VideoCapture(video_path)
    if idx < 0:
        idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame = apply_orientation(frame, chosen_mode)
    if max_display_width and max_display_width > 0:
        frame = _resize_keep_aspect(frame, max_w=max_display_width)
    return frame


# =========================================================
# POSE MODEL + LANDMARK ARRAYS (pickle-safe for caching)
# =========================================================
@st.cache_resource
def get_pose_model(model_complexity=2, smooth_landmarks=True):
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(model_complexity),
        smooth_landmarks=bool(smooth_landmarks),
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def lm_to_array(lm, conf_thresh=CONF_THRESH):
    """(33,4): [x,y,z,visibility], NaN for missing/low-vis."""
    if lm is None:
        return None
    arr = np.full((33, 4), np.nan, dtype=np.float32)
    for i, p in enumerate(lm):
        vis = float(getattr(p, "visibility", 0.0))
        if vis < conf_thresh:
            continue
        arr[i, 0] = float(p.x)
        arr[i, 1] = float(p.y)
        arr[i, 2] = float(getattr(p, "z", 0.0))
        arr[i, 3] = vis
    return arr


def tracked_joint_count_arr(lm_arr, conf_thresh=CONF_THRESH):
    if lm_arr is None:
        return 0, 33
    vis = lm_arr[:, 3]
    good = int(np.sum(np.isfinite(vis) & (vis >= conf_thresh)))
    return good, 33


def get_xy_from_arr(lm_arr, idx, conf_thresh=CONF_THRESH):
    if lm_arr is None:
        return None
    vis = lm_arr[idx, 3]
    if not np.isfinite(vis) or vis < conf_thresh:
        return None
    return np.array([lm_arr[idx, 0], lm_arr[idx, 1]], dtype=np.float32)


POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def draw_pose_from_arr(frame_bgr, lm_arr, w, h):
    img = frame_bgr.copy()
    if lm_arr is None:
        return img

    for a, b in POSE_CONNECTIONS:
        pa = lm_arr[int(a)]
        pb = lm_arr[int(b)]
        va = pa[3]
        vb = pb[3]
        if not np.isfinite(va) or not np.isfinite(vb) or va < CONF_THRESH or vb < CONF_THRESH:
            continue
        x1, y1 = int(pa[0] * w), int(pa[1] * h)
        x2, y2 = int(pb[0] * w), int(pb[1] * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    key_ids = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
        mp.solutions.pose.PoseLandmark.NOSE,
    ]
    for idx in key_ids:
        p = lm_arr[int(idx)]
        v = p[3]
        if not np.isfinite(v) or v < CONF_THRESH:
            continue
        x, y = int(p[0] * w), int(p[1] * h)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    return img


def clip_quality_label(q: float) -> str:
    if q >= CLIP_QUALITY_GOOD:
        return "Good"
    if q >= CLIP_QUALITY_OK:
        return "OK"
    return "Poor"


# =========================================================
# SCORING / BANDS HELPERS
# =========================================================
def dot_status(status: str) -> str:
    s = (status or "").strip().upper()
    if s == "GREEN":
        return "🟢"
    if s == "YELLOW":
        return "🟡"
    if s == "RED":
        return "🔴"
    return "⚪"


def score_pelvis_coil_deg(coil_deg):
    if coil_deg is None or (isinstance(coil_deg, float) and np.isnan(coil_deg)):
        return None, "NA"

    x = float(coil_deg)
    if x < PELVIS_COIL_YELLOW_MIN:
        status = "RED"
        score = 49.0 * max(0.0, min(1.0, x / max(PELVIS_COIL_YELLOW_MIN, 1e-6)))
    elif x < PELVIS_COIL_GREEN_MIN:
        status = "YELLOW"
        t = (x - PELVIS_COIL_YELLOW_MIN) / max(PELVIS_COIL_GREEN_MIN - PELVIS_COIL_YELLOW_MIN, 1e-6)
        score = 50.0 + 29.0 * max(0.0, min(1.0, t))
    else:
        status = "GREEN"
        t = (x - PELVIS_COIL_GREEN_MIN) / 10.0
        score = 80.0 + 20.0 * max(0.0, min(1.0, t))

    return float(score), status


def score_posture_deg(tilt_deg):
    if tilt_deg is None or (isinstance(tilt_deg, float) and np.isnan(tilt_deg)):
        return None, "NA"

    x = float(tilt_deg)

    if x <= POSTURE_GREEN_MAX:
        status = "GREEN"
        t = 1.0 - max(0.0, min(1.0, x / max(POSTURE_GREEN_MAX, 1e-6)))
        score = 90.0 + 10.0 * t
    elif x <= POSTURE_YELLOW_MAX:
        status = "YELLOW"
        t = (x - POSTURE_GREEN_MAX) / max(POSTURE_YELLOW_MAX - POSTURE_GREEN_MAX, 1e-6)
        score = 89.0 - 29.0 * max(0.0, min(1.0, t))
    else:
        status = "RED"
        over = min(40.0, x - POSTURE_YELLOW_MAX)
        t = over / 40.0
        score = 59.0 - 59.0 * max(0.0, min(1.0, t))

    return float(score), status


def score_head_behind_hip(norm_dx):
    if norm_dx is None or (isinstance(norm_dx, float) and np.isnan(norm_dx)):
        return None, "NA"

    x = float(norm_dx)

    if x < HEAD_BEHIND_HIP_YELLOW_MIN:
        status = "RED"
        t = (x - (-0.25)) / 0.25
        t = max(0.0, min(1.0, t))
        score = 0.0 + 59.0 * t
    elif x < HEAD_BEHIND_HIP_GREEN_MIN:
        status = "YELLOW"
        t = (x - HEAD_BEHIND_HIP_YELLOW_MIN) / max(HEAD_BEHIND_HIP_GREEN_MIN - HEAD_BEHIND_HIP_YELLOW_MIN, 1e-6)
        t = max(0.0, min(1.0, t))
        score = 60.0 + 19.0 * t
    else:
        status = "GREEN"
        t = (x - HEAD_BEHIND_HIP_GREEN_MIN) / 0.15
        t = max(0.0, min(1.0, t))
        score = 80.0 + 20.0 * t

    return float(score), status


def score_pelvis_drift_timing(pct):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return None, "NA"

    x = float(pct)
    x = max(0.0, min(1.0, x))

    if PELVIS_DRIFT_GREEN_LO <= x <= PELVIS_DRIFT_GREEN_HI:
        status = "GREEN"
        center = 0.5 * (PELVIS_DRIFT_GREEN_LO + PELVIS_DRIFT_GREEN_HI)
        halfw = 0.5 * (PELVIS_DRIFT_GREEN_HI - PELVIS_DRIFT_GREEN_LO)
        t = 1.0 - min(1.0, abs(x - center) / max(halfw, 1e-6))
        score = 85.0 + 15.0 * t
    elif (PELVIS_DRIFT_YELLOW_LO <= x < PELVIS_DRIFT_GREEN_LO) or (PELVIS_DRIFT_GREEN_HI < x <= PELVIS_DRIFT_YELLOW_HI):
        status = "YELLOW"
        if x < PELVIS_DRIFT_GREEN_LO:
            dist = (PELVIS_DRIFT_GREEN_LO - x) / max(PELVIS_DRIFT_GREEN_LO - PELVIS_DRIFT_YELLOW_LO, 1e-6)
        else:
            dist = (x - PELVIS_DRIFT_GREEN_HI) / max(PELVIS_DRIFT_YELLOW_HI - PELVIS_DRIFT_GREEN_HI, 1e-6)
        dist = max(0.0, min(1.0, dist))
        score = 80.0 - 25.0 * dist
    else:
        status = "RED"
        if x < PELVIS_DRIFT_YELLOW_LO:
            dist = (PELVIS_DRIFT_YELLOW_LO - x) / max(PELVIS_DRIFT_YELLOW_LO, 1e-6)
        else:
            dist = (x - PELVIS_DRIFT_YELLOW_HI) / max(1.0 - PELVIS_DRIFT_YELLOW_HI, 1e-6)
        dist = max(0.0, min(1.0, dist))
        score = 54.0 - 54.0 * dist

    return float(score), status


def score_hip_shoulder_separation_deg(sep_deg):
    if sep_deg is None or (isinstance(sep_deg, float) and np.isnan(sep_deg)):
        return None, "NA"

    x = float(sep_deg)

    if x < HIP_SHO_SEP_YELLOW_MIN:
        status = "RED"
        t = max(0.0, min(1.0, x / max(HIP_SHO_SEP_YELLOW_MIN, 1e-6)))
        score = 54.0 * t
    elif x < HIP_SHO_SEP_GREEN_MIN:
        status = "YELLOW"
        t = (x - HIP_SHO_SEP_YELLOW_MIN) / max(HIP_SHO_SEP_GREEN_MIN - HIP_SHO_SEP_YELLOW_MIN, 1e-6)
        t = max(0.0, min(1.0, t))
        score = 55.0 + 24.0 * t
    else:
        status = "GREEN"
        t = (x - HIP_SHO_SEP_GREEN_MIN) / 20.0
        t = max(0.0, min(1.0, t))
        score = 80.0 + 20.0 * t

    return float(score), status


def score_stride_length_norm(stride_norm):
    """
    Stride Length (normalized by body height proxy) @ Foot Strike.

    stride_norm ~= (front ankle - back ankle) along plate direction, divided by a height proxy.
    Bands:
      GREEN: [0.80, 1.00]
      YELLOW: [0.70, 0.80) or (1.00, 1.10]
      RED: <0.70 or >1.10
    """
    if stride_norm is None or (isinstance(stride_norm, float) and np.isnan(stride_norm)):
        return None, "NA"

    x = float(stride_norm)
    x = max(0.0, min(1.35, x))  # keep sane

    if STRIDE_GREEN_LO <= x <= STRIDE_GREEN_HI:
        status = "GREEN"
        center = 0.5 * (STRIDE_GREEN_LO + STRIDE_GREEN_HI)
        halfw = 0.5 * (STRIDE_GREEN_HI - STRIDE_GREEN_LO)
        t = 1.0 - min(1.0, abs(x - center) / max(halfw, 1e-6))
        score = 85.0 + 15.0 * t
    elif (STRIDE_YELLOW_LO <= x < STRIDE_GREEN_LO) or (STRIDE_GREEN_HI < x <= STRIDE_YELLOW_HI):
        status = "YELLOW"
        if x < STRIDE_GREEN_LO:
            dist = (STRIDE_GREEN_LO - x) / max(STRIDE_GREEN_LO - STRIDE_YELLOW_LO, 1e-6)
        else:
            dist = (x - STRIDE_GREEN_HI) / max(STRIDE_YELLOW_HI - STRIDE_GREEN_HI, 1e-6)
        dist = max(0.0, min(1.0, dist))
        score = 80.0 - 25.0 * dist
    else:
        status = "RED"
        if x < STRIDE_YELLOW_LO:
            dist = (STRIDE_YELLOW_LO - x) / max(STRIDE_YELLOW_LO, 1e-6)
        else:
            dist = (x - STRIDE_YELLOW_HI) / max(1.35 - STRIDE_YELLOW_HI, 1e-6)
        dist = max(0.0, min(1.0, dist))
        score = 54.0 - 54.0 * dist

    return float(score), status


def score_trunk_tilt_ffp_deg(trunk_fwd_deg):
    """
    Trunk Tilt @ FFP (signed, toward plate).

    GREEN:  [TRUNK_FWD_GREEN_LO, TRUNK_FWD_GREEN_HI]
    YELLOW: [TRUNK_FWD_YELLOW_LO, TRUNK_FWD_GREEN_LO) or (TRUNK_FWD_GREEN_HI, TRUNK_FWD_YELLOW_HI]
    RED:    <TRUNK_FWD_YELLOW_LO (too upright/back) or >TRUNK_FWD_YELLOW_HI (too much forward dive)
    """
    if trunk_fwd_deg is None or (isinstance(trunk_fwd_deg, float) and np.isnan(trunk_fwd_deg)):
        return None, "NA"

    x = float(trunk_fwd_deg)

    # Treat negative as "leaning back" -> red quickly
    if x < TRUNK_FWD_YELLOW_LO:
        status = "RED"
        # map [-15 .. YELLOW_LO] -> [0..59] (clamped)
        t = (x - (-15.0)) / max(TRUNK_FWD_YELLOW_LO - (-15.0), 1e-6)
        t = max(0.0, min(1.0, t))
        score = 0.0 + 59.0 * t
        return float(score), status

    # within sensible forward-lean range
    if TRUNK_FWD_GREEN_LO <= x <= TRUNK_FWD_GREEN_HI:
        status = "GREEN"
        center = 0.5 * (TRUNK_FWD_GREEN_LO + TRUNK_FWD_GREEN_HI)
        halfw = 0.5 * (TRUNK_FWD_GREEN_HI - TRUNK_FWD_GREEN_LO)
        t = 1.0 - min(1.0, abs(x - center) / max(halfw, 1e-6))
        score = 85.0 + 15.0 * t
    elif (TRUNK_FWD_YELLOW_LO <= x < TRUNK_FWD_GREEN_LO) or (TRUNK_FWD_GREEN_HI < x <= TRUNK_FWD_YELLOW_HI):
        status = "YELLOW"
        if x < TRUNK_FWD_GREEN_LO:
            dist = (TRUNK_FWD_GREEN_LO - x) / max(TRUNK_FWD_GREEN_LO - TRUNK_FWD_YELLOW_LO, 1e-6)
        else:
            dist = (x - TRUNK_FWD_GREEN_HI) / max(TRUNK_FWD_YELLOW_HI - TRUNK_FWD_GREEN_HI, 1e-6)
        dist = max(0.0, min(1.0, dist))
        score = 80.0 - 25.0 * dist
    else:
        status = "RED"
        over = min(30.0, x - TRUNK_FWD_YELLOW_HI)
        t = max(0.0, min(1.0, over / 30.0))
        score = 54.0 - 54.0 * t

    return float(score), status


# =========================================================
# COACH LANGUAGE HELPERS
# =========================================================
def pelvis_coil_language(coil_deg, status):
    if coil_deg is None or (isinstance(coil_deg, float) and np.isnan(coil_deg)) or status == "NA":
        observed = "We couldn’t score pelvis coil because the hip/shoulder landmarks weren’t reliably detected at max leg lift."
        tips = (
            "Try a steadier clip (full body in frame, good lighting, minimal motion blur). "
            "If the pitcher is partially cut off or too small in frame, landmark tracking at leg lift can fail."
        )
        return observed, tips

    x = float(coil_deg)

    if status == "GREEN":
        observed = (
            f"At max leg lift, you showed strong early loading: pelvis–torso separation was about **{x:.1f}°**. "
            "That indicates good coil and sets up better stretch later at foot strike."
        )
        tips = (
            "Maintain this by keeping the lift controlled and the pelvis ‘loaded’ while the torso stays quiet. "
            "Cue: **“Ride the back hip—hips resist, torso waits.”**"
        )
    elif status == "YELLOW":
        observed = (
            f"At max leg lift, pelvis–torso separation was **{x:.1f}°**. "
            "That’s some coil, but it’s modest—there may be early turning where the pelvis and torso move together."
        )
        tips = (
            "Work on a slower, controlled leg lift with a brief **pause-at-the-top** (1-count) while staying stacked. "
            "Focus on keeping the pelvis loaded and avoiding early opening. Cue: **“Stay closed longer at the top.”**"
        )
    else:
        observed = (
            f"At max leg lift, pelvis–torso separation was only **{x:.1f}°**, suggesting minimal coil. "
            "This often means the hips and torso are turning together too early, reducing stored energy."
        )
        tips = (
            "Use **lift-and-hold** drills (pause at max leg lift) and **step-behind / walk-through** drills to feel the hips load first. "
            "Cue: **“Back hip loaded—don’t turn yet.”**"
        )

    return observed, tips


def posture_language(tilt_deg, status):
    if tilt_deg is None or (isinstance(tilt_deg, float) and np.isnan(tilt_deg)) or status == "NA":
        observed = (
            "We couldn’t confidently score posture because the hip/shoulder landmarks weren’t reliably tracked at max leg lift."
        )
        tips = (
            "Try a steadier clip with full body in frame, better lighting, and less blur. "
            "If the pitcher is cut off or too small, landmark tracking at leg lift can fail."
        )
        return observed, tips

    x = float(tilt_deg)

    if status == "GREEN":
        observed = (
            f"At max leg lift, you stayed stacked with your chest over your hips. "
            f"Your trunk tilt was about **{x:.1f}°** from vertical, which is a strong, repeatable posture."
        )
        tips = (
            "Keep this feel: **ribs stacked over hips** as the knee lifts. "
            "Cue: **‘Grow tall at the top.’**\n\n"
            "Maintain the same stacked trunk as you start moving toward the plate—don’t let the chest tip first."
        )
    elif status == "YELLOW":
        observed = (
            f"At max leg lift, your trunk leaned away from vertical by about **{x:.1f}°**. "
            "That’s a noticeable tilt, which can throw off balance and make timing harder to repeat."
        )
        tips = (
            "Goal: stay stacked longer at the top. Try a **1-count pause at max leg lift** while keeping the chest tall.\n\n"
            "Cue: **‘Belt buckle under ribs’** or **‘ribs over hips’**. If you feel tipping, slow the lift and re-center before moving out.\n\n"
            "Drill: **knee-up holds (2 seconds)** × 5 reps, then repeat with a slow move-out."
        )
    else:
        observed = (
            f"At max leg lift, your trunk tilt was about **{x:.1f}°** from vertical. "
            "This is a significant lean that often causes early drift or a rushed move to foot strike."
        )
        tips = (
            "Start with balance + stack. Drill: **knee-up holds (2–3 seconds)**, focusing on a tall torso and calm head.\n\n"
            "Use a wall or mirror drill: lift to the top without letting your chest fall or your head drift. "
            "Cue: **‘Tall spine, quiet head.’**\n\n"
            "Slow the leg lift and control the move-out. The goal is: **hips and torso move together**, not chest tipping early."
        )

    return observed, tips


def head_behind_hip_language(norm_dx, status):
    if norm_dx is None or (isinstance(norm_dx, float) and np.isnan(norm_dx)) or status == "NA":
        observed = (
            "We couldn’t score head position because the head/hip landmarks weren’t reliably tracked at max leg lift."
        )
        tips = (
            "Try a steadier clip with the full body in frame and good lighting. "
            "If the head or hips are blurred/cut off, tracking at the top of leg lift can fail."
        )
        return observed, tips

    x = float(norm_dx)

    if status == "GREEN":
        observed = (
            f"At max leg lift, your head stayed **clearly behind** your back hip (normalized offset **{x:.2f}**). "
            "That’s a strong balance position that helps keep your move-out under control."
        )
        tips = (
            "Maintain this feel by keeping the head quiet as the knee lifts. "
            "Cue: **“Stay tall and keep the head back.”**\n\n"
            "As you start moving forward, let the **hips lead** while the head stays calm—avoid reaching with the chest."
        )
    elif status == "YELLOW":
        observed = (
            f"At max leg lift, your head was **about even with** your back hip (normalized offset **{x:.2f}**). "
            "This is workable, but it can drift forward early if the move-out is rushed."
        )
        tips = (
            "Try a **1-count pause at the top** and feel the head stay slightly behind the back hip.\n\n"
            "Cue: **“Head stays back while hips start forward.”**\n\n"
            "Drill: **knee-up holds** (2 seconds) × 5 reps, then slow move-out keeping the head quiet."
        )
    else:
        observed = (
            f"At max leg lift, your head was **in front of** your back hip (normalized offset **{x:.2f}**). "
            "That usually means early drift, which can force you to rush into foot strike and rush the arm."
        )
        tips = (
            "Start with balance. Drill: **knee-up holds (2–3 seconds)** with a ‘quiet head’ focus.\n\n"
            "Use a wall/mirror drill: lift to the top without letting the head drift toward the plate.\n\n"
            "Cue: **“Stack first, then ride the hips.”** Let the pelvis start forward before the head/chest."
        )

    return observed, tips


def pelvis_drift_timing_language(pct, onset_frame, MKL, FFP):
    if pct is None or (isinstance(pct, float) and np.isnan(pct)) or onset_frame is None:
        observed = (
            "We couldn’t confidently detect the first clear forward move of the pelvis between max leg lift and foot strike. "
            "This usually happens when hip landmarks are noisy, the view isn’t true side-view, or the pelvis motion is very small in the clip."
        )
        tips = (
            "Try filming a steadier true side-view (full body in frame, good lighting). "
            "If possible, avoid zooming and keep the camera fixed.\n\n"
            "Mechanically, think **“ride the hips forward smoothly”** rather than staying completely still and then rushing late."
        )
        return observed, tips

    x = float(pct)
    frm = int(onset_frame)
    pct_str = f"{100.0 * x:.0f}%"

    if PELVIS_DRIFT_GREEN_LO <= x <= PELVIS_DRIFT_GREEN_HI:
        observed = (
            f"Your pelvis began moving toward the plate around **Frame {frm}**, about **{pct_str}** of the way from max leg lift to foot strike. "
            "That’s a good ‘on-time’ move—early enough to build momentum, but not so early that you leak forward."
        )
        tips = (
            "Keep this pattern: a **controlled move-out** where the hips lead and the head stays calm.\n\n"
            "Cue: **“Slow early… fast late.”** Let the pelvis start drifting while you stay stacked."
        )
    elif x < PELVIS_DRIFT_GREEN_LO:
        observed = (
            f"Your pelvis started drifting forward early (around **Frame {frm}**, ~**{pct_str}** of the MKL→FFP window). "
            "Early drift often pulls the head/chest forward and can make foot strike timing harder to repeat."
        )
        tips = (
            "Try a cleaner top: **pause 1-count at max leg lift** and feel the pelvis stay ‘loaded’ before moving out.\n\n"
            "Cue: **“Stack first, then ride.”** Keep the head behind the back hip as the hips start forward.\n\n"
            "Drill: **knee-up holds (2 seconds)** → then slow move-out keeping the chest tall."
        )
    else:
        observed = (
            f"Your pelvis didn’t clearly start moving forward until late (around **Frame {frm}**, ~**{pct_str}** of the MKL→FFP window). "
            "Late drift often forces a rushed move into foot strike and can make the arm feel late."
        )
        tips = (
            "Work on starting the move-out earlier and smoother—avoid ‘staying back’ too long then lunging.\n\n"
            "Cue: **“Ride the back hip forward.”** Think of the pelvis gliding forward while the torso stays quiet.\n\n"
            "Drill: **slow rocker / glide drill** (small forward glide from the top) focusing on continuous motion."
        )

    return observed, tips


def hip_shoulder_separation_language(sep_deg, status):
    if sep_deg is None or (isinstance(sep_deg, float) and np.isnan(sep_deg)) or status == "NA":
        observed = (
            "We couldn’t confidently score hip–shoulder separation at foot strike because the shoulder/hip landmarks "
            "weren’t reliably tracked at (or near) foot strike."
        )
        tips = (
            "Try a clearer true side-view with full body in frame, better lighting, and less motion blur. "
            "If the torso/hips are partially cut off, separation often becomes NA."
        )
        return observed, tips

    x = float(sep_deg)

    if status == "GREEN":
        observed = (
            f"At foot strike, you showed strong hip–shoulder separation (**{x:.1f}°**). "
            "That’s a good sign your hips are leading and your torso is staying closed long enough to create stretch."
        )
        tips = (
            "Keep the same sequence: **hips start rotation, shoulders stay back** until the front foot lands.\n\n"
            "Cue: **“Hips go—shoulders wait.”**"
        )
    elif status == "YELLOW":
        observed = (
            f"At foot strike, hip–shoulder separation was **{x:.1f}°**. "
            "That’s some separation, but you may be opening the torso a little early or not getting the hips out in front enough."
        )
        tips = (
            "Focus on letting the **hips rotate first** while keeping the chest/shoulders quiet into foot strike.\n\n"
            "Cues: **“Show the back pocket”** (hips lead) and **“keep the numbers back”** (shoulders stay closed).\n\n"
            "Drill idea: **step-behind throws** or **separation holds** (pause just before foot strike with shoulders closed)."
        )
    else:
        observed = (
            f"At foot strike, hip–shoulder separation was low (**{x:.1f}°**). "
            "That often means the shoulders are turning with the hips (early torso rotation) or the hips aren’t leading enough."
        )
        tips = (
            "Prioritize sequencing: **ride the hips forward** into foot strike and keep the shoulders closed longer.\n\n"
            "Cue: **“Land closed, then fire.”**\n\n"
            "Drill idea: **closed-to-open drill** (stride/land with shoulders closed, then rotate). "
            "Also consider a lighter ‘hands stay back’ feel as the front foot lands."
        )

    return observed, tips


def stride_length_language(stride_norm, status):
    """
    Coach-friendly narrative for stride length.

    stride_norm is normalized to a body-height proxy, so it works across zoom levels.
    """
    if stride_norm is None or (isinstance(stride_norm, float) and np.isnan(stride_norm)) or status == "NA":
        observed = (
            "We couldn’t confidently score stride length because the ankle/shoulder landmarks weren’t reliably tracked at (or near) foot strike."
        )
        tips = (
            "Try a clearer side-view with both feet visible at foot strike (avoid cutting off the bottom of the frame). "
            "Good lighting + less blur helps ankle tracking a lot."
        )
        return observed, tips

    x = float(stride_norm)

    if status == "GREEN":
        observed = (
            f"At foot strike, your stride length was **{x:.2f}× body height (proxy)**. "
            "That’s in a strong range—enough to create momentum without forcing a reach."
        )
        tips = (
            "Maintain this by letting the hips lead the move-out and allowing the stride to happen naturally.\n\n"
            "Cue: **“Ride the hips—let the stride happen.”**"
        )
    elif status == "YELLOW":
        if x < STRIDE_GREEN_LO:
            observed = (
                f"At foot strike, your stride length was **{x:.2f}× body height (proxy)**—a bit short. "
                "Short strides can limit momentum and make it harder to get into a strong front-side block."
            )
            tips = (
                "Focus on a smoother, earlier move-out so you can cover ground without rushing.\n\n"
                "Cue: **“Glide out, don’t jump.”**\n\n"
                "Drill: **walk-throughs / step-behind throws** to feel momentum and a longer stride without reaching."
            )
        else:
            observed = (
                f"At foot strike, your stride length was **{x:.2f}× body height (proxy)**—a bit long. "
                "Over-striding can pull you forward early, make you land soft, and force the arm to play catch-up."
            )
            tips = (
                "Aim for a stride that lands firm and under control. Let the front foot land **under the hip line**, not reaching.\n\n"
                "Cue: **“Land strong, not far.”**\n\n"
                "Drill: **stride-to-balance** (stride and stick the landing) focusing on a stable front side."
            )
    else:
        if x < STRIDE_YELLOW_LO:
            observed = (
                f"At foot strike, your stride length was **{x:.2f}× body height (proxy)**—very short. "
                "This often reduces momentum and can push you into spinning/rotating early."
            )
            tips = (
                "Build momentum earlier from the top and let the pelvis glide.\n\n"
                "Cue: **“Hips forward early and smooth.”**\n\n"
                "Drill: **rocker drill** (small controlled glide) → then normal delivery keeping the same glide feel."
            )
        else:
            observed = (
                f"At foot strike, your stride length was **{x:.2f}× body height (proxy)**—very long. "
                "This is commonly linked to reaching, landing soft, and losing posture into foot strike."
            )
            tips = (
                "Shorten slightly and prioritize a firm, balanced landing.\n\n"
                "Cue: **“Stride to stability.”** (land and stick)\n\n"
                "Drill: **stride-to-stick** with a 1–2 second hold at foot strike before finishing the throw."
            )

    return observed, tips


def trunk_tilt_language(trunk_fwd_deg, status):
    if trunk_fwd_deg is None or (isinstance(trunk_fwd_deg, float) and np.isnan(trunk_fwd_deg)) or status == "NA":
        observed = (
            "We couldn’t confidently score trunk tilt at foot strike because the hip/shoulder landmarks "
            "weren’t reliably tracked at (or near) foot strike."
        )
        tips = (
            "Try a clearer true side-view with full body in frame, good lighting, and less motion blur. "
            "If the torso/hips are partially cut off, trunk-tilt at foot strike can become NA."
        )
        return observed, tips

    x = float(trunk_fwd_deg)

    if status == "GREEN":
        observed = (
            f"At foot strike, your trunk was leaning forward toward the plate by about **{x:.1f}°**. "
            "That’s a strong, athletic position—enough forward intent without diving."
        )
        tips = (
            "Maintain this by letting the **hips lead** into foot strike while keeping the chest controlled.\n\n"
            "Cue: **“Hips lead, chest follows.”**\n\n"
            "Goal feel: you land balanced, then rotate hard—without your head/chest shooting forward early."
        )
    elif status == "YELLOW":
        if x < TRUNK_FWD_GREEN_LO:
            observed = (
                f"At foot strike, your trunk forward lean was about **{x:.1f}°**—a bit upright. "
                "Being too upright at foot strike can reduce intent/extension and sometimes forces more ‘spin’ instead of drive."
            )
            tips = (
                "Allow a little more forward move with the pelvis while keeping the head quiet.\n\n"
                "Cue: **“Glide, then land.”** (hips move, chest stays controlled)\n\n"
                "Drill: **rocker / glide drill** into a firm foot strike with the chest slightly over the front hip."
            )
        else:
            observed = (
                f"At foot strike, your trunk forward lean was about **{x:.1f}°**—a bit much. "
                "Extra forward tilt can look like ‘diving’ and may rush the arm or soften the front-side block."
            )
            tips = (
                "Keep the forward move, but delay the chest a touch so you land more stable.\n\n"
                "Cue: **“Land tall, then rotate.”**\n\n"
                "Drill: **stride-to-stick** (hold 1–2 seconds at foot strike) focusing on a firm front leg and controlled chest."
            )
    else:
        if x < TRUNK_FWD_YELLOW_LO:
            observed = (
                f"At foot strike, your trunk forward lean was only **{x:.1f}°** (or even leaning back). "
                "This often shows up as ‘staying back’ too long, then rushing rotation/arm late."
            )
            tips = (
                "Start the move-out earlier with the pelvis and let the trunk come with it—without tipping early.\n\n"
                "Cue: **“Ride the back hip forward.”**\n\n"
                "Drill: **slow glide from the top** into a firm foot strike, keeping the head calm and moving with the hips."
            )
        else:
            observed = (
                f"At foot strike, your trunk forward lean was **{x:.1f}°**, which is a strong ‘dive’ forward. "
                "This can collapse posture, reduce block, and force the arm to play catch-up."
            )
            tips = (
                "Prioritize **front-side stability** and keep the chest from passing the front foot too early.\n\n"
                "Cue: **“Strong front leg—rotate over it.”**\n\n"
                "Drill: **stride-to-stick**, then add a controlled throw—land stable first, then finish."
            )

    return observed, tips


# =========================================================
# AUTO ORIENTATION
# =========================================================
def score_orientation(video_path, pose_obj, conf_thresh, sample_frames=SAMPLE_ORIENT_FRAMES):
    modes = ["None", "90° CW", "180°", "270° CW"]
    cap = cv2.VideoCapture(video_path)

    scores = {m: [] for m in modes}
    count = 0
    while count < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        for m in modes:
            fr = apply_orientation(frame, m)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = pose_obj.process(rgb)
            lm = res.pose_landmarks.landmark if res.pose_landmarks else None
            arr = lm_to_array(lm, conf_thresh=conf_thresh)
            good, total = tracked_joint_count_arr(arr, conf_thresh=conf_thresh)
            scores[m].append(good / max(total, 1))

    cap.release()

    best_mode = "None"
    best_val = -1.0
    for m in modes:
        val = float(np.mean(scores[m])) if len(scores[m]) else -1.0
        if val > best_val:
            best_val = val
            best_mode = m
    return best_mode, best_val


# =========================================================
# VIDEO PROCESSING (cached, arrays only)
# =========================================================
@st.cache_data(show_spinner=False)
def process_video_landmarks_cached(
    video_hash: str,
    video_path: str,
    chosen_mode: str,
    conf_thresh: float,
    max_process_width: int,
    model_complexity: int,
    smooth_landmarks: bool,
):
    pose_obj = get_pose_model(model_complexity=model_complexity, smooth_landmarks=smooth_landmarks)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    lms_arr = []
    pose_detected = 0
    pose_quality = []
    first_hw = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_orientation(frame, chosen_mode)

        if max_process_width and max_process_width > 0:
            frame = _resize_keep_aspect(frame, max_w=max_process_width)

        if first_hw is None:
            first_hw = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_obj.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None

        arr = lm_to_array(lm, conf_thresh=conf_thresh)
        lms_arr.append(arr)

        if arr is not None:
            pose_detected += 1
            good, total = tracked_joint_count_arr(arr, conf_thresh=conf_thresh)
            pose_quality.append(good / max(total, 1))
        else:
            pose_quality.append(0.0)

    cap.release()

    n = len(lms_arr)
    if n == 0:
        return None

    h, w = first_hw if first_hw else (0, 0)
    fps = float(fps) if fps and fps > 1e-6 else 0.0
    duration_s = (n / fps) if fps > 1e-6 else 0.0

    return {
        "lms_arr": lms_arr,
        "pose_detected": int(pose_detected),
        "pose_quality": [float(x) for x in pose_quality],
        "n": int(n),
        "h": int(h),
        "w": int(w),
        "fps": float(fps),
        "duration_s": float(duration_s),
        "frame_count_hint": int(frame_count_hint),
    }


# =========================================================
# EVENTS HELPERS
# =========================================================
def nearest_good_frame(idx, pose_quality, min_q=0.40, radius=8):
    n = len(pose_quality)
    if 0 <= idx < n and pose_quality[idx] >= min_q:
        return idx, True
    best = None
    best_d = 9999
    for d in range(1, radius + 1):
        for cand in (idx - d, idx + d):
            if 0 <= cand < n and pose_quality[cand] >= min_q:
                if d < best_d:
                    best = cand
                    best_d = d
        if best is not None:
            return best, False
    return idx, False


def find_first_stable_window2(v1, v2, start_idx, abs_thresh1, abs_thresh2, window):
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    n_ = len(a)
    for i in range(start_idx, n_ - window):
        c1 = a[i:i + window]
        c2 = b[i:i + window]
        if np.any(np.isnan(c1)) or np.any(np.isnan(c2)):
            continue
        if np.all(np.abs(c1) <= abs_thresh1) and np.all(np.abs(c2) <= abs_thresh2):
            return i
    return None


# =========================================================
# FRAME HELPERS
# =========================================================
def midpoint(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2


def compute_frame_helpers(lms_arr, n, conf_thresh):
    P = mp.solutions.pose.PoseLandmark

    def get_xy(idx, t):
        return get_xy_from_arr(lms_arr[t], int(idx), conf_thresh=conf_thresh)

    def shoulder_mid(t):
        L = get_xy(P.LEFT_SHOULDER, t)
        R = get_xy(P.RIGHT_SHOULDER, t)
        return midpoint(L, R)

    def pelvis_mid(t):
        L = get_xy(P.LEFT_HIP, t)
        R = get_xy(P.RIGHT_HIP, t)
        return midpoint(L, R)

    def ankle_mid(t):
        L = get_xy(P.LEFT_ANKLE, t)
        R = get_xy(P.RIGHT_ANKLE, t)
        return midpoint(L, R)

    def shoulder_line_angle(t):
        L = get_xy(P.LEFT_SHOULDER, t)
        R = get_xy(P.RIGHT_SHOULDER, t)
        if L is None or R is None:
            return None
        return line_angle_deg(L, R)

    def pelvis_line_angle(t):
        L = get_xy(P.LEFT_HIP, t)
        R = get_xy(P.RIGHT_HIP, t)
        if L is None or R is None:
            return None
        return line_angle_deg(L, R)

    return get_xy, shoulder_mid, pelvis_mid, ankle_mid, shoulder_line_angle, pelvis_line_angle


def compute_pelvis_coil_at_mkl_deg(MKL, shoulder_line_angle, pelvis_line_angle, pose_quality, n, min_q=0.35, half_window=2):
    vals = []
    for t in range(max(0, MKL - half_window), min(n - 1, MKL + half_window) + 1):
        if pose_quality[t] < min_q:
            continue
        sa = shoulder_line_angle(t)
        pa = pelvis_line_angle(t)
        if sa is None or pa is None:
            continue
        if np.isnan(sa) or np.isnan(pa):
            continue
        d = wrap180(float(sa) - float(pa))
        if d is None:
            continue
        vals.append(abs(float(d)))

    if not vals:
        return None
    return float(np.median(np.array(vals, dtype=np.float32)))


def compute_posture_at_mkl_deg(MKL, shoulder_mid, pelvis_mid, pose_quality, n, min_q=0.35, half_window=2):
    vals = []
    for t in range(max(0, MKL - half_window), min(n - 1, MKL + half_window) + 1):
        if pose_quality[t] < min_q:
            continue
        sm = shoulder_mid(t)
        pm = pelvis_mid(t)
        tilt = trunk_tilt_from_vertical_deg(pm, sm)
        if tilt is None or np.isnan(tilt):
            continue
        vals.append(float(tilt))
    if not vals:
        return None
    return float(np.median(np.array(vals, dtype=np.float32)))


def compute_head_behind_backhip_at_mkl_norm(
    MKL,
    get_xy,
    pose_quality,
    n,
    hand,
    plate_dir_sign,
    conf_thresh,
    min_q=0.35,
    half_window=2,
):
    P = mp.solutions.pose.PoseLandmark
    back_hip = P.RIGHT_HIP if hand == "R" else P.LEFT_HIP
    head_lm = P.NOSE

    vals = []

    for t in range(max(0, MKL - half_window), min(n - 1, MKL + half_window) + 1):
        if pose_quality[t] < min_q:
            continue

        head = get_xy(head_lm, t)
        bh = get_xy(back_hip, t)
        lh = get_xy(P.LEFT_HIP, t)
        rh = get_xy(P.RIGHT_HIP, t)

        if head is None or bh is None or lh is None or rh is None:
            continue

        hip_width = abs(float(lh[0]) - float(rh[0]))
        if not np.isfinite(hip_width) or hip_width < 1e-5:
            continue

        dx = (float(bh[0]) - float(head[0])) * float(plate_dir_sign)
        norm_dx = dx / hip_width
        if np.isfinite(norm_dx):
            vals.append(float(norm_dx))

    if not vals:
        return None

    return float(np.median(np.array(vals, dtype=np.float32)))


def compute_pelvis_drift_timing_pct(
    MKL,
    FFP,
    pelvis_mid,
    get_xy,
    pose_quality,
    n,
    plate_dir_sign,
    min_q=0.35,
    consec=3,
):
    if MKL is None or FFP is None:
        return None, None
    MKL = int(MKL)
    FFP = int(FFP)
    if FFP <= MKL + 3:
        return None, None

    px = [np.nan] * n
    for t in range(n):
        if pose_quality[t] < min_q:
            continue
        pm = pelvis_mid(t)
        if pm is None:
            continue
        px[t] = float(pm[0]) * float(plate_dir_sign)

    px_s = moving_median(forward_fill_nan(px), k=9)
    v = finite_diff(px_s)

    b0 = max(1, MKL - 5)
    b1 = min(n - 1, MKL + 5)
    base = v[b0:b1 + 1]
    base = base[np.isfinite(base)]
    base_std = float(np.std(base)) if base.size else 0.0

    thr = max(0.0008, 3.0 * base_std)

    run = 0
    onset = None
    for t in range(MKL + 1, min(FFP, n - 1) + 1):
        if not np.isfinite(v[t]):
            run = 0
            continue
        if float(v[t]) >= float(thr):
            run += 1
            if run >= int(consec):
                onset = int(t - (consec - 1))
                break
        else:
            run = 0

    if onset is None:
        return None, None

    pct = (float(onset - MKL) / float(max(1, FFP - MKL)))
    pct = max(0.0, min(1.0, pct))
    return int(onset), float(pct)


def compute_hip_shoulder_separation_at_ffp_deg(
    FFP,
    shoulder_line_angle,
    pelvis_line_angle,
    pose_quality,
    n,
    min_q=0.35,
    half_window=2,
):
    if FFP is None:
        return None

    vals = []
    FFP = int(FFP)

    for t in range(max(0, FFP - half_window), min(n - 1, FFP + half_window) + 1):
        if pose_quality[t] < min_q:
            continue

        sa = shoulder_line_angle(t)
        pa = pelvis_line_angle(t)
        if sa is None or pa is None:
            continue
        if np.isnan(sa) or np.isnan(pa):
            continue

        d = wrap180(float(sa) - float(pa))
        if d is None:
            continue

        vals.append(abs(float(d)))

    if not vals:
        return None

    return float(np.median(np.array(vals, dtype=np.float32)))


def compute_stride_length_at_ffp_norm(
    FFP,
    get_xy,
    shoulder_mid,
    ankle_mid,
    pose_quality,
    n,
    hand,
    plate_dir_sign,
    min_q=0.35,
    half_window=2,
):
    """
    Stride length proxy at foot strike:
      stride_fwd = (lead_ankle_x - back_ankle_x) * plate_dir_sign

    Normalize by a "height proxy" to handle zoom:
      height_proxy = || shoulder_mid - ankle_mid || at same frame

    We median over a small window around FFP to reduce one-bad-frame issues.
    """
    if FFP is None:
        return None, None, None  # stride_norm, stride_fwd_raw, height_proxy_raw

    P = mp.solutions.pose.PoseLandmark
    lead_ankle = P.LEFT_ANKLE if hand == "R" else P.RIGHT_ANKLE
    back_ankle = P.RIGHT_ANKLE if hand == "R" else P.LEFT_ANKLE

    vals_norm = []
    vals_stride = []
    vals_h = []

    FFP = int(FFP)

    for t in range(max(0, FFP - half_window), min(n - 1, FFP + half_window) + 1):
        if pose_quality[t] < min_q:
            continue

        la = get_xy(lead_ankle, t)
        ba = get_xy(back_ankle, t)
        sm = shoulder_mid(t)
        am = ankle_mid(t)

        if la is None or ba is None or sm is None or am is None:
            continue

        stride_fwd = (float(la[0]) - float(ba[0])) * float(plate_dir_sign)

        hproxy = float(norm(sm - am))
        if not np.isfinite(hproxy) or hproxy < 1e-6:
            continue

        stride_norm = stride_fwd / hproxy
        if np.isfinite(stride_norm):
            vals_norm.append(float(stride_norm))
            vals_stride.append(float(stride_fwd))
            vals_h.append(float(hproxy))

    if not vals_norm:
        return None, None, None

    return (
        float(np.median(np.array(vals_norm, dtype=np.float32))),
        float(np.median(np.array(vals_stride, dtype=np.float32))),
        float(np.median(np.array(vals_h, dtype=np.float32))),
    )


def compute_trunk_forward_tilt_at_ffp_deg(
    FFP,
    shoulder_mid,
    pelvis_mid,
    pose_quality,
    n,
    plate_dir_sign,
    min_q=0.35,
    half_window=2,
):
    """
    Trunk Tilt @ Foot Strike:
      signed forward lean of trunk from vertical, toward home plate.

    We median over a small window around FFP for robustness.
    """
    if FFP is None:
        return None

    vals = []
    FFP = int(FFP)

    for t in range(max(0, FFP - half_window), min(n - 1, FFP + half_window) + 1):
        if pose_quality[t] < min_q:
            continue
        sm = shoulder_mid(t)
        pm = pelvis_mid(t)
        ang = trunk_forward_tilt_signed_deg(pm, sm, plate_dir_sign)
        if ang is None or np.isnan(ang):
            continue
        vals.append(float(ang))

    if not vals:
        return None

    # Median of signed angles
    return float(np.median(np.array(vals, dtype=np.float32)))


# =========================================================
# TORSO-OPEN AUTO
# =========================================================
def detect_torso_open_idx(
    shoulder_angle_series,
    ffp_idx,
    opening_sign,
    deg_thresh=TORSO_OPEN_DEG,
    consec=TORSO_OPEN_CONSEC,
    max_off=TORSO_SEARCH_MAX_OFF,
):
    n = len(shoulder_angle_series)
    base = shoulder_angle_series[ffp_idx]
    if base is None or np.isnan(base):
        return None

    hi = min(n - 1, ffp_idx + int(max_off))
    run = 0
    for t in range(ffp_idx + 1, hi + 1):
        a = shoulder_angle_series[t]
        if a is None or np.isnan(a):
            run = 0
            continue
        delta = wrap180(float(a) - float(base))
        if delta is None:
            run = 0
            continue
        if float(delta) * float(opening_sign) >= float(deg_thresh):
            run += 1
            if run >= int(consec):
                return t - (consec - 1)
        else:
            run = 0
    return None


# =========================================================
# AUTO RELEASE (AI proxy): max forward wrist extension
# =========================================================
def suggest_release_by_wrist_extension(
    get_xy,
    shoulder_mid,
    hand,
    plate_dir_sign,
    ffp_idx,
    n,
    pose_quality,
    min_pose_q=0.35,
    start_off=REL_WIN_START_OFF,
    end_off=REL_WIN_END_OFF,
):
    P = mp.solutions.pose.PoseLandmark
    throw_wrist = P.RIGHT_WRIST if hand == "R" else P.LEFT_WRIST

    lo = max(0, int(ffp_idx) + int(start_off))
    hi = min(n - 1, int(ffp_idx) + int(end_off))
    best_t = None
    best_val = -1e18

    for t in range(lo, hi + 1):
        if pose_quality[t] < min_pose_q:
            continue
        w = get_xy(throw_wrist, t)
        sm = shoulder_mid(t)
        if w is None or sm is None:
            continue
        ext = (float(w[0]) - float(sm[0])) * float(plate_dir_sign)
        if ext > best_val:
            best_val = ext
            best_t = t

    return best_t


# =========================================================
# REPORT UI HELPERS
# =========================================================
def show_pose_frame(tmp_path, chosen_mode, lms_arr, t, caption, overlay=True, max_w=1000):
    fr = read_frame_at(tmp_path, int(t), chosen_mode, max_display_width=max_w)
    if fr is None:
        st.info("Frame not available.")
        return
    img = draw_pose_from_arr(fr, lms_arr[int(t)], fr.shape[1], fr.shape[0]) if overlay else fr
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption=caption)


def show_flipbook(tmp_path, chosen_mode, lms_arr, frames, key, title=None, overlay=True, max_w=1000):
    frames = [int(x) for x in frames if x is not None]
    frames = [x for x in frames if 0 <= x < len(lms_arr)]
    if not frames:
        st.info("No frames available.")
        return

    if title:
        st.caption(title)

    ss_key = f"flip_{key}_idx"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = 0
    st.session_state[ss_key] = int(np.clip(st.session_state[ss_key], 0, len(frames) - 1))

    b1, b2, b3 = st.columns([1, 1, 4])
    with b1:
        if st.button("◀", key=f"{key}_prev", use_container_width=True):
            st.session_state[ss_key] = max(0, st.session_state[ss_key] - 1)
    with b2:
        if st.button("▶", key=f"{key}_next", use_container_width=True):
            st.session_state[ss_key] = min(len(frames) - 1, st.session_state[ss_key] + 1)
    with b3:
        st.session_state[ss_key] = st.slider(
            "Frame",
            0,
            len(frames) - 1,
            int(st.session_state[ss_key]),
            key=f"{key}_slider",
        )

    t = frames[int(st.session_state[ss_key])]
    show_pose_frame(tmp_path, chosen_mode, lms_arr, t, caption=f"Frame {t}", overlay=overlay, max_w=max_w)


def render_metric_panel(
    metric_id: int,
    category_title: str,
    metric_title: str,
    score_value,
    explanation: str,
    observed: str,
    tips: str,
    right_mode: str,
    right_frames,
    tmp_path,
    chosen_mode,
    lms_arr,
    overlay=True,
    status: str = "NA",
    raw_value_str: str = None,
):
    st.markdown("---")
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown(f"### {metric_id}. {metric_title}")

        if score_value is None or (isinstance(score_value, float) and np.isnan(score_value)):
            st.metric("Score", "NA")
        else:
            st.metric("Score", f"{float(score_value):.1f}")

        dot = dot_status(status)
        if status and status != "NA":
            st.markdown(f"**Band:** {dot} **{status.title()}**")
        else:
            st.markdown(f"**Band:** {dot} **NA**")

        if raw_value_str:
            st.markdown(f"**Raw:** {raw_value_str}")

        st.markdown("**What we're measuring**")
        st.write(explanation)

        st.markdown("**What we observed in your clip**")
        st.write(observed)

        st.markdown("**How to improve**")
        st.write(tips)

    with right:
        if right_mode == "single":
            if right_frames is None:
                st.info("Frame not available.")
            else:
                show_pose_frame(
                    tmp_path, chosen_mode, lms_arr, int(right_frames),
                    caption=f"{metric_title} — key frame (Frame {int(right_frames)})",
                    overlay=overlay
                )
        elif right_mode == "flip":
            show_flipbook(
                tmp_path, chosen_mode, lms_arr, right_frames,
                key=f"m{metric_id}",
                title=f"{metric_title} — flipbook",
                overlay=overlay
            )
        else:
            st.info("No visual configured.")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Settings")

    orientation_mode = st.selectbox(
        "Video orientation",
        ["Auto (recommended)", "None", "90° CW", "180°", "270° CW"],
        index=0,
    )

    hand = st.selectbox("Pitcher handedness", ["R", "L"], index=0)

    plate_dir = st.selectbox(
        "Plate direction in video",
        ["Right (+x)", "Left (-x)"],
        index=0,
        help="For side view. Choose the direction the pitcher moves toward home plate.",
    )
    plate_dir_sign = 1.0 if "Right" in plate_dir else -1.0

    max_process_width = st.slider(
        "Processing width (px)",
        min_value=0,
        max_value=1280,
        value=DEFAULT_MAX_PROCESS_WIDTH,
        step=80,
        help="Downscale during processing to reduce CPU/RAM. Set 0 to disable.",
    )

    st.divider()
    st.subheader("Pose model (advanced)")
    model_complexity = st.selectbox("Model complexity", [0, 1, 2], index=2)
    smooth_landmarks = st.checkbox("Smooth landmarks", value=True)

    st.divider()
    st.subheader("Release window (for AI)")
    rel_start_off = st.number_input(
        "Start offset (frames after FFP)",
        value=int(REL_WIN_START_OFF),
        step=1,
        min_value=0,
        max_value=300,
    )
    rel_end_off = st.number_input(
        "End offset (frames after FFP)",
        value=int(REL_WIN_END_OFF),
        step=1,
        min_value=1,
        max_value=400,
    )

    st.divider()
    results_overlay = st.checkbox("Show pose overlay in Results visuals", value=True)


# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload pitching video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
if not uploaded:
    st.info("Upload a video to begin.")
    st.stop()

video_bytes = uploaded.getvalue()
video_hash = hashlib.md5(video_bytes).hexdigest()

tmp_path = None
try:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes)
    tmp_path = tmp.name
    tmp.close()

    pose_obj = get_pose_model(model_complexity=model_complexity, smooth_landmarks=smooth_landmarks)

    chosen_mode = orientation_mode
    if orientation_mode == "Auto (recommended)":
        with st.spinner("Auto-orienting..."):
            chosen_mode, _ = score_orientation(tmp_path, pose_obj, CONF_THRESH, sample_frames=SAMPLE_ORIENT_FRAMES)

    with st.spinner("Processing video..."):
        processed = process_video_landmarks_cached(
            video_hash=video_hash,
            video_path=tmp_path,
            chosen_mode=chosen_mode,
            conf_thresh=CONF_THRESH,
            max_process_width=int(max_process_width),
            model_complexity=int(model_complexity),
            smooth_landmarks=bool(smooth_landmarks),
        )

    if processed is None:
        st.error("Could not read video.")
        st.stop()

    lms_arr = processed["lms_arr"]
    pose_quality = processed["pose_quality"]
    n = processed["n"]
    fps = processed["fps"]
    duration_s = processed["duration_s"]
    pose_detected = processed["pose_detected"]

    if n < 20:
        st.error("Video too short to analyze reliably (need at least ~20 frames).")
        st.stop()

    pose_pct = 100.0 * (pose_detected / max(n, 1))
    clip_quality = float(np.mean(np.array(pose_quality, dtype=np.float32)))

    # =========================================================
    # AUTO EVENTS: MKL + FFP
    # =========================================================
    P = mp.solutions.pose.PoseLandmark
    lead_knee = P.LEFT_KNEE if hand == "R" else P.RIGHT_KNEE
    lead_ankle = P.LEFT_ANKLE if hand == "R" else P.RIGHT_ANKLE

    def get_y(idx, t):
        p = get_xy_from_arr(lms_arr[t], int(idx), conf_thresh=CONF_THRESH)
        return np.nan if p is None else float(p[1])

    knee_y = [get_y(lead_knee, t) for t in range(n)]
    ankle_y = [get_y(lead_ankle, t) for t in range(n)]

    knee_y_s = moving_median(forward_fill_nan(knee_y), k=9)
    ankle_y_s = moving_median(forward_fill_nan(ankle_y), k=9)

    search_end = int(0.70 * n)
    MKL_raw = int(np.nanargmin(knee_y_s[:search_end])) if not np.all(np.isnan(knee_y_s[:search_end])) else 0
    MKL_auto, _ = nearest_good_frame(MKL_raw, pose_quality, min_q=0.40, radius=8)

    knee_v = finite_diff(knee_y_s)
    ankle_v = finite_diff(ankle_y_s)

    start_search = min(n - 1, MKL_auto + 5)

    fps_eff = float(fps) if fps and fps > 1e-6 else 30.0
    win = max(6, int(round(0.08 * fps_eff)))

    base_fps = 30.0
    scale = base_fps / max(fps_eff, 1e-6)
    abs_thresh1 = 0.0035 * scale
    abs_thresh2 = 0.0045 * scale

    FFP_raw = find_first_stable_window2(
        ankle_v, knee_v, start_idx=start_search,
        abs_thresh1=abs_thresh1, abs_thresh2=abs_thresh2, window=win
    )
    if FFP_raw is None:
        seg = ankle_y_s[start_search:]
        FFP_raw = int(np.nanargmax(seg) + start_search) if not np.all(np.isnan(seg)) else min(n - 1, start_search + 10)
    FFP_auto, _ = nearest_good_frame(int(FFP_raw), pose_quality, min_q=0.40, radius=10)

    # =========================================================
    # SESSION STATE: events
    # =========================================================
    if "events" not in st.session_state:
        st.session_state["events"] = {}
    if "auto_events" not in st.session_state:
        st.session_state["auto_events"] = {}

    auto_events = st.session_state["auto_events"]
    auto_events["MKL"] = int(MKL_auto)
    auto_events["FFP"] = int(FFP_auto)

    events = st.session_state["events"]
    events.setdefault("MKL", int(MKL_auto))
    events.setdefault("FFP", int(FFP_auto))
    events.setdefault("RELEASE", None)
    events.setdefault("TORSO_OPEN", None)

    if st.session_state.get("_last_video_hash") != video_hash:
        st.session_state["_last_video_hash"] = video_hash
        events["MKL"] = int(MKL_auto)
        events["FFP"] = int(FFP_auto)
        events["RELEASE"] = None
        events["TORSO_OPEN"] = None

    # =========================================================
    # FRAME HELPERS
    # =========================================================
    get_xy, shoulder_mid, pelvis_mid, ankle_mid, shoulder_line_angle, pelvis_line_angle = compute_frame_helpers(lms_arr, n, CONF_THRESH)

    shoulder_ang_series = []
    for t in range(n):
        sa = shoulder_line_angle(t)
        shoulder_ang_series.append(np.nan if sa is None else float(sa))
    shoulder_ang_series = moving_median(shoulder_ang_series, k=7)

    opening_sign = float(plate_dir_sign)

    # =========================================================
    # AUTO TORSO OPEN (based on current FFP)
    # =========================================================
    torso_open_auto = detect_torso_open_idx(
        shoulder_angle_series=shoulder_ang_series,
        ffp_idx=events["FFP"],
        opening_sign=opening_sign,
        deg_thresh=TORSO_OPEN_DEG,
        consec=TORSO_OPEN_CONSEC,
        max_off=TORSO_SEARCH_MAX_OFF,
    )
    events["TORSO_OPEN"] = torso_open_auto

    # =========================================================
    # AUTO RELEASE (AI) — ALWAYS SET
    # =========================================================
    rel_lo = max(0, int(events["FFP"]) + int(rel_start_off))
    rel_hi = min(n - 1, int(events["FFP"]) + int(rel_end_off))
    if rel_hi <= rel_lo:
        rel_hi = min(n - 1, rel_lo + 10)

    release_auto_A = suggest_release_by_wrist_extension(
        get_xy=get_xy,
        shoulder_mid=shoulder_mid,
        hand=hand,
        plate_dir_sign=plate_dir_sign,
        ffp_idx=events["FFP"],
        n=n,
        pose_quality=pose_quality,
        min_pose_q=0.35,
        start_off=int(rel_start_off),
        end_off=int(rel_end_off),
    )

    if events["RELEASE"] is None:
        fallback = release_auto_A if release_auto_A is not None else rel_lo
        events["RELEASE"] = int(fallback)

    events["RELEASE"] = int(np.clip(int(events["RELEASE"]), 0, n - 1))

    # =========================================================
    # PRESENTATION FIRST: OVERALL SCORE AT TOP
    # =========================================================
    overall_score = None  # placeholder until wired

    st.markdown(
        f"""
        <div style="padding: 18px 18px 14px 18px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.15);">
          <div style="font-size: 18px; opacity: 0.85; margin-bottom: 6px;">Overall Score</div>
          <div style="font-size: 54px; font-weight: 900; line-height: 1.0;">{"NA" if overall_score is None else f"{overall_score:.1f}"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # =========================================================
    # COMPUTE KEY FRAMES + VISUAL WINDOWS
    # =========================================================
    MKL = int(events["MKL"])
    FFP = int(events["FFP"])
    REL = int(events["RELEASE"])

    mkl_frame = MKL
    ffp_frame = FFP
    rel_frame = REL

    frames_mkl_to_ffp = list(range(min(mkl_frame, ffp_frame), max(mkl_frame, ffp_frame) + 1))
    frames_preffp_to_rel = list(range(max(0, ffp_frame - 3), rel_frame + 1))
    frames_preffp_to_postrel = list(range(max(0, ffp_frame - 3), min(n - 1, rel_frame + 3) + 1))
    frames_full = list(range(0, n))

    # =========================================================
    # METRICS
    # =========================================================
    pelvis_coil_deg = compute_pelvis_coil_at_mkl_deg(
        MKL=MKL,
        shoulder_line_angle=shoulder_line_angle,
        pelvis_line_angle=pelvis_line_angle,
        pose_quality=pose_quality,
        n=n,
        min_q=0.35,
        half_window=2,
    )
    pelvis_coil_score, pelvis_coil_status = score_pelvis_coil_deg(pelvis_coil_deg)
    pelvis_coil_observed, pelvis_coil_tips = pelvis_coil_language(pelvis_coil_deg, pelvis_coil_status)

    posture_deg = compute_posture_at_mkl_deg(
        MKL=MKL,
        shoulder_mid=shoulder_mid,
        pelvis_mid=pelvis_mid,
        pose_quality=pose_quality,
        n=n,
        min_q=0.35,
        half_window=2,
    )
    posture_score, posture_status = score_posture_deg(posture_deg)
    posture_observed, posture_tips = posture_language(posture_deg, posture_status)

    head_behind_norm = compute_head_behind_backhip_at_mkl_norm(
        MKL=MKL,
        get_xy=get_xy,
        pose_quality=pose_quality,
        n=n,
        hand=hand,
        plate_dir_sign=plate_dir_sign,
        conf_thresh=CONF_THRESH,
        min_q=0.35,
        half_window=2,
    )
    head_score, head_status = score_head_behind_hip(head_behind_norm)
    head_observed, head_tips = head_behind_hip_language(head_behind_norm, head_status)

    drift_onset_frame, drift_pct = compute_pelvis_drift_timing_pct(
        MKL=MKL,
        FFP=FFP,
        pelvis_mid=pelvis_mid,
        get_xy=get_xy,
        pose_quality=pose_quality,
        n=n,
        plate_dir_sign=plate_dir_sign,
        min_q=0.35,
        consec=3,
    )
    drift_score, drift_status = score_pelvis_drift_timing(drift_pct)
    drift_observed, drift_tips = pelvis_drift_timing_language(drift_pct, drift_onset_frame, MKL, FFP)

    hip_sho_sep_deg = compute_hip_shoulder_separation_at_ffp_deg(
        FFP=FFP,
        shoulder_line_angle=shoulder_line_angle,
        pelvis_line_angle=pelvis_line_angle,
        pose_quality=pose_quality,
        n=n,
        min_q=0.35,
        half_window=2,
    )
    hip_sho_sep_score, hip_sho_sep_status = score_hip_shoulder_separation_deg(hip_sho_sep_deg)
    hip_sho_sep_observed, hip_sho_sep_tips = hip_shoulder_separation_language(hip_sho_sep_deg, hip_sho_sep_status)

    # --- Stride Length (FIXED / WIRED) ---
    stride_norm, stride_fwd_raw, stride_hproxy_raw = compute_stride_length_at_ffp_norm(
        FFP=FFP,
        get_xy=get_xy,
        shoulder_mid=shoulder_mid,
        ankle_mid=ankle_mid,
        pose_quality=pose_quality,
        n=n,
        hand=hand,
        plate_dir_sign=plate_dir_sign,
        min_q=0.35,
        half_window=2,
    )
    stride_score, stride_status = score_stride_length_norm(stride_norm)
    stride_observed, stride_tips = stride_length_language(stride_norm, stride_status)

    # --- Trunk Tilt @ Foot Strike (FIXED / WIRED) ---
    trunk_fwd_deg = compute_trunk_forward_tilt_at_ffp_deg(
        FFP=FFP,
        shoulder_mid=shoulder_mid,
        pelvis_mid=pelvis_mid,
        pose_quality=pose_quality,
        n=n,
        plate_dir_sign=plate_dir_sign,
        min_q=0.35,
        half_window=2,
    )
    trunk_score, trunk_status = score_trunk_tilt_ffp_deg(trunk_fwd_deg)
    trunk_observed, trunk_tips = trunk_tilt_language(trunk_fwd_deg, trunk_status)

    # =========================================================
    # REPORT
    # =========================================================
    st.subheader("Leg Lift → Foot Strike")
    render_metric_panel(
        1, "Leg Lift → Foot Strike", "Pelvis Coil",
        pelvis_coil_score,
        "Measures how well you “load” your hips at the top of leg lift. "
        "We estimate this by comparing the direction of your hips to the direction of your shoulders at max leg lift. "
        "If the hips are slightly more turned back than the shoulders, you’ve created a small stretch in the body. "
        "Why it matters: this stretch is like winding up a spring — it helps you stay closed longer, "
        "build separation later at foot strike, and often leads to better velocity and consistency without rushing the arm.",
        pelvis_coil_observed,
        pelvis_coil_tips,
        "single", mkl_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=pelvis_coil_status,
        raw_value_str=("NA" if pelvis_coil_deg is None else f"{pelvis_coil_deg:.1f}° separation @ MKL"),
    )

    render_metric_panel(
        2, "Leg Lift → Foot Strike", "Posture",
        posture_score,
        "Measures how ‘stacked’ your trunk is at the top of leg lift. "
        "We estimate this by looking at the line from your hips to your shoulders and checking how far it tilts away from vertical. "
        "Why it matters: being stacked helps balance and makes your timing repeatable, so you don’t have to rush to find the strike position.",
        posture_observed,
        posture_tips,
        "single", mkl_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=posture_status,
        raw_value_str=("NA" if posture_deg is None else f"{posture_deg:.1f}° trunk tilt from vertical @ MKL"),
    )

    render_metric_panel(
        3, "Leg Lift → Foot Strike", "Head Position Relative to Back Hip (head behind hip)",
        head_score,
        "Checks whether the head stays slightly behind the back hip at max leg lift. "
        "We measure the horizontal offset between the nose (head proxy) and the back hip, then normalize it by hip width. "
        "Why it matters: keeping the head behind the back hip supports balance and helps prevent early drift toward the plate.",
        head_observed,
        head_tips,
        "single", mkl_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=head_status,
        raw_value_str=("NA" if head_behind_norm is None else f"{head_behind_norm:.2f} (hip-widths) @ MKL"),
    )

    render_metric_panel(
        4, "Leg Lift → Foot Strike", "Pelvis Drift Timing",
        drift_score,
        "Measures **when your pelvis first clearly starts moving toward home plate** between max leg lift and foot strike. "
        "We track the pelvis midpoint’s horizontal position (toward the plate) and detect the first sustained forward move. "
        "Why it matters: if the pelvis drifts too early, you can ‘leak’ forward and lose control; too late, you often rush into foot strike and rush the arm.",
        drift_observed,
        drift_tips,
        "flip", frames_mkl_to_ffp, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=drift_status,
        raw_value_str=(
            "NA"
            if (drift_onset_frame is None or drift_pct is None)
            else f"Onset @ Frame {int(drift_onset_frame)}  |  {100.0*float(drift_pct):.0f}% of MKL→FFP window"
        ),
    )

    st.subheader("Foot Strike")
    render_metric_panel(
        5, "Foot Strike", "Hip-Shoulder Separation",
        hip_sho_sep_score,
        "Measures how much the hips have started opening while the shoulders are still ‘closed’ **at foot strike**. "
        "We estimate this by comparing the **pelvis line angle** (hips) to the **shoulder line angle** (torso) at (or near) foot strike, "
        "and taking the absolute difference in degrees.\n\n"
        "Why it matters: strong separation at foot strike is often a sign of good sequencing (lower body leads → torso follows), "
        "which helps create stretch, improves energy transfer, and can support velocity without rushing the arm.",
        hip_sho_sep_observed,
        hip_sho_sep_tips,
        "single", ffp_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=hip_sho_sep_status,
        raw_value_str=("NA" if hip_sho_sep_deg is None else f"{hip_sho_sep_deg:.1f}° separation @ FFP"),
    )

    render_metric_panel(
        6, "Foot Strike", "Stride Length",
        stride_score,
        "Measures how far you get down the mound **by foot strike**, scaled to your body so it works even if the camera zoom changes. "
        "We compute the forward distance from the back ankle to the lead ankle (toward home plate) at (or near) foot strike, "
        "then divide by a body-size proxy (shoulder-mid to ankle-mid distance).\n\n"
        "Why it matters: a good stride helps you build momentum and gives you more time/space to sequence (hips → torso → arm). "
        "Too short can limit momentum; too long often turns into reaching/landing soft and can mess with timing.",
        stride_observed,
        stride_tips,
        "single", ffp_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=stride_status,
        raw_value_str=("NA" if stride_norm is None else f"{stride_norm:.2f}× height (proxy) @ FFP"),
    )

    render_metric_panel(
        7, "Foot Strike", "Trunk Tilt",
        trunk_score,
        "Measures how much your trunk is leaning **toward home plate at foot strike** (forward lean), using a robust side-view proxy. "
        "We take the line from pelvis-mid to shoulder-mid and compute its angle relative to vertical. "
        "Positive values mean you’re leaning toward the plate; negative values mean you’re leaning back.\n\n"
        "Why it matters: at foot strike, you want athletic forward intent without ‘diving’. "
        "Too upright can reduce momentum/extension; too much forward tilt can collapse posture, soften the front side, and rush the arm.",
        trunk_observed,
        trunk_tips,
        "single", ffp_frame, tmp_path, chosen_mode, lms_arr, results_overlay,
        status=trunk_status,
        raw_value_str=("NA" if trunk_fwd_deg is None else f"{trunk_fwd_deg:+.1f}° forward trunk tilt @ FFP"),
    )

    render_metric_panel(
        8, "Foot Strike", "Separation Persistence (FS → Torso-Open)",
        None,
        "Measures whether separation is maintained from just before foot strike through release.",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_rel, tmp_path, chosen_mode, lms_arr, results_overlay
    )

    st.subheader("Foot Strike → Release (Where Velocity is Decided)")
    render_metric_panel(
        9, "Foot Strike → Release", "Center of Mass Velocity Toward Home Plate (proxy)",
        None,
        "Proxy estimate for how fast the body keeps moving toward home plate after foot strike.",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_postrel, tmp_path, chosen_mode, lms_arr, results_overlay
    )
    render_metric_panel(
        10, "Foot Strike → Release", "Lead Leg Block Effectiveness",
        None,
        "Measures whether the lead leg firms up so rotation happens over a stable front side.",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_postrel, tmp_path, chosen_mode, lms_arr, results_overlay
    )
    render_metric_panel(
        11, "Foot Strike → Release", "Torso Rotation Delay After Foot Strike",
        None,
        "Measures how long torso rotation is delayed after foot strike (uses AI release).",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_postrel, tmp_path, chosen_mode, lms_arr, results_overlay
    )
    render_metric_panel(
        12, "Foot Strike → Release", "Forward Trunk Tilt at Release",
        None,
        "Measures forward trunk tilt at release (uses AI release).",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_postrel, tmp_path, chosen_mode, lms_arr, results_overlay
    )
    render_metric_panel(
        13, "Foot Strike → Release", "Release Height and Extension",
        None,
        "Measures release height and how far out in front the throwing hand is at release (extension proxy).",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_preffp_to_postrel, tmp_path, chosen_mode, lms_arr, results_overlay
    )

    st.subheader("Overall")
    render_metric_panel(
        14, "Overall", "Sequencing",
        None,
        "Measures whether the motion unfolds efficiently (lower body → torso → arm).",
        "NA (metric not wired yet).",
        "NA (tips will be generated once metric is computed).",
        "flip", frames_full, tmp_path, chosen_mode, lms_arr, results_overlay
    )

    with st.expander("Optional: Key Frames + Scrubber + Clip Info", expanded=False):
        st.subheader("Key Frames")
        k1, k2, k3 = st.columns(3)
        with k1:
            show_pose_frame(tmp_path, chosen_mode, lms_arr, MKL, f"MKL (Frame {MKL})", overlay=True)
        with k2:
            show_pose_frame(tmp_path, chosen_mode, lms_arr, FFP, f"FFP (Frame {FFP})", overlay=True)
        with k3:
            show_pose_frame(tmp_path, chosen_mode, lms_arr, REL, f"Release (Frame {REL})", overlay=True)

        st.subheader("Frame Scrubber")
        t = st.slider("Select frame", 0, n - 1, int(FFP), 1, key="scrubber_slider")
        show_pose_frame(tmp_path, chosen_mode, lms_arr, t, f"Frame {t}", overlay=True)

        st.subheader("Clip Info")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames", f"{n}")
        c2.metric("FPS", f"{fps:.2f}" if fps > 0 else "NA")
        c3.metric("Duration", f"{duration_s:.2f}s" if duration_s > 0 else "NA")
        c4.metric("Pose detected", f"{pose_pct:.1f}%")

finally:
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
 
