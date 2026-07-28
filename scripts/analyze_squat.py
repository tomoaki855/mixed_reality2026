#!/usr/bin/env python3
"""
Single-camera squat analysis prototype using MediaPipe Pose Landmarker.

Outputs:
  1. Annotated MP4 video
  2. CSV file containing frame-by-frame measurements

Example:
  python scripts/analyze_squat.py \
      --input videos/squat_side.mp4 \
      --output outputs/squat_analysis.mp4 \
      --csv outputs/squat_analysis.csv

Notes:
- Film the whole body from the side.
- This is a small prototype for visualizing squat motion.
- It does NOT directly estimate lumbar flexion or thoracic extension.
  It uses 2D joint landmarks to estimate knee angle, hip angle,
  and whole-torso inclination.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

POSE_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
)

LEFT_SIDE = (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
RIGHT_SIDE = (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a side-view squat video with MediaPipe."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video")
    parser.add_argument("--output", type=Path, required=True, help="Annotated MP4")
    parser.add_argument("--csv", type=Path, required=True, help="Frame metrics CSV")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
        help="MediaPipe Pose Landmarker model",
    )
    parser.add_argument(
        "--side",
        choices=("auto", "left", "right"),
        default="auto",
        help="Body side used for angle calculation",
    )
    parser.add_argument("--start-angle", type=float, default=145.0, help="Knee angle threshold for starting a squat")
    parser.add_argument("--deep-angle", type=float, default=100.0, help="Knee angle threshold for deep squat")
    parser.add_argument("--down-angle", type=float, default=100.0)
    parser.add_argument("--up-angle", type=float, default=160.0)
    parser.add_argument("--max-torso-lean", type=float, default=50.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.30)
    return parser.parse_args()


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as error:
        raise RuntimeError(
            "Could not download the MediaPipe model automatically.\n"
            f"Download it manually from:\n{MODEL_URL}\n"
            f"and save it to:\n{model_path}"
        ) from error


def point(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y], dtype=np.float64)

def squat_depth_ratio(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
) -> float:
    lower_leg_length = np.linalg.norm(
        knee - ankle
    )

    if lower_leg_length < 1e-8:
        return float("nan")

    return float(
        (hip[1] - knee[1])
        / lower_leg_length
    )


def upper_back_flexion_angle(
    ear: np.ndarray,
    shoulder: np.ndarray,
    hip: np.ndarray,
) -> float:
    return angle_degrees(
        ear,
        shoulder,
        hip,
    )


def foot_balance_ratio(
    shoulder: np.ndarray,
    hip: np.ndarray,
    heel: np.ndarray,
    toe: np.ndarray,
) -> float:
    foot_length_x = toe[0] - heel[0]

    if abs(foot_length_x) < 1e-8:
        return float("nan")

    body_center_x = (
        shoulder[0] + hip[0]
    ) / 2.0

    return float(
        (body_center_x - heel[0])
        / foot_length_x
    )


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def torso_inclination_degrees(shoulder: np.ndarray, hip: np.ndarray) -> float:
    torso = shoulder - hip
    vertical_up = np.array([0.0, -1.0], dtype=np.float64)
    denominator = np.linalg.norm(torso)
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.dot(torso, vertical_up) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def side_visibility(landmarks, indices: Sequence[int]) -> float:
    return float(np.mean([landmarks[index].visibility for index in indices]))


def select_side(landmarks, requested: str) -> str:
    if requested in {"left", "right"}:
        return requested
    return (
        "left"
        if side_visibility(landmarks, LEFT_SIDE)
        >= side_visibility(landmarks, RIGHT_SIDE)
        else "right"
    )


def side_indices(side: str) -> tuple[int, int, int, int, int, int, int]:
    if side == "left":
        return LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_EAR,LEFT_HEEL, LEFT_FOOT_INDEX
    return RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_HEEL, RIGHT_FOOT_INDEX


def ema(previous: Optional[float], current: float, alpha: float) -> float:
    if not np.isfinite(current):
        return previous if previous is not None else float("nan")
    if previous is None or not np.isfinite(previous):
        return current
    return alpha * current + (1.0 - alpha) * previous


def to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x, 0.0, 1.0) * width)
    y = int(np.clip(landmark.y, 0.0, 1.0) * height)
    return x, y


def draw_pose(frame: np.ndarray, landmarks, min_visibility: float) -> None:
    height, width = frame.shape[:2]
    for start, end in POSE_CONNECTIONS:
        if (
            landmarks[start].visibility < min_visibility
            or landmarks[end].visibility < min_visibility
        ):
            continue
        cv2.line(
            frame,
            to_pixel(landmarks[start], width, height),
            to_pixel(landmarks[end], width, height),
            (70, 220, 70),
            3,
            cv2.LINE_AA,
        )
    for landmark in landmarks:
        if landmark.visibility < min_visibility:
            continue
        cv2.circle(
            frame,
            to_pixel(landmark, width, height),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_panel(
    frame: np.ndarray,
    metrics: dict,
    warnings: Iterable[str],
) -> None:
    warning_list = list(warnings)

    lines = [
        f"Repetitions: {metrics['repetitions']}",
        f"Phase: {metrics['phase']}",
        f"Side: {metrics['side']}",
        f"Knee angle: {metrics['knee_angle']:.1f} deg",
        f"Hip angle: {metrics['hip_angle']:.1f} deg",
        f"Torso lean: {metrics['torso_lean']:.1f} deg",
        f"Upper-back angle: {metrics['upper_back_angle']:.1f} deg",
        f"Upper-back drop: {metrics['upper_back_drop']:.1f} deg",
        f"Squat depth: {metrics['squat_depth']:.2f}",
        f"Balance ratio: {metrics['balance_ratio']:.2f}",
    ]

    line_height = 28
    panel_height = (
        20
        + line_height * len(lines)
        + (38 if warning_list else 10)
    )
    panel_width = min(560, frame.shape[1])

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    y = 28

    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_height

    if warning_list:
        cv2.putText(
            frame,
            " / ".join(warning_list),
            (15, panel_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1].")

    ensure_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoothed_knee: Optional[float] = None
    smoothed_hip: Optional[float] = None
    smoothed_torso: Optional[float] = None
    smoothed_upper_back: Optional[float] = None
    smoothed_depth_ratio: Optional[float] = None
    neutral_upper_back_samples = []
    neutral_upper_back_angle: Optional[float] = None
    phase = "UP"
    repetitions = 0
    frame_index = 0

    csv_fields = [
        "frame",
        "time_sec",
        "detected",
        "side",
        "knee_angle_deg",
        "hip_angle_deg",
        "torso_lean_deg",
        "upper_back_angle_deg",
        "upper_back_drop_deg",
        "squat_depth_ratio",
        "balance_ratio",
        "phase",
        "repetitions",
        "warnings",
    ]

    with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                detected = bool(result.pose_landmarks)
                warning_messages: list[str] = []
                selected_side = ""
                knee_angle = float("nan")
                hip_angle = float("nan")
                torso_lean = float("nan")
                upper_back_angle = float("nan")
                upper_back_drop = float("nan")
                depth_ratio = float("nan")
                balance_ratio = float("nan")

                if detected:
                    landmarks = result.pose_landmarks[0]
                    selected_side = select_side(landmarks, args.side)
                    shoulder_idx, hip_idx, knee_idx, ankle_idx, ear_idx, heel_idx, toe_idx = side_indices(selected_side)
                    relevant = (shoulder_idx, hip_idx, knee_idx, ankle_idx, ear_idx, heel_idx, toe_idx)
                    minimum_visibility = min(
                        landmarks[index].visibility for index in relevant
                    )
                    draw_pose(frame, landmarks, args.min_visibility)

                    if minimum_visibility >= args.min_visibility:
                        shoulder = point(landmarks[shoulder_idx])
                        hip = point(landmarks[hip_idx])
                        knee = point(landmarks[knee_idx])
                        ankle = point(landmarks[ankle_idx])
                        ear = point(landmarks[ear_idx])
                        heel = point(landmarks[heel_idx])
                        toe = point(landmarks[toe_idx])

                        smoothed_knee = ema(
                            smoothed_knee,
                            angle_degrees(hip, knee, ankle),
                            args.ema_alpha,
                        )
                        smoothed_hip = ema(
                            smoothed_hip,
                            angle_degrees(shoulder, hip, knee),
                            args.ema_alpha,
                        )
                        smoothed_torso = ema(
                            smoothed_torso,
                            torso_inclination_degrees(shoulder, hip),
                            args.ema_alpha,
                        )
                        smoothed_upper_back = ema(
                            smoothed_upper_back,
                            upper_back_flexion_angle(ear, shoulder, hip),
                            args.ema_alpha,
                        )
                        smoothed_depth_ratio = ema(
                            smoothed_depth_ratio,
                            squat_depth_ratio(hip, knee, ankle),
                            args.ema_alpha,
                        )
                        balance_ratio = foot_balance_ratio(shoulder, hip, heel, toe)
                        
                        knee_angle = float(smoothed_knee)
                        hip_angle = float(smoothed_hip)
                        torso_lean = float(smoothed_torso)
                        depth_ratio = float(smoothed_depth_ratio)
                        upper_back_angle = float(smoothed_upper_back)
                        if (neutral_upper_back_angle is None and np.isfinite(upper_back_angle)):
                            neutral_upper_back_samples.append(upper_back_angle)
                            if len(neutral_upper_back_samples) >= 10:
                                neutral_upper_back_angle = float(np.median(neutral_upper_back_samples))
                        if (neutral_upper_back_angle is not None and np.isfinite(upper_back_angle)):
                            upper_back_drop = (neutral_upper_back_angle - upper_back_angle)
                        if (neutral_upper_back_angle is not None and np.isfinite(upper_back_angle)):
                            upper_back_drop = neutral_upper_back_angle - upper_back_angle
                        if phase == "UP" and knee_angle < args.start_angle:
                            phase = "DOWN"
                        elif phase == "DOWN" and knee_angle > args.up_angle:
                            phase = "UP"
                            repetitions += 1
                        if (
                            phase == "DOWN"
                            and knee_angle < args.deep_angle
                            and depth_ratio < 0.05
                        ):
                            warning_messages.append("Squat depth is too shallow")
                        if (np.isfinite(upper_back_drop) and upper_back_drop > 7.0):
                            warning_messages.append("Upper back rounding detected")
                        if torso_lean > args.max_torso_lean:
                            warning_messages.append("Excessive torso lean")
                        if np.isfinite(balance_ratio):
                            if(balance_ratio < 0.25):
                                warning_messages.append("Weight may be shifted to heals")
                            elif(balance_ratio > 0.75):
                                warning_messages.append("Weight may be shifted to toes")
                    else:
                        warning_messages.append("Low landmark visibility")
                else:
                    warning_messages.append("Pose not detected")

                metrics = {
                    "repetitions": repetitions,
                    "phase": phase,
                    "side": selected_side or "-",
                    "knee_angle": knee_angle,
                    "hip_angle": hip_angle,
                    "torso_lean": torso_lean,
                    "squat_depth": depth_ratio,
                    "upper_back_angle": upper_back_angle,
                    "upper_back_drop": upper_back_drop,
                    "balance_ratio": balance_ratio,
                }
                draw_panel(frame, metrics, warning_messages)
                writer.write(frame)

                csv_writer.writerow(
                    {
                        "frame": frame_index,
                        "time_sec": f"{frame_index / fps:.4f}",
                        "detected": int(detected),
                        "side": selected_side,
                        "knee_angle_deg": (
                            f"{knee_angle:.4f}" if np.isfinite(knee_angle) else ""
                        ),
                        "hip_angle_deg": (
                            f"{hip_angle:.4f}" if np.isfinite(hip_angle) else ""
                        ),
                        "torso_lean_deg": (
                            f"{torso_lean:.4f}" if np.isfinite(torso_lean) else ""
                        ),
                        "squat_depth_ratio": (
                            f"{depth_ratio:.4f}" if np.isfinite(depth_ratio) else ""
                        ),
                        "upper_back_drop_deg": (
                            f"{upper_back_drop:.4f}" if np.isfinite(upper_back_drop) else ""
                        ),
                        "upper_back_angle_deg": (
                            f"{upper_back_angle:.4f}" if np.isfinite(upper_back_angle) else ""
                        ),
                        "balance_ratio": (
                            f"{balance_ratio:.4f}" if np.isfinite(balance_ratio) else ""
                        ),
                        "phase": phase,
                        "repetitions": repetitions,
                        "warnings": ";".join(warning_messages),
                    }
                )
                frame_index += 1

    capture.release()
    writer.release()

    print("Finished.")
    print(f"Annotated video: {args.output}")
    print(f"Metrics CSV:     {args.csv}")
    print(f"Repetitions:     {repetitions}")


if __name__ == "__main__":
    main()
