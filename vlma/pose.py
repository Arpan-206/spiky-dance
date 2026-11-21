"""Extract tracked MediaPipe poses and render an episode skeleton video."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


COCO_INDICES = np.array([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16), (0, 1), (0, 2), (1, 3), (2, 4))
COLORS = ((80, 230, 100), (255, 160, 60), (200, 100, 255))
MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "pose_landmarker_full.task"


def _track(detections, count):
    """Anchor left-to-right identities at the first complete observation."""
    n_frames = len(detections)
    image = np.zeros((n_frames, count, 33, 3), dtype=np.float32)
    world = np.full_like(image, np.nan)
    vis = np.zeros((n_frames, count, 33), dtype=np.float32)
    seen = np.zeros((n_frames, count), dtype=bool)
    anchor = next((i for i, poses in enumerate(detections) if len(poses) >= count), None)
    if anchor is None:
        anchor = max(range(n_frames), key=lambda i: len(detections[i]))
    initial = sorted(detections[anchor], key=lambda p: p[0][[23, 24], 0].mean())[:count]
    centers = np.full((count, 2), np.nan)
    for dancer, pose in enumerate(initial):
        centers[dancer] = pose[0][[23, 24], :2].mean(axis=0)

    def save(frame, dancer, pose):
        image[frame, dancer], world[frame, dancer], vis[frame, dancer] = pose
        seen[frame, dancer] = True

    for dancer, pose in enumerate(initial):
        save(anchor, dancer, pose)
    for frames in (range(anchor + 1, n_frames), range(anchor - 1, -1, -1)):
        last = centers.copy()
        for frame in frames:
            poses = detections[frame]
            if not poses:
                continue
            current = np.array([p[0][[23, 24], :2].mean(axis=0) for p in poses])
            known = np.flatnonzero(np.isfinite(last).all(axis=1))
            assigned = set()
            if len(known):
                rows, cols = linear_sum_assignment(np.linalg.norm(last[known, None] - current[None], axis=2))
                for row, col in zip(rows, cols):
                    dancer = known[row]
                    save(frame, dancer, poses[col])
                    last[dancer] = current[col]
                    assigned.add(int(col))
            free = np.flatnonzero(~np.isfinite(last).all(axis=1))
            remaining = sorted((j for j in range(len(poses)) if j not in assigned), key=lambda j: current[j, 0])
            for dancer, col in zip(free, remaining):
                save(frame, dancer, poses[col])
                last[dancer] = current[col]
    return image, world, vis, seen


def _writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open {path}")
    return writer


def run_pose(clip_path, out_dir, dancers=None, max_frames=None):
    """Write an episode and return detection, confidence and timing statistics.

    Automatic counts require at least two frames and ten percent of observations.
    Empty clips with frames use one missing dancer. max_frames trims clip.mp4 too.
    """
    started = time.perf_counter()
    if dancers is not None and not 1 <= dancers <= 3:
        raise ValueError("dancers must be between 1 and 3")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    source, output = Path(clip_path).resolve(), Path(out_dir).resolve()
    if source == output / "clip.mp4":
        raise ValueError("Source must be outside the output clip.mp4 path")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture failed to open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video fps: {fps}")

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_poses=3,
    )
    detections, inference_seconds = [], 0.0
    try:
        with vision.PoseLandmarker.create_from_options(options) as model:
            timestamp = -1
            while max_frames is None or len(detections) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = max(timestamp + 1, round(len(detections) * 1000 / fps))
                tick = time.perf_counter()
                result = model.detect_for_video(rgb, timestamp)
                inference_seconds += time.perf_counter() - tick
                poses = []
                for points, points3d in zip(result.pose_landmarks, result.pose_world_landmarks):
                    image = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float32)
                    world = np.array([(p.x, p.y, p.z) for p in points3d], dtype=np.float32)
                    vis = np.array([p.visibility if p.visibility is not None else 0.0 for p in points], dtype=np.float32)
                    poses.append((image, world, vis))
                detections.append(poses)
    finally:
        cap.release()
    if not detections:
        raise ValueError(f"No readable frames in {source}")
    count = dancers
    if count is None:
        threshold = min(len(detections), max(2, int(np.ceil(len(detections) * 0.1))))
        count = max((n for n in range(1, 4) if sum(len(p) >= n for p in detections) >= threshold), default=1)
    image, world, vis, seen = _track(detections, count)
    poses = np.zeros((len(detections), count, 17, 3), dtype=np.float32)
    poses[..., :2] = image[:, :, COCO_INDICES, :2] * np.array([width, height], dtype=np.float32)
    poses[..., 2] = vis[:, :, COCO_INDICES]
    poses[poses[..., 2] == 0] = 0
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "poses.npy", poses)
    np.savez_compressed(output / "pose3d.npz", world=world, vis=vis, image=image, fps=fps)
    metadata = dict(fps=fps, width=width, height=height, n_frames=len(detections), dancers=count, clip_id=source.stem)
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cap = cv2.VideoCapture(str(source))
    skeleton, trimmed = None, None
    try:
        skeleton = _writer(output / "skeleton.mp4", fps, (width, height))
        if max_frames is not None:
            trimmed = _writer(output / "clip.mp4", fps, (width, height))
        for frame_index in range(len(detections)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not reread source frame {frame_index}")
            if trimmed is not None:
                trimmed.write(frame)
            frame = (frame.astype(np.float32) * 0.35).astype(np.uint8)
            for dancer, points in enumerate(poses[frame_index]):
                color = COLORS[dancer]
                for a, b in EDGES:
                    if points[a, 2] > 0.3 and points[b, 2] > 0.3:
                        cv2.line(frame, tuple(np.rint(points[a, :2]).astype(int)), tuple(np.rint(points[b, :2]).astype(int)), color, 2)
                for x, y, confidence in points:
                    if confidence > 0.3:
                        cv2.circle(frame, (round(float(x)), round(float(y))), 3, color, -1)
            skeleton.write(frame)
    finally:
        cap.release()
        if skeleton is not None:
            skeleton.release()
        if trimmed is not None:
            trimmed.release()
    if max_frames is None:
        shutil.copy2(source, output / "clip.mp4")
    seconds = time.perf_counter() - started
    summary = dict(metadata, detected_fraction=seen.mean(axis=0).tolist(),
                   mean_confidence=poses[..., 2].mean(axis=(0, 2)).tolist(),
                   seconds=seconds, inference_seconds=inference_seconds,
                   model_fps=len(detections) / inference_seconds if inference_seconds else 0.0,
                   processing_fps=len(detections) / seconds, out_dir=str(output))
    print(f"frames={len(detections)} dancers={count} detected_fraction={summary['detected_fraction']} "
          f"mean_confidence={summary['mean_confidence']} seconds={seconds:.3f} "
          f"model_fps={summary['model_fps']:.2f} processing_fps={summary['processing_fps']:.2f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dancers", type=int, choices=range(1, 4))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    run_pose(args.clip_path, args.out, args.dancers, args.max_frames)


if __name__ == "__main__":
    main()
