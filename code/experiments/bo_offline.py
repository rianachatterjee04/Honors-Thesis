#!/usr/bin/env python3
"""
GenAssist — Phase 2 Offline Shader Bayesian Optimization (LIVE PLAYBACK)
=========================================================================
Fixes in this version:
  - VLMWorker._worker now fires on_done BEFORE setting busy=False,
    so the freeze loop actually runs while VLM is processing.
  - Freeze elapsed timer now uses a captured freeze_start, not t=time.time()
    reset each iteration (which always showed 0s).
  - goto_end flag properly breaks out of the video loop on Q press.

Usage:
    # BO picks threshold automatically:
    python3 bo_offline.py --condition amd --video results/VR/AMD/trial_x_0.1_raw.mp4

    # Manual threshold:
    python3 bo_offline.py --condition amd --video results/VR/AMD/trial_x_0.1_raw.mp4 --threshold 0.35

    # No live window (headless):
    python3 bo_offline.py --condition amd --video results/VR/AMD/trial_x_0.1_raw.mp4 --no_display

    # Custom BO history path (for offline fresh runs):
    python3 bo_offline.py --condition amd --video results/VR/AMD/trial_x_0.1_raw.mp4 --bo_db results/VR/AMD_Offline/bo_history_offline.csv
"""

import os
import cv2
import csv
import time
import base64
import argparse
import threading
import textwrap
import numpy as np
import pandas as pd
import requests

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from ultralytics import YOLO

os.environ['LD_PRELOAD'] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

VLM_ENDPOINT = "http://localhost:5000/vlm"
SOURCE_VIDEOS = []

# ── Video discovery ────────────────────────────────────────────────────────────

def discover_videos(results_dir: str, thresh_min: float, thresh_max: float) -> list:
    import glob, re
    found = []
    pattern = os.path.join(results_dir, "trial_x_*.mp4")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        m = re.match(r"trial_x_([0-9.]+)\.mp4$", fname)
        if not m:
            continue
        try:
            tval = float(m.group(1))
        except ValueError:
            continue
        if thresh_min <= tval <= thresh_max:
            env_label = os.path.basename(results_dir).replace("_results", "")
            found.append({"path": fpath, "env": env_label, "threshold": tval})
    found.sort(key=lambda x: x["threshold"])
    return found

# ── Shader filters ─────────────────────────────────────────────────────────────

def apply_amd(frame: np.ndarray) -> np.ndarray:
    """Dark central scotoma + metamorphopsia. Periphery intact."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    out = frame.astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (w // 5, h // 5), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (81, 81), 40)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    wave_x = map_x + mask * 9 * np.sin(map_y / 10.0)
    wave_y = map_y + mask * 9 * np.sin(map_x / 10.0)
    distorted = cv2.remap(out, wave_x, wave_y, cv2.INTER_LINEAR)
    darkened = distorted * 0.08
    mask3 = np.stack([mask] * 3, axis=-1)
    result = out * (1 - mask3) + darkened * mask3
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_glaucoma(frame: np.ndarray) -> np.ndarray:
    """Peripheral blackout, central oval preserved."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (w // 3, h // 3), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 50)
    mask3 = np.stack([mask] * 3, axis=-1)
    return (frame * mask3).astype(np.uint8)

def apply_rp(frame: np.ndarray) -> np.ndarray:
    """Severe tunnel vision with white peripheral loss."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (w // 3, h // 3), 0, 0, 360, 1, -1)  # was w//5, h//5
    mask = cv2.GaussianBlur(mask, (71, 71), 35)
    mask3 = np.stack([mask] * 3, axis=-1)
    white = np.full_like(frame, 255, dtype=np.float32)
    result = frame.astype(np.float32) * mask3 + white * (1 - mask3)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_dr(frame: np.ndarray, seed: int = None) -> np.ndarray:
    """Diabetic Retinopathy — heavy dark grain/noise with overall dimming."""
    rng = np.random.RandomState(seed if seed is not None else 42)
    # Heavy dark noise layer
    noise = rng.randint(0, 80, frame.shape, dtype=np.uint8)
    # Darken the frame overall
    darkened = (frame.astype(np.float32) * 0.85).astype(np.uint8)
    # Blend noise on top — noise darkens, not lightens
    result = np.where(noise < 20, np.zeros_like(darkened), darkened)
    return result.astype(np.uint8)

SHADER_MAP = {
    "amd":      apply_amd,
    "glaucoma": apply_glaucoma,
    "rp":       apply_rp,
    "dr":       apply_dr,
}
CONDITION_LABELS = {
    "amd":      "AMD (Central Scotoma)",
    "glaucoma": "Glaucoma (Peripheral Blackout)",
    "rp":       "Retinitis Pigmentosa (Extreme Tunnel)",
    "dr":       "Diabetic Retinopathy (Random Scotomas)",
}

# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, threshold: float, max_conf: float,
             status: str, vlm_text: str, condition: str,
             vlm_calls: int, total_steps: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Top HUD bar
    bar_h = 38
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    cv2.putText(out, f"[{CONDITION_LABELS[condition]}]",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    thresh_color = (0, 255, 0) if max_conf >= threshold else (0, 80, 255)
    cv2.putText(out, f"thresh={threshold:.3f}  conf={max_conf:.3f}",
                (w - 280, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, thresh_color, 1, cv2.LINE_AA)

    usage = vlm_calls / max(total_steps, 1) * 100
    cv2.putText(out, f"VLM: {vlm_calls} calls ({usage:.1f}%)",
                (w // 2 - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Status flash bar
    if status == "VLM_TRIGGERED":
        cv2.rectangle(out, (0, bar_h), (w, bar_h + 22), (0, 60, 180), -1)
        cv2.putText(out, "  VLM TRIGGERED — analyzing frame...",
                    (8, bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

    # VLM response panel at bottom
    if vlm_text and vlm_text not in ("N/A", "PENDING...", "VLM_ERROR", ""):
        lines = textwrap.wrap(vlm_text, width=80)
        panel_h = len(lines) * 20 + 14
        panel_y = h - panel_h - 4
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, panel_y), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.75, out, 0.25, 0, out)
        cv2.putText(out, "VLM:", (8, panel_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 200, 80), 1, cv2.LINE_AA)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (52, panel_y + 16 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    return out

# ── BO utilities ───────────────────────────────────────────────────────────────

def get_next_threshold(db_path: str) -> float:
    initial_points = [0.1, 0.5, 0.9]

    if not os.path.exists(db_path):
        return initial_points[0]
    try:
        df = pd.read_csv(db_path)
    except Exception:
        return initial_points[0]

    if len(df) < len(initial_points):
        return initial_points[len(df)]

    try:
        df = df[df['latency'] > 0] if 'latency' in df.columns else df
        if len(df) < 2:
            import random
            return round(random.uniform(0.2, 0.6), 2)
        df2 = df[['threshold', 'reward']].dropna()
        df2 = df2.groupby('threshold', as_index=False)['reward'].mean()
        train_x = torch.tensor(df2['threshold'].values, dtype=torch.double).unsqueeze(-1).clamp(0.0, 1.0)
        train_y = torch.tensor(df2['reward'].values, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(train_x, train_y,
                          input_transform=Normalize(d=1),
                          outcome_transform=Standardize(m=1))
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-3))

        with gpytorch.settings.cholesky_jitter(1e-2):
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            gp.likelihood.eval()
            UCB = UpperConfidenceBound(gp, beta=1.0)
            try:
                new_x, _ = optimize_acqf(
                    UCB,
                    bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
                    q=1, num_restarts=5, raw_samples=128,
                    options={"maxiter": 200},
                )
                return float(np.clip(new_x.item(), 0.0, 1.0))
            except Exception as e:
                print(f"[BO] optimize_acqf failed: {e} — grid fallback")
                with torch.no_grad():
                    grid = torch.linspace(0.0, 1.0, 201, dtype=torch.double).view(-1, 1, 1)
                    vals = UCB(grid)
                    vals = torch.nan_to_num(vals, neginf=-1e9, posinf=1e9)
                    return float(grid[torch.argmax(vals)].item())
    except Exception as e:
        print(f"[BO] GP failed: {e} — random fallback")
        import random
        return round(random.uniform(0.1, 0.9), 2)


def save_trial_result(threshold, latency, usage, user_score, db_path):
    clamped_latency_penalty = min(latency / 1000.0, 1.0)
    tech_penalty = -clamped_latency_penalty - (usage * 5.0)
    human_reward = float(user_score) / 2.0
    reward = tech_penalty + human_reward
    new_data = pd.DataFrame(
        [[threshold, latency, usage, float(user_score), reward]],
        columns=['threshold', 'latency', 'usage', 'user_score', 'reward']
    )
    if not os.path.exists(db_path):
        new_data.to_csv(db_path, index=False, header=True)
    else:
        new_data.to_csv(db_path, mode='a', header=False, index=False)
    print(f"\n[BO] Saved → threshold={threshold:.4f}  reward={reward:.4f}")

# ── VLM worker ────────────────────────────────────────────────────────────────

class VLMWorker:
    def __init__(self):
        self.busy = False
        self.last_description = ""
        self.last_latency_ms = 0.0
        self.total_calls = 0
        self._lock = threading.Lock()

    def call(self, frame: np.ndarray, on_done=None):
        with self._lock:
            if self.busy:
                return
            self.busy = True
            self.total_calls += 1
        threading.Thread(target=self._worker, args=(frame, on_done), daemon=True).start()

    def _worker(self, frame, on_done):
        start = time.time()
        try:
            small = cv2.resize(frame, (320, 240))
            _, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_b64 = base64.b64encode(buf).decode('utf-8')
            payload = {
                "prompt": (
                    "Identify obstacles and hazards in the room. "
                    "Note: The noise is a camera defect simulating Diabetic Retinopathy-ignore it completely. "
                    "Focus only on the clear areas of the image to describe the actual floor and path."
                    "Answer in a sentence or two."
                ),
                "image": img_b64,
            }
            resp = requests.post(VLM_ENDPOINT, json=payload, timeout=30.0)
            if resp.status_code == 200:
                self.last_description = resp.json().get("response", "").strip().replace(",", ";")
                self.last_latency_ms = (time.time() - start) * 1000
        except Exception as e:
            print(f"[VLM] Error: {e}")
            self.last_description = "VLM_ERROR"
            self.last_latency_ms = 0.0
        finally:
            # FIX: fire on_done BEFORE releasing busy so the freeze loop
            # in run_trial stays active for the full VLM duration.
            if on_done:
                on_done(self.last_description, self.last_latency_ms)
            with self._lock:
                self.busy = False

# ── Main trial runner ──────────────────────────────────────────────────────────

def run_trial(condition: str, threshold: float, output_dir: str,
              trial_seed: int, show_display: bool = True):
    shader_fn = SHADER_MAP[condition]
    yolo = YOLO('yolov8n.pt')

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"bo_trial_{condition}_x_{threshold:.4f}_offline.csv")
    vid_path = os.path.join(output_dir, f"trial_{condition}_x_{threshold:.4f}_offline.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, 15.0, (640, 480))

    csvf = open(csv_path, 'w', newline='')
    writer_csv = csv.writer(csvf)
    writer_csv.writerow(["frame_idx", "source_env", "source_file",
                         "yolo_max_conf", "detected_classes", "status",
                         "vlm_response", "latency_ms", "usage_ratio"])

    vlm = VLMWorker()
    latencies = []
    total_steps = 0
    current_vlm_text = ""
    goto_end = False

    if show_display:
        win_name = f"GenAssist Phase 2 — {CONDITION_LABELS[condition]}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)

    def on_vlm_done(desc, lat, frame_idx, env, src):
        nonlocal current_vlm_text
        latencies.append(lat)
        current_vlm_text = desc
        usage = vlm.total_calls / max(total_steps, 1)
        writer_csv.writerow([frame_idx, env, src, "N/A",
                              "VLM_RESPONSE_RECEIVED", desc,
                              f"{lat:.1f}", f"{usage:.4f}"])
        csvf.flush()
        print(f"\n  [VLM @ frame {frame_idx}] ({lat:.0f}ms): {desc[:120]}")

    for video_info in SOURCE_VIDEOS:
        if goto_end:
            break

        vid_file = video_info["path"]
        env_label = video_info["env"]

        if not os.path.isabs(vid_file):
            candidates = [
                vid_file,
                os.path.join(os.path.dirname(__file__), vid_file),
                os.path.join(os.path.dirname(__file__), "data", vid_file),
            ]
            resolved = next((p for p in candidates if os.path.exists(p)), None)
            if resolved is None:
                print(f"[WARN] Video not found: {vid_file} — skipping")
                continue
            vid_file = resolved

        cap = cv2.VideoCapture(vid_file)
        if not cap.isOpened():
            print(f"[WARN] Could not open {vid_file} — skipping")
            continue

        src_name = os.path.basename(vid_file)
        thresh_tag = f"  [thresh={video_info.get('threshold', '?'):.3f}]" \
            if isinstance(video_info.get("threshold"), float) else ""
        print(f"\n  --> Analyzing: {src_name} | env={env_label}{thresh_tag}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        frame_delay = max(1, int(1000 / fps))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_steps += 1
            frame_idx = total_steps

            # Apply vision shader BEFORE YOLO
            if condition == "dr":
                filtered = shader_fn(frame, seed=trial_seed)
            else:
                filtered = shader_fn(frame)
            filtered = cv2.resize(filtered, (640, 480))

            # YOLO inference
            results = yolo(filtered, verbose=False)
            confs = results[0].boxes.conf.tolist()
            cls_ids = results[0].boxes.cls.tolist()
            names = results[0].names
            detected_classes = ";".join([names[int(c)] for c in cls_ids]) if cls_ids else "none"
            max_conf = max(confs) if confs else 0.0
            usage_ratio = vlm.total_calls / total_steps

            # VLM trigger
            if 0.01 < max_conf < threshold and not vlm.busy:
                fi, ei, si = frame_idx, env_label, src_name
                vlm.call(
                    filtered,
                    on_done=lambda d, l, f=fi, e=ei, s=si: on_vlm_done(d, l, f, e, s)
                )
                status = "VLM_TRIGGERED"
            else:
                status = "YOLO_ONLY"

            # CSV log
            writer_csv.writerow([
                frame_idx, env_label, src_name,
                f"{max_conf:.4f}", detected_classes, status,
                "PENDING..." if status == "VLM_TRIGGERED" else "N/A",
                "0", f"{usage_ratio:.4f}"
            ])
            csvf.flush()

            # Build display frame
            annotated = results[0].plot()
            display = draw_hud(
                annotated, threshold, max_conf, status,
                current_vlm_text, condition,
                vlm.total_calls, total_steps
            )

            writer.write(display)

            if show_display:
                cv2.imshow(win_name, display)

                if status == "VLM_TRIGGERED":
                    # ── FREEZE: hold frame while VLM processes ───────────────
                    # FIX: capture freeze_start once here so elapsed is correct
                    freeze_frame = display.copy()
                    freeze_start = time.time()

                    # FIX: vlm.busy stays True until after on_done fires,
                    # so this loop now actually runs for the full VLM duration
                    while vlm.busy:
                        pulse = freeze_frame.copy()
                        elapsed = time.time() - freeze_start   # correct elapsed
                        alpha = 0.5 + 0.5 * abs(((time.time()) % 1.0) - 0.5) * 2
                        cv2.rectangle(pulse, (0, 38), (pulse.shape[1], 60), (0, 40, 140), -1)
                        cv2.putText(pulse,
                                    f"  VLM processing... ({elapsed:.1f}s)",
                                    (8, 54), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (int(100 * alpha), int(200 * alpha), 255),
                                    1, cv2.LINE_AA)
                        cv2.imshow(win_name, pulse)
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('q'):
                            goto_end = True
                            break

                    if not goto_end and current_vlm_text:
                        # VLM finished — show response for 2 seconds
                        response_frame = draw_hud(
                            annotated, threshold, max_conf, "YOLO_ONLY",
                            current_vlm_text, condition,
                            vlm.total_calls, total_steps
                        )
                        cv2.imshow(win_name, response_frame)
                        cv2.waitKey(2000)

                    if goto_end:
                        break

                else:
                    key = cv2.waitKey(frame_delay) & 0xFF
                    if key == ord('q'):
                        print("\n[Display] Quit pressed — stopping early.")
                        goto_end = True
                        break

        cap.release()

    # Wait for any final pending VLM call
    wait_start = time.time()
    while vlm.busy and (time.time() - wait_start) < 35:
        print("[VLM] Waiting for final response...")
        time.sleep(0.5)

    if show_display:
        cv2.destroyAllWindows()

    writer.release()
    csvf.close()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    usage_ratio = vlm.total_calls / max(total_steps, 1)

    print(f"\n  Frames processed : {total_steps}")
    print(f"  VLM calls        : {vlm.total_calls}  ({usage_ratio*100:.1f}% of frames)")
    print(f"  Avg VLM latency  : {avg_latency:.0f} ms")
    print(f"  CSV  → {csv_path}")
    print(f"  Video → {vid_path}")

    return avg_latency, usage_ratio

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GenAssist Phase 2 — Offline shader BO trials (live playback)"
    )
    parser.add_argument('--condition', required=True,
                        choices=['amd', 'glaucoma', 'rp', 'dr'])
    parser.add_argument('--bo_db', type=str, default=None,
                        help="Path to BO history CSV (default: bo_history_<condition>.csv)")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Manual threshold (omit to let BO suggest)")
    parser.add_argument('--video', type=str, default=None,
                        help="Path to a single MP4 to analyze")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Session folder — auto-discovers videos by threshold range")
    parser.add_argument('--thresh_min', type=float, default=0.01)
    parser.add_argument('--thresh_max', type=float, default=0.99)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_display', action='store_true',
                        help="Disable live cv2 window (headless mode)")
    args = parser.parse_args()

    condition = args.condition
    bo_db = args.bo_db if args.bo_db else f"bo_history_{condition}.csv"
    output_dir = args.output_dir or os.path.join("phase2_results", condition)
    show_display = not args.no_display

    # Threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\n[Manual] Using threshold = {threshold:.4f}")
    else:
        threshold = get_next_threshold(bo_db)
        print(f"\n[BO] Suggested threshold = {threshold:.4f}")
        if os.path.exists(bo_db):
            df = pd.read_csv(bo_db)
            print(f"     (based on {len(df)} previous trial(s) in {bo_db})")

    print(f"\n{'='*55}")
    print(f"  PHASE 2 TRIAL — LIVE PLAYBACK")
    print(f"  Condition  : {CONDITION_LABELS[condition]}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  BO history : {bo_db}")
    print(f"  Output dir : {output_dir}")
    print(f"  Display    : {'ON (press Q to quit early)' if show_display else 'OFF'}")
    print(f"{'='*55}\n")

    import random
    trial_seed = random.randint(0, 9999)

    # Video resolution
    if args.video:
        import glob, random as _rand
        if os.path.isdir(args.video):
            all_mp4s = glob.glob(os.path.join(args.video, "trial_x_*.mp4"))
            all_mp4s = [p for p in all_mp4s if os.path.getsize(p) > 10000]
            if not all_mp4s:
                print(f"[ERROR] No valid mp4s found in {args.video}")
                return
            chosen = _rand.choice(all_mp4s)
            env_label = os.path.basename(args.video).replace("_results", "")
            SOURCE_VIDEOS.clear()
            SOURCE_VIDEOS.append({"path": chosen, "env": env_label, "threshold": 0.0})
            print(f"[Video] Randomly selected: {os.path.basename(chosen)}\n")
        elif os.path.isfile(args.video):
            SOURCE_VIDEOS.clear()
            SOURCE_VIDEOS.append({"path": args.video, "env": "custom", "threshold": 0.0})
            print(f"[Video] Single video: {args.video}\n")
        else:
            print(f"[ERROR] Not a valid file or directory: {args.video}")
            return

    elif args.results_dir:
        if not os.path.isdir(args.results_dir):
            print(f"[ERROR] results_dir not found: {args.results_dir}")
            return
        discovered = discover_videos(args.results_dir, args.thresh_min, args.thresh_max)
        if not discovered:
            print(f"[ERROR] No videos found in {args.results_dir} between "
                  f"{args.thresh_min} and {args.thresh_max}")
            return
        SOURCE_VIDEOS.clear()
        SOURCE_VIDEOS.extend(discovered)
        session_name = os.path.basename(os.path.normpath(args.results_dir))
        output_dir = f"{session_name}_offline_analysis"
        print(f"[Video] {len(SOURCE_VIDEOS)} video(s) discovered in {args.results_dir}")
        for v in SOURCE_VIDEOS:
            print(f"        {os.path.basename(v['path'])}  (thresh={v['threshold']})")
        print(f"[Output] {output_dir}\n")

    else:
        print("[ERROR] Specify --video or --results_dir")
        return

    # Run trial
    avg_latency, usage_ratio = run_trial(
        condition, threshold, output_dir, trial_seed, show_display
    )

    # Human score
    print(f"\n{'='*55}")
    print("TRIAL COMPLETE — HUMAN FEEDBACK REQUIRED")
    print(f"Condition  : {CONDITION_LABELS[condition]}")
    print(f"Threshold  : {threshold:.4f}")
    print(f"VLM usage  : {usage_ratio*100:.1f}% of frames")
    print(f"Avg latency: {avg_latency:.0f} ms")
    print()
    print("Based on what you just watched, score the guidance quality:")
    print("  1 = VLM triggered too rarely / threshold too high / unsafe")
    print("  5 = Reasonable but not optimal")
    print("  10 = VLM triggered at exactly the right moments / smooth")
    print()

    while True:
        try:
            score = float(input("Rate 1-10: "))
            if 1 <= score <= 10:
                break
            print("Enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input.")

    save_trial_result(threshold, avg_latency, usage_ratio, score, bo_db)

    next_thresh = get_next_threshold(bo_db)
    print(f"\n[BO] Next suggested threshold for {condition}: {next_thresh:.4f}")
    print(f"     Run: python3 bo_offline.py --condition {condition} --video <path>")
    print()


if __name__ == '__main__':
    main()