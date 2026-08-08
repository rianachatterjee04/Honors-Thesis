#!/usr/bin/env python3
"""
Diabetic Retinopathy Video Shader
==================================
Applies a per-frame DR simulation overlay to a video:
  - Overall frame darkening (15% brightness reduction)
  - Random scotomas (dark blotchy patches that shift each frame)
  - Visual noise / haze (gaussian noise + slight blur)
  - Vascular-damage scatter (small dark hemorrhage-like dots)

Usage:
    python apply_dr_shader.py
  or override defaults:
    python apply_dr_shader.py --input /path/to/video.mp4 --output /path/to/output.mp4

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path

# ── DEFAULT PATHS ────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/Users/rianachatterjee/Downloads/in-lab/YOLO_VLM_switch_off/videos/clip_28_to_29_ffmpeg.mp4"
DEFAULT_OUTPUT = "/Users/rianachatterjee/Downloads/in-lab/YOLO_VLM_switch_off/videos/clip_28_to_29_DR_shader.mp4"

# ── DR SHADER PARAMETERS (tune these to taste) ───────────────────────────────
DARKENING          = 0.78    # overall brightness multiplier (1.0 = no change, 0.78 = 22% darker)
NOISE_SIGMA        = 18      # gaussian noise standard deviation (0 = off)
BLUR_KERNEL        = 3       # light haze blur kernel size (1 = off, must be odd)
NUM_SCOTOMAS       = 6       # number of random dark patches per frame
SCOTOMA_ALPHA      = 0.72    # how dark scotomas are (0 = invisible, 1 = fully black)
SCOTOMA_SIZE_RANGE = (0.04, 0.14)   # scotoma radius as fraction of frame min-dimension
SCOTOMA_DRIFT      = 0.015   # how much scotoma centres drift between frames (fraction)
NUM_HEMORRHAGES    = 22      # tiny dark dot scatter (retinal microaneurysms)
HEMORRHAGE_RADIUS  = (1, 4)  # radius range in pixels
SEED               = 42      # base RNG seed for reproducibility


def build_scotoma_mask(h, w, centres, radii):
    """Build a float32 darkness mask [0,1] from a list of elliptical scotoma centres."""
    mask = np.zeros((h, w), dtype=np.float32)
    for (cx, cy), r in zip(centres, radii):
        Y, X = np.ogrid[:h, :w]
        rx, ry = r, r * np.random.uniform(0.5, 1.0)   # slightly elliptical
        angle  = np.random.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        Xr = (X - cx) * cos_a + (Y - cy) * sin_a
        Yr = -(X - cx) * sin_a + (Y - cy) * cos_a
        dist = np.sqrt((Xr / max(rx, 1)) ** 2 + (Yr / max(ry, 1)) ** 2)
        # soft edge: 1 at centre → 0 at edge
        soft = np.clip(1.0 - dist, 0, 1) ** 1.8
        mask = np.maximum(mask, soft)
    return mask


def apply_dr_frame(frame, h, w, scotoma_centres, scotoma_radii, rng):
    """Apply all DR effects to one frame. Returns a uint8 BGR frame."""
    out = frame.astype(np.float32)

    # 1. Global darkening
    out *= DARKENING

    # 2. Gaussian noise (simulate haze / visual noise from vascular damage)
    if NOISE_SIGMA > 0:
        noise = rng.normal(0, NOISE_SIGMA, out.shape).astype(np.float32)
        out += noise

    # 3. Mild haze blur
    if BLUR_KERNEL > 1:
        out = cv2.GaussianBlur(out, (BLUR_KERNEL, BLUR_KERNEL), 0)

    # 4. Scotoma mask — dark blotchy regions
    scotoma_mask = build_scotoma_mask(h, w, scotoma_centres, scotoma_radii)
    scotoma_mask_3ch = scotoma_mask[:, :, np.newaxis]           # broadcast over BGR
    out = out * (1.0 - SCOTOMA_ALPHA * scotoma_mask_3ch)

    # 5. Microaneurysm / hemorrhage dots
    n_hem = rng.integers(max(0, NUM_HEMORRHAGES - 5), NUM_HEMORRHAGES + 5)
    for _ in range(n_hem):
        hx = int(rng.integers(0, w))
        hy = int(rng.integers(0, h))
        hr = int(rng.integers(*HEMORRHAGE_RADIUS))
        # dark reddish dot
        cv2.circle(out, (hx, hy), hr, (0, 0, int(rng.integers(0, 40))), -1)

    # Clamp and return
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Video : {w}x{h} @ {fps:.2f} fps  |  {total_frames} frames")
    print(f"Shader: darkening={DARKENING}, noise_sigma={NOISE_SIGMA}, "
          f"scotomas={NUM_SCOTOMAS}, hemorrhages={NUM_HEMORRHAGES}")
    print("Processing frames...", flush=True)

    min_dim = min(w, h)
    rng     = np.random.default_rng(SEED)

    # Initialise scotoma centres and radii (will drift per frame)
    scotoma_centres = [
        (int(rng.integers(0, w)), int(rng.integers(0, h)))
        for _ in range(NUM_SCOTOMAS)
    ]
    scotoma_radii = [
        int(rng.uniform(*SCOTOMA_SIZE_RANGE) * min_dim)
        for _ in range(NUM_SCOTOMAS)
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Drift scotoma centres slightly each frame (random walk)
        scotoma_centres = [
            (
                int(np.clip(cx + rng.integers(-int(SCOTOMA_DRIFT * w),
                                              int(SCOTOMA_DRIFT * w) + 1), 0, w - 1)),
                int(np.clip(cy + rng.integers(-int(SCOTOMA_DRIFT * h),
                                              int(SCOTOMA_DRIFT * h) + 1), 0, h - 1)),
            )
            for (cx, cy) in scotoma_centres
        ]
        # Occasionally spawn a new scotoma / retire one
        if rng.random() < 0.03 and NUM_SCOTOMAS > 1:
            idx = int(rng.integers(0, NUM_SCOTOMAS))
            scotoma_centres[idx] = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            scotoma_radii[idx]   = int(rng.uniform(*SCOTOMA_SIZE_RANGE) * min_dim)

        processed = apply_dr_frame(frame, h, w, scotoma_centres, scotoma_radii, rng)
        writer.write(processed)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {frame_idx}/{total_frames}  ({pct:.1f}%)", flush=True)

    cap.release()
    writer.release()
    print(f"\nDone! Output saved to:\n  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply Diabetic Retinopathy shader to a video.")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to input .mp4")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output .mp4")

    global DARKENING, NOISE_SIGMA, NUM_SCOTOMAS, SCOTOMA_ALPHA, NUM_HEMORRHAGES

    # Shader tuning overrides
    parser.add_argument("--darkening",     type=float, default=DARKENING,
                        help="Brightness multiplier, e.g. 0.78 = 22%% darker")
    parser.add_argument("--noise-sigma",   type=int,   default=NOISE_SIGMA,
                        help="Gaussian noise strength (0 = off)")
    parser.add_argument("--num-scotomas",  type=int,   default=NUM_SCOTOMAS,
                        help="Number of dark scotoma patches per frame")
    parser.add_argument("--scotoma-alpha", type=float, default=SCOTOMA_ALPHA,
                        help="Scotoma darkness (0=invisible, 1=fully black)")
    parser.add_argument("--num-hemorrhages", type=int, default=NUM_HEMORRHAGES,
                        help="Number of microaneurysm dots scattered per frame")
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    DARKENING       = args.darkening
    NOISE_SIGMA     = args.noise_sigma
    NUM_SCOTOMAS    = args.num_scotomas
    SCOTOMA_ALPHA   = args.scotoma_alpha
    NUM_HEMORRHAGES = args.num_hemorrhages

    process_video(args.input, args.output)


if __name__ == "__main__":
    main()