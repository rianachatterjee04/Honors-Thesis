# GenAssist — Honors Thesis

**Author:** Riana Chatterjee  
**Institution:** Barrett, The Honors College — Arizona State University  

Code, experiments, and media from an honors thesis on **personalized assistive navigation** for people with low vision. The system combines fast YOLO perception with slower Vision-Language Model (VLM) reasoning on a Unitree Go2 robot, and uses **Bayesian optimization** to learn a per-user / per-condition threshold that balances safety, latency, and compute cost.

> Repository: [github.com/rianachatterjee04/Honors-Thesis](https://github.com/rianachatterjee04/Honors-Thesis)

---

## Research overview

Many low-vision conditions (AMD, diabetic retinopathy, glaucoma, retinitis pigmentosa) degrade different parts of the visual field. A single fixed perception policy is a poor fit: always running a VLM is too slow and expensive; always relying on YOLO can miss context that a language model would catch.

**GenAssist** treats the YOLO→VLM handoff as a tunable threshold \(x \in [0,1]\) and personalizes it with multi-objective Bayesian optimization:

| Objective | Role |
|-----------|------|
| **S(x)** — safety / task success | Maximize (e.g., collision-free trial) |
| **L(x)** — latency | Minimize mean perception-to-action time |
| **P(x)** — VLM usage / compute | Minimize fraction of frames that call the VLM |

Gaussian-process surrogates (BoTorch) and acquisition functions (UCB / EHVI-style search) propose the next threshold to try during short robot or offline video trials. Condition-specific shaders (AMD scotoma, DR blur, glaucoma tunnel, RP periphery loss) support Phase-2 offline evaluation on recorded runs.

---

## Repository layout

```text
.
├── code/
│   ├── application/           # Modular perception helpers (depth, YOLO proximity, Q&A, directions)
│   ├── combination_pipeline/  # End-to-end YOLO + depth + VLM navigation pipeline
│   ├── perception_modes/      # Speed / balance / descriptive / general YOLO modes + analytics
│   ├── bayesian_optimization/ # Online BO trial scripts for the Go2
│   ├── experiments/           # In-lab BO, VLM/OWL servers, offline Phase-2 shaders, Pareto tools
│   ├── yolo_vlm_analysis/     # Notebooks comparing YOLO vs VLM switch-off behavior
│   ├── robot_control/         # CrewAI + ROS 2 Unitree Go2 control agent
│   └── database/              # Lightweight DB helpers
├── models/                    # YOLO / walkable-segmentation weights + dataset YAML
├── training/                  # Walkable-area dataset + Ultralytics training runs
├── results/                   # Trial videos, CSVs, VR overlays (hallway, rooms, conditions)
├── outputs/                   # Pipeline demo clips + Phase-2 sample outputs
├── media/                     # Figures, demo video, VR overlay HTML
├── requirements.txt
└── README.md
```

---

## Key components

### 1. Perception pipeline (`code/combination_pipeline/`, `code/application/`)
- YOLOv8 detection / segmentation and a custom **walkable-area** model  
- Optional depth cues and VLM queries (Ollama / local HTTP VLM server) for directions and scene description  

### 2. Operating modes (`code/perception_modes/`)
- **Speed** — lean on YOLO for low latency  
- **Balance** — mixed YOLO / VLM usage  
- **Descriptive** — richer VLM scene descriptions  
- Analytics notebooks summarize mode trade-offs  

### 3. Bayesian optimization (`code/bayesian_optimization/`, `code/experiments/`)
- Online episodic trials on the robot (`bo_trials*.py`, `bayesian_optimization.py`)  
- Offline Phase-2 evaluation with vision-condition shaders (`bo_offline.py`, `bo_offline_vlm_only.py`)  
- Conditions: **AMD**, **DR**, **glaucoma**, **RP**  

### 4. Robot control (`code/robot_control/`)
- CrewAI agent scripts publishing Unitree ROS 2 requests (stand, navigate, etc.)  
- Requires Ubuntu 22.04 + ROS 2 Humble + `unitree_ros2` on the robot compute  

### 5. Results & media
- `results/` — hallway / computer-room / robot-room BO trials and VR condition folders  
- `media/overlays/eye-condition-video.html` — side-by-side VR vision overlay comparison  
- `media/figures/` — thesis figures and experiment screenshots  

---

## Setup

```bash
git clone https://github.com/rianachatterjee04/Honors-Thesis.git
cd Honors-Thesis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Optional services (for live / offline VLM trials):**
- VLM HTTP server on `http://localhost:5000/vlm` (see `code/experiments/vlm_server.py`)  
- Ollama at `http://localhost:11434/api/generate` when using local LLMs  

**Robot (Unitree Go2):**
```bash
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/cyclonedx_ws/install/setup.bash
# then run agent / BO scripts on the robot compute module
```

Model weights live under `models/` (e.g. `yolov8n.pt`, `walkable_model1.pt`). Some notebooks under `code/yolo_vlm_analysis/` also keep local copies of weights for self-contained runs.

---

## Example workflows

### Online BO trial (on robot)
```bash
cd code/experiments
# ensure VLM server is up, then:
python3 bayesian_optimization.py
# or force a threshold:
python3 bayesian_optimization.py --threshold 0.45
```

### Offline Phase-2 shader BO (from repo root)
```bash
python3 code/experiments/bo_offline.py \
  --condition amd \
  --results_dir results/hallway_results \
  --thresh_min 0.3 --thresh_max 0.45
```

Supported `--condition` values include `amd`, `dr`, `glaucoma`, and `rp` (see script for labels and shaders).

### Perception modes
```bash
cd code/perception_modes
python3 speed_mode.py
python3 balance_mode.py
python3 descriptive_mode.py
```

### VR overlay viewer
Open `media/overlays/eye-condition-video.html` in a browser (videos referenced by the page should live under `results/VR_Overlays/` or be updated to match your local paths).

---

## Eye conditions studied

| Condition | Shader idea (offline) |
|-----------|------------------------|
| **AMD** | Central scotoma / metamorphopsia; periphery preserved |
| **Diabetic retinopathy (DR)** | Patchy blur / scotomas |
| **Glaucoma** | Tunnel / peripheral field loss |
| **Retinitis pigmentosa (RP)** | Severe peripheral loss |

Prompts used with the VLM instruct the model to ignore simulated artifacts and reason about the clear regions of the frame.

---

## Notes

- Paths inside some robot scripts still point at the original onboard directory (`/home/unitree/GenAssist_Riana/...`). Update those when redeploying.  
- Large trial videos and CSVs are kept under `results/` so experiment provenance stays with the code.  
- Virtual environments were removed from this tree; recreate with `requirements.txt` as above.  

---

## Acknowledgments

Developed as undergraduate honors thesis research at Arizona State University (Barrett, The Honors College), including in-lab evaluation with the Unitree Go2 platform.
