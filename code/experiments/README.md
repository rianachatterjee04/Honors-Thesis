> **Paths:** From the repository root, scripts in this folder are run as `python3 code/experiments/<script>.py`. Result videos live under `results/`.

vlm server- running on port 5000
ollama server- running on http://localhost:11434/api/generate



if running the C++ code-
yolo export model=yolov8n.pt format=onnx
cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov8n.onnx");
package.xml dependencies-
<depend>rclcpp</depend>
<depend>sensor_msgs</depend>
<depend>geometry_msgs</depend>
<depend>cv_bridge</depend>
<depend>opencv</depend>




Bayesian Optimization-
Overview
This system uses multi-objective Bayesian optimization to automatically personalize a single control threshold x ∈ [0, 1] that determines when the robot escalates from fast YOLO-based perception to slower Vision-Language Model (VLM) reasoning.
The goal is to balance navigation safety, system latency, and computational cost.

Optimization Objectives
For a given threshold x, a short human–robot interaction trial produces noisy measurements of three objectives:
S(x) — Task success / safety (maximize)

Implemented as collision-free navigation:
S(x) = 1 if no collision occurs during the trial
S(x) = 0 if a collision occurs
L(x) — Average system latency (minimize)
Measured as the mean perception-to-action latency per control step (milliseconds)
P(x) — Power or computational cost (minimize)
Approximated by the fraction of timesteps in which the VLM is invoked
Each objective is treated as a noisy function evaluation due to sensor noise, environmental variability, and interaction stochasticity.

Multi-Objective Optimization Problem
The optimization problem is formulated as:
Maximize S(x)
Minimize L(x) and P(x)
Subject to x ∈ [0, 1]
For implementation, all objectives are converted into a minimization form:
f(x) = ( -S(x), L(x), P(x) )
The goal is to identify Pareto-optimal threshold values that balance safety, latency, and efficiency.
Practical Constraints
To ensure real-time deployability, the optimization is restricted by system constraints:
L(x) <= L_max
(maximum allowable average latency)
P(x) <= P_max
(maximum allowable fraction of VLM usage)
Only threshold values satisfying these constraints are considered feasible during optimization.
Gaussian Process Surrogate Models
Each objective is modeled using an independent Gaussian Process (GP):
A GP is learned for:
S(x) (task success)
L(x) (latency)
P(x) (power / computation)
The GP models provide a probabilistic estimate (mean and uncertainty) of each objective across the threshold space based on observed interaction data.

Bayesian Optimization Loop
During an induction phase, the system iteratively:
Selects a candidate threshold x
Runs a short navigation trial using that threshold

Measures:
Collision outcome (S(x))
Average latency (L(x))
VLM usage fraction (P(x))
Updates the GP surrogate models with the new observation
Selects the next threshold by maximizing Expected Hypervolume Improvement (EHVI) over the feasible set
EHVI encourages exploration of thresholds that improve the trade-off between safety, latency, and computational cost.

Outcome
The optimization converges to a personalized operating threshold that:
Preserves collision-free navigation
Minimizes unnecessary VLM calls
Maintains real-time system performance
This threshold is then used for the full navigation task (e.g., guiding the robot forward over a fixed distance).

Implementation Notes
Bayesian optimization is implemented using BoTorch
Gaussian Processes are used as surrogate models for each objective
Constraint handling ensures latency and power budgets are respected
Collision avoidance is treated as a primary success metric



# BO OFFLINE ANALYSIS

# Run on all hallway videos between threshold 0.30–0.45
python3 bo_offline.py --condition amd \
  --results_dir ../results/hallway_results \
  --thresh_min 0.3 --thresh_max 0.45

# That will auto-discover and print:
#   trial_x_0.32.mp4   (thresh=0.320)
#   trial_x_0.335.mp4  (thresh=0.335)
#   trial_x_0.37.mp4   (thresh=0.370)
#   trial_x_0.375.mp4  (thresh=0.375)
# and save everything to: hallway_results_offline_analysis/


# Run on all hallway videos between threshold 0.30–0.45
python3 in_line_test/bo_offline.py --condition amd --results_dir results/hallway_results --thresh_min 0.3 --thresh_max 0.45

# Computer room
python3 in_line_test/bo_offline.py --condition amd --results_dir results/computer_room_results --thresh_min 0.3 --thresh_max 0.45

# Computer room part two
python3 in_line_test/bo_offline.py --condition amd --results_dir results/computer_room_results_part_two --thresh_min 0.3 --thresh_max 0.45

# Robot room 1
python3 in_line_test/bo_offline.py --condition amd --results_dir results/robot_room_results_first_trial --thresh_min 0.3 --thresh_max 0.45

# Robot room 2
python3 in_line_test/bo_offline.py --condition amd --results_dir results/robot_room_results_second_trial --thresh_min 0.3 --thresh_max 0.45

# That will auto-discover and print:
#   trial_x_0.32.mp4   (thresh=0.320)
#   trial_x_0.335.mp4  (thresh=0.335)
#   trial_x_0.37.mp4   (thresh=0.370)
#   trial_x_0.375.mp4  (thresh=0.375)
# and save everything to: hallway_results_offline_analysis/