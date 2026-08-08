# 🤖 Bayesian Optimization for YOLO-VLM Thresholding
This project uses BoTorch (Bayesian Optimization) to automatically find the optimal confidence threshold (x) for a Go2 robot. The goal is to balance VLM usage (expensive/slow) with Processing Latency while maintaining navigation safety.

# 📋 How it Works: The "Stateful" Loop
Because the robot requires a manual autonomy reset and a warm-up waypoint between trials, this script is designed to be episodic:

The Brain: On startup, the script reads bo_history.csv to see past results.

The Prediction: It fits a Gaussian Process to that data and uses an Acquisition Function (UCB) to pick the next threshold to test.

The Trial: The robot walks for 20 seconds using that threshold.

The Memory: Before shutting down, the script saves the average latency and VLM usage ratio back to the CSV.

# 🚀 Setup & Installation
Ensure you have the following dependencies installed on the robot's compute module:

Bash
pip install torch botorch gpytorch pandas ultralytics
Ensure your VLM Server is running at the endpoint specified in the script (default: http://localhost:5000/vlm).

# 🏃 Running the Experiment
To conduct a full optimization (recommended: 10–15 trials), follow these steps:

1. Initialization (First Run)

If this is a brand new experiment, delete the old history to start fresh:

Bash
rm bo_history.csv
2. The Trial Cycle

Perform the following steps for every trial:

Step A: Physical Reset Place the robot at the starting line of your obstacle course.

Step B: Autonomy Warm-up Restart your ROS2 autonomy stack and send the "warm-up" waypoint to calibrate the robot's hardcoded positional values.

Step C: Execute BO Trial Run the main script:

Bash
python3 bayesian_optimization.py
The script will print the suggested threshold, run for 20 seconds, save the results, and exit automatically.

# 3. Manual Override (Optional)

If you want to test a specific threshold without the "Brain" deciding for you:

Bash
python3 bayesian_optimization.py --threshold 0.45
📊 Understanding the Data
bo_history.csv

This is the master database. It tracks:

Threshold: The x value tested.

Latency: The average time (ms) for VLM/Vision processing.

Usage: Percentage of frames that triggered a VLM call.

Reward: A calculated score where higher is better.

in_line_test/ folder

Each trial generates:

trial_x_[VALUE].mp4: Annotated video showing YOLO boxes and VLM responses.

bo_trial_x_[VALUE].csv: Frame-by-frame raw data for deep analysis.

# 🧠 The Optimization Logic
The "Learning" happens via the Reward Function defined in the save_trial_result function:

Reward=−(Latency/1000)−(Usage×5.0)
If the robot is too "scared" (threshold too high), it calls the VLM too much, and the Usage penalty lowers the reward.

If the robot is too "fast" but misses hazards, the performance metrics will reflect that in the vision logs.


# How do you know when you've run "enough"?

In Bayesian Optimization, you know you are done when the algorithm reaches Convergence. You can tell this is happening by looking at two things in your bo_history.csv:

The "Exploration" stops: In early trials (4–8), the script might jump around a lot (e.g., trying 0.2, then 0.7, then 0.4).

The Threshold stabilizes: In later trials (10–15), the script will start picking numbers that are very close to each other (e.g., 0.31, 0.32, 0.315). This means it has found the "Peak" of the reward and is just fine-tuning.

# When to "Adjust" the Environment

You only change the environment after the Bayesian Optimization has finished (usually after 10–15 trials).

Stage 1 (The Training): Run 10 trials in "Room A" until the BO finds an optimal threshold (e.g., 0.35).

Stage 2 (The Validation): Now that you have that "Best" number, take the robot to "Room B" or move the cones around. This tests if the threshold the BO learned is actually "robust" or if it only works in one room.