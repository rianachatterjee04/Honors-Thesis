#!/bin/bash
# ============================================================
# GenAssist Phase 2 — AMD condition, per-video per-threshold
# Run from: in-lab/
# Each command = one trial. Watch the output video, score 1-10, hit enter.
# Output CSVs save to: in-lab/phase2_results/amd/
# BO history saves to: in-lab/bo_history_amd.csv
# ============================================================


"prompt": (
                    "Identify obstacles and hazards in the room. "
                    "Note: The black blurry circle in the center is a Age-Related Macular Degeneration simulation- ignore it completely. "
                    "Focus only on the clear areas of the image to describe the actual floor and path. "
                    " Answer in 2 breif sentences."
                ),


# --- hallway_results (12 trials) ---
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.0.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.1.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.32.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.335.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.37.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.375.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.43.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.435.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.485.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.5.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.515.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/hallway_results/trial_x_0.9.mp4

# --- computer_room_results (11 trials) ---
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.1.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.13.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.16.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.26.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.39.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.47.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.5.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.54.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.55.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.72.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results/trial_x_0.9.mp4

# --- computer_room_results_part_two (17 trials) ---
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.1.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.19.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.32.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.325.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.33.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.335.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.35.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.39.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.41.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.42.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.455.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.47.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.5.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.74.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.8.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.85.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/computer_room_results_part_two/trial_x_0.9.mp4

# --- robot_room_results_first_trial (19 trials) ---
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.05.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.1.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.15.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.18.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.19.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.2.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.21.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.25.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.29.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.37.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.41.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.44.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.45.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.5.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.63.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.66.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.8.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.81.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_first_trial/trial_x_0.9.mp4

# --- robot_room_results_second_trial (22 trials) ---
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.1.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.13.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.19.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.23.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.25.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.3.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.32.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.37.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.39.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.46.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.5.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.51.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.64.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.66.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.67.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.68.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.69.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.78.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.84.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.86.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.88.mp4
python3 in_line_test/bo_offline.py --condition amd --video results/robot_room_results_second_trial/trial_x_0.9.mp4