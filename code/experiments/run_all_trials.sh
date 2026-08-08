#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# run_all_trials.sh  —  macOS compatible (sh/zsh safe)
# Runs BO trials for each condition, stopping when any threshold has been
# tried 3 times (convergence signal).
# - Creates a brand new bo_history_offline.csv inside each _Offline folder
# - Shows live cv2 window for each video
# - Stores annotated video + CSV in <CONDITION>_Offline/
#
# Usage (run from your in-lab/ directory):
#   ./in_line_test/run_all_trials.sh           # all 4 conditions
#   ./in_line_test/run_all_trials.sh AMD       # one condition only
# ─────────────────────────────────────────────────────────────────────────────

VR_DIR="results/VR"
SCRIPT="in_line_test/bo_offline.py"
REPEAT_STOP=3   # stop when any threshold has been tried this many times

get_condition() {
  case "$1" in
    AMD)      echo "amd" ;;
    DR)       echo "dr" ;;
    glaucoma) echo "glaucoma" ;;
    RP)       echo "rp" ;;
    *)        echo "" ;;
  esac
}

if [ -n "$1" ]; then
  FOLDERS="$1"
else
  FOLDERS="AMD DR glaucoma RP"
fi

for FOLDER in $FOLDERS; do
  CONDITION=$(get_condition "$FOLDER")
  if [ -z "$CONDITION" ]; then
    echo "[WARN] Unknown folder '$FOLDER' — skipping"
    continue
  fi

  INPUT_DIR="${VR_DIR}/${FOLDER}"
  OUTPUT_DIR="${VR_DIR}/${FOLDER}_Offline"
  BO_HISTORY="${OUTPUT_DIR}/bo_history_offline.csv"

  if [ ! -d "$INPUT_DIR" ]; then
    echo "[WARN] Directory not found: $INPUT_DIR — skipping"
    continue
  fi

  # Create output folder and wipe any previous offline BO history
  mkdir -p "$OUTPUT_DIR"
  if [ -f "$BO_HISTORY" ]; then
    echo "  [INFO] Removing old $BO_HISTORY for fresh run"
    rm "$BO_HISTORY"
  fi

  # Collect videos once
  VIDEOS=$(ls "${INPUT_DIR}"/trial_x_*_raw.mp4 2>/dev/null | sort -V)
  if [ -z "$VIDEOS" ]; then
    echo "[WARN] No _raw.mp4 files found in $INPUT_DIR — skipping"
    continue
  fi

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Condition : $FOLDER  (--condition $CONDITION)"
  echo "  Input     : $INPUT_DIR"
  echo "  Output    : $OUTPUT_DIR"
  echo "  BO history: $BO_HISTORY  (fresh)"
  echo "  Stop when : any threshold tried ${REPEAT_STOP}x"
  echo "════════════════════════════════════════════════════"
  echo "  Found videos:"
  for V in $VIDEOS; do echo "    $(basename $V)"; done
  echo ""

  trial_count=0

  while true; do

    # ── Convergence check ──────────────────────────────────────────────────
    DONE=$(python3 - <<EOF
import pandas as pd, os
db = "$BO_HISTORY"
stop = $REPEAT_STOP
if not os.path.exists(db):
    print("no")
else:
    try:
        df = pd.read_csv(db)
        if len(df) == 0:
            print("no")
        elif df['threshold'].round(3).value_counts().max() >= stop:
            best = df['threshold'].round(3).value_counts().idxmax()
            count = df['threshold'].round(3).value_counts().max()
            print(f"yes|threshold={best:.3f} tried {count}x")
        else:
            tried = len(df)
            unique = df['threshold'].round(3).nunique()
            print(f"no|{tried} trials so far, {unique} unique thresholds")
    except Exception as e:
        print(f"no|error reading db: {e}")
EOF
)

    # Parse result
    STATUS=$(echo "$DONE" | cut -d'|' -f1)
    MSG=$(echo "$DONE" | cut -d'|' -f2)

    if [ "$STATUS" = "yes" ]; then
      echo ""
      echo "  [CONVERGED] $MSG"
      echo "  Stopping trials for $FOLDER."
      break
    else
      echo "  [BO status] $MSG"
    fi

    # ── Cycle through videos, round-robin ─────────────────────────────────
    for VIDEO in $VIDEOS; do

      # Re-check convergence before each video in case we hit it mid-cycle
      DONE2=$(python3 - <<EOF
import pandas as pd, os
db = "$BO_HISTORY"
stop = $REPEAT_STOP
if not os.path.exists(db):
    print("no")
else:
    try:
        df = pd.read_csv(db)
        if len(df) > 0 and df['threshold'].round(3).value_counts().max() >= stop:
            print("yes")
        else:
            print("no")
    except:
        print("no")
EOF
)
      if [ "$DONE2" = "yes" ]; then
        break
      fi

      trial_count=$((trial_count + 1))
      echo "──────────────────────────────────────────────────"
      echo "  Trial #${trial_count} — $(basename $VIDEO)"
      echo "──────────────────────────────────────────────────"

      python3 "$SCRIPT" \
        --condition "$CONDITION" \
        --video "$VIDEO" \
        --output_dir "$OUTPUT_DIR" \
        --bo_db "$BO_HISTORY"

      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Exited with code $EXIT_CODE for $VIDEO — continuing..."
      fi
      echo ""

    done

    # If inner loop broke due to convergence, break outer while too
    DONE3=$(python3 - <<EOF
import pandas as pd, os
db = "$BO_HISTORY"
stop = $REPEAT_STOP
if not os.path.exists(db):
    print("no")
else:
    try:
        df = pd.read_csv(db)
        if len(df) > 0 and df['threshold'].round(3).value_counts().max() >= stop:
            print("yes")
        else:
            print("no")
    except:
        print("no")
EOF
)
    if [ "$DONE3" = "yes" ]; then
      echo ""
      echo "  [CONVERGED] Stopping trials for $FOLDER."
      break
    fi

  done

  echo ""
  echo "[DONE] $FOLDER complete after $trial_count trial(s) → $OUTPUT_DIR"
  echo ""

done

echo "════════════════════════════════════════════════════"
echo "All conditions complete."
echo "════════════════════════════════════════════════════"