# config.py

import os

# ─── MODE SELECTION ─────────────────────────────────────────────────────
METHOD        = 'plv'      # 'plv' or 'csp'
ADAPTATION    = True       # False = static, True = adaptive
ADAPT_N       = 10         # how many windows to accumulate before adapting

# ─── GAME PARAMETERS ────────────────────────────────────────────────────
NUM_LEVELS        = 10     # total number of levels
TRIALS_PER_LEVEL  = 20     # trials per level - https://infoscience.epfl.ch/server/api/core/bitstreams/548d8f37-01c2-4d4f-86b0-717797e9b8a8/content

# ─── DATA / MODEL PATHS ─────────────────────────────────────────────────
# Paths to exact training artifacts
TRAINING_DATA   = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\training_data.pkl"
GAT_MODEL_PT    = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\best_finetuned_model.pt"


# ─── REAL-TIME VISUALISATION ─────────────────────────────────────────────
VISUALISE_PLV   = True

# ─── EEG PREPROCESSING ───────────────────────────────────────────────────
WINDOW_SIZE     = 256 * 3   # 3 s @ 256 Hz
STEP_SIZE       = 128       # ~500 ms step
BUFFER_SIZE     = 10
THRESHOLD       = 3
SAMPLING_RATE   = 256

# ─── SUBJECT & SESSION ──────────────────────────────────────────────────
SUBJECT_ID      = "000"     # change per participant
SESSION_ID      = "001"     # change per session
RESULTS_DIR     = "./collection_results"  # base output dir

# Derived paths (don’t modify)
SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)
