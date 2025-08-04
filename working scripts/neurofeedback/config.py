import os

# ─── PARAMETERS ─────────────────────────────────────
VISUALISE_PLV   = True     # True to enable dummy PLV view
WINDOW_SIZE     = 256 * 1
STEP_SIZE       = 128
SAMPLING_RATE   = 256

# ─── SUBJECT & SESSION ──────────────────────────────
SUBJECT_ID      = "000"
SESSION_ID      = "001"
RESULTS_DIR     = "./nf_results"

# Derived paths
SUBJECT_DIR = os.path.join(RESULTS_DIR, f"Subject_{SUBJECT_ID}")
SESSION_DIR = os.path.join(SUBJECT_DIR, f"Session_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)
