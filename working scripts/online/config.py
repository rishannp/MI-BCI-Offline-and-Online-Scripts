# config.py

# Static vs Adaptive
ADAPTATION      = True        # False = static, True = adaptive
ADAPT_N         = 10          # how many windows to accumulate before adapting

# Game parameters (new)
NUM_LEVELS        = 10        # total number of levels in the game
TRIALS_PER_LEVEL  = 20        # number of trials per level (used for level‐up and adaptation)

# Paths to exact training artifacts
TRAINING_DATA   = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\training_data.pkl"
GAT_MODEL_PT    = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\best_finetuned_model.pt"

# Core BCI parameters
WINDOW_SIZE     = 256 * 3     # 3 s @ 256 Hz
STEP_SIZE       = 128         # ≈500 ms step (75% overlap)
BUFFER_SIZE     = 10          # smoothing buffer for raw predictions
THRESHOLD       = 3           # majority-vote threshold
SAMPLING_RATE   = 256         # Hz

# Subject/session identifiers
SUBJECT_ID      = "001"       # Change per participant
RESULTS_DIR     = "./results"
