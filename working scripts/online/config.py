# static vs adaptive
ADAPTATION   = False      # True or False  
ADAPT_N      = 20        # how many windows to accumulate before adapting

# paths to your exact training artifacts
TRAINING_DATA = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\training_data.pkl"
GAT_MODEL_PT  = r"C:\Users\uceerjp\Desktop\PhD\Year 2\online experiments\working scripts\training\best_finetuned_model.pt"

# keep the rest as before
WINDOW_SIZE    = 256*3
STEP_SIZE      = 128
BUFFER_SIZE    = 10
THRESHOLD      = 3
SAMPLING_RATE  = 256
