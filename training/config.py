
# Data paths
DATA_DIR = "/home/kkolpetinou/Task07_Pancreas_Preprocessed"
CHECKPOINT_DIR = "./checkpoints"
LOG_NAME = "segformer3d_variant_logs"

# Data loading
BATCH_SIZE = 2
NUM_WORKERS = 2
TRAIN_SIZE = 0.7
VAL_SIZE = 0.3
PATCH_SIZE = (32, 160, 208)

# Model
IN_CHANNELS = 1
NUM_CLASSES = 3
EMBED_DIMS = [32, 64, 160, 256]
DEPTHS = [2, 2, 2, 2]
NUM_HEADS = [1, 2, 5, 8]
SR_RATIOS = [4, 2, 1, 1]
DECODER_C = 128
ASPP_RATES = (1, 2, 3)
FUSE_ALL = True

# Training
LR = 3e-4
WEIGHT_DECAY = 5e-5
WARMUP_STEPS = 9000
TOTAL_STEPS = 150000
POLY_POWER = 0.9
MAX_STEPS = 250_000
GRAD_CLIP = 1.0

# Early stopping
PATIENCE = 150
MIN_DELTA = 0.001

