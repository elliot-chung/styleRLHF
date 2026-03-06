"""Project configuration: slots, paths, and hyperparameters."""

from pathlib import Path

# Outfit slots (fixed order)
TOP_SLOT = "Top"
BOTTOM_SLOT = "Bottom"
SHOES_SLOT = "Shoes"
ACCESSORY_SLOT = "Accessory"
SLOTS = [TOP_SLOT, BOTTOM_SLOT, SHOES_SLOT, ACCESSORY_SLOT]
NUM_SLOTS = len(SLOTS)
NUM_CHUNKS = NUM_SLOTS * (NUM_SLOTS - 1) / 2

# Special token: empty slot
EMPTY_TOKEN_ID = 0
# In the catalog we use 1-indexed item IDs; 0 is reserved for empty

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = ROOT / "outputs"
CORPUS_CSV = DATA_DIR / "catalog.csv"

FPI_ARCHIVE = DATA_DIR / "fashion_product_images(small).zip"

# SFT
SFT_BATCH_SIZE = 32
SFT_LR = 1e-4
SFT_EPOCHS = 20
SFT_EMBED_DIM = 128
SFT_NHEADS = 4
SFT_NLAYERS = 2
SFT_DIM_FEEDFORWARD = 256
SFT_DROPOUT = 0.1

# Reward model
RM_EMBED_DIM = 128
RM_HIDDEN = 64
RM_LR = 1e-4
RM_EPOCHS = 50
RM_BATCH_SIZE = 32

# PPO
PPO_BATCH_SIZE = 64
PPO_LR = 1e-5
PPO_KL_COEF = 0.1
PPO_EPOCHS = 5
PPO_STEPS = 500

# Preference collection
PREFERENCE_NUM_PAIRS = 1000
VLM_TEMPERATURE = 0.8
VLM_JUDGE_PROMPT = (
    "Which of these two outfits (A or B) is more stylistically coherent and looks better? "
    "Briefly explain why, then output exactly 'A' or 'B' on the last line."
)
MAX_CONCURRENT_JUDGES = 10
# RPM limits (requests per minute) for rate-limited parallel judge calls
OPENAI_RPM_LIMIT = 60  # placeholder; adjust to your OpenAI tier
GEMINI_RPM_LIMIT = 60  # placeholder; adjust to your Gemini quota
GEMMA_RPM_LIMIT = 30   # 30 requests per minute
GEMMA_RPD_LIMIT = 14400  # 14400 requests per day