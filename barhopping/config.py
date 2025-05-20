import os
import yaml

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration from YAML
with open(os.path.join(PROJECT_ROOT, "config", "default.yml"), "r") as f:
    config = yaml.safe_load(f)

CITY = config["city"]
NUM_BARS = config["num_bars"]
NUM_PIC = config["num_pics"]
NUM_REV = config["num_review"]
DB_PATH = config["db_path"]
HF_TOKEN = os.getenv("HF_TOKEN", config["hf_token"])
OPENAI_KEY = os.getenv("OPENAI_KEY", config["openai_key"])
BARS_DB = os.path.join(PROJECT_ROOT, "bars_tpe.db")
QUESTIONS_DB = config["questions_db"]
NEG_DB = config["neg_db"]
GEMMA_MODEL = config["gemma_model"]
GRANITE_MODEL = config.get("granite_model", "ibm-granite/granite-embedding-125m-english")