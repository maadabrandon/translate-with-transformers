import os 
from pathlib import Path 

PARENT_DIR = Path("_file_").parent.resolve()
os.chdir(path=PARENT_DIR)

DATA_DIR = PARENT_DIR/"data"
ORIGINAL_DATA_DIR = DATA_DIR/"originals"
SAVED_MODELS_DIR = PARENT_DIR/"saved_models"
TOKENS_DIR = PARENT_DIR/"tokens"
WORD_lEVEL_TOKENS_DIR = TOKENS_DIR/"WordLevel"
MBART_TOKENS_DIR = TOKENS_DIR/"MBart"


def make_fundamental_paths():

    for path in [
        DATA_DIR, ORIGINAL_DATA_DIR, TOKENS_DIR, SAVED_MODELS_DIR, WORD_lEVEL_TOKENS_DIR, MBART_TOKENS_DIR
    ]:
        if not Path(path).exists():
            os.mkdir(path)
        else:
            continue

if __name__ == "__main__":
    make_fundamental_paths()
