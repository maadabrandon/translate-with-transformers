import os 
from pathlib import Path 

PARENT_DIR = Path("_file_").parent.resolve()
os.chdir(path=PARENT_DIR)

DATA_DIR = PARENT_DIR/"data"
ORIGINAL_DATA_DIR = DATA_DIR/"originals"
MODELS_DIR = PARENT_DIR/"models"
PRETRAINED_DIR = MODELS_DIR/"pre-trained"

TOKENS_DIR = PARENT_DIR/"tokens"
SPACY_OBJECTS_DIR = PARENT_DIR/"spacy_objects"
DOC_BINS_DIR = SPACY_OBJECTS_DIR/"doc_bins"
WORD_lEVEL_TOKENS_DIR = TOKENS_DIR/"WordLevel"
MBART_TOKENS_DIR = TOKENS_DIR/"MBart"


def make_fundamental_paths():

    for path in [
        DATA_DIR, ORIGINAL_DATA_DIR, TOKENS_DIR, MODELS_DIR, PRETRAINED_DIR, WORD_lEVEL_TOKENS_DIR, 
        MBART_TOKENS_DIR, SPACY_OBJECTS_DIR, DOC_BINS_DIR
    ]:
        if not Path(path).exists():
            os.mkdir(path)
        else:
            continue

if __name__ == "__main__":
    make_fundamental_paths()
