import os 
from pathlib import Path 


PARENT_DIR = Path("_file_").parent.resolve().parent

DATA_DIR = PARENT_DIR/"data"
MODELS_DIR = PARENT_DIR/"models"
TOKENIZERS_DIR = PARENT_DIR/"tokenizers"


def make_fundamental_paths():

    for path in [DATA_DIR, TOKENIZERS_DIR, MODELS_DIR]:
        if not Path(path).exists():
            os.mkdir(path)
        else:
            continue


def make_tokenizer_paths():
    
    from src.feature_pipeline.data_sourcing import languages

    for lang in languages.values():

        if not Path(TOKENIZERS_DIR/f"{lang}-en").exists():
            os.mkdir(TOKENIZERS_DIR/f"{lang}-en")
        else:
            continue
        

if __name__ == "__main__":

    make_fundamental_paths()
    make_tokenizer_paths()