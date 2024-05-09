import os 
from pathlib import Path 

PARENT_DIR = Path("_file_").parent.resolve()

DATA_DIR = PARENT_DIR/"data"
SAVED_MODELS_DIR = PARENT_DIR/"saved_models"
TOKENS_DIR = PARENT_DIR/"tokens"


def make_fundamental_paths():

    for path in [DATA_DIR, TOKENS_DIR, SAVED_MODELS_DIR]:
        if not Path(path).exists():
            os.mkdir(path)
        else:
            continue


def make_tokenizer_path(source_lang: str):
    """
    Create the directories where the individualtokenizers are going to be stored.
    """
    from src.feature_pipeline.data_sourcing import languages

    if source_lang.lower() in languages.keys() or languages.values():

        if not Path(TOKENS_DIR/f"{source_lang}-en").exists():
            os.mkdir(TOKENS_DIR/f"{source_lang}-en")
        

if __name__ == "__main__":

    make_fundamental_paths()
    make_tokenizer_paths()