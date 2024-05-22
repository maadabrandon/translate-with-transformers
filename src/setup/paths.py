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


def make_path_to_tokens(source_lang: str, path: Path):
    """
    Create the directories where the individualtokenizers are going to be stored.
    """
    from src.feature_pipeline.data_sourcing import source_languages

    if source_lang.lower() in source_languages.keys() or source_languages.values():
        if not Path(path/f"{source_lang}-en").exists():
            os.mkdir(path/f"{source_lang}-en")
        

if __name__ == "__main__":
    make_fundamental_paths()
