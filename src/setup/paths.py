import os 
from pathlib import Path 

PARENT_DIR = Path("_file_").parent.resolve().parent

DATA_DIR = PARENT_DIR/"data"
MODELS_DIR = PARENT_DIR/"models"
TOKENIZERS_DIR = PARENT_DIR/"tokenizers"

for path in [DATA_DIR, TOKENIZERS_DIR, MODELS_DIR]:
    if not Path(path).exists():
        os.mkdir(path)
    else:
        continue