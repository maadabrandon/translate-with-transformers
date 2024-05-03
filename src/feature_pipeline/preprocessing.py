import os 
import torch.optim as optim 

from pathlib import Path

from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from torch.utils.data import DataLoader, Dataset, random_split

from src.setup.paths import DATA_DIR, TOKENIZERS_DIR
from src.feature_pipeline.data_sourcing import languages, allow_full_names


def get_tokenizer(path: Path = TOKENIZERS_DIR):
    return Tokenizer.from_file(path)


def build_tokenizer(dataset:list[str], language: str) -> Tokenizer:

    if len(os.listdir(path=TOKENIZERS_DIR)) == 0:
        # If an unknown word is encountered, the tokenizer will map it to the number which corresponds to "UNK"
        tokenizer = Tokenizer(
            model=WordLevel(unk_token="UNK")
        )

        # Choose the whitespace pretokenizer 
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)
        tokenizer.save(path=f"{DATA_DIR}/tokenizers")

    else: 
        get_tokenizer(path=TOKENIZERS_DIR)


class DataSet():

    def __init__(self, source_lang: str):

        allow_full_names(source_lang=source_lang)

        self.folder_path = DATA_DIR/f"{source_lang}-en"
        self.en_file_name = f"europarl-v7.{source_lang}-en.en"
        self.source_lang_file_name = f"europarl-v7.{source_lang}-en.{source_lang}"

        self.en_text = self._load_text(language="en")
        self.source_text = self._load_text(language=source_lang)

        self.source_tokenizer = build_tokenizer(dataset=self.source_text, language=source_lang)
        self.en_tokenizer = build_tokenizer(dataset=self.en_text, language="en")

    def _load_text(self, language: str) -> list[str]:
        """
        Access and read the text written in the specified language, or its
        English translation..

        Returns:
            list[str]: a list containing the version of the text in either
            the source language or English.
        """
        match language:

            case version if version in ["en", "english", "English"]:
                with open(self.folder_path/self.en_file_name) as file:
                    lines = file.readlines()

            case source_lang:
                with open(self.folder_path/self.source_lang_file_name) as file:
                    lines = file.readlines()

        return lines 
