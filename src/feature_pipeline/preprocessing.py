import os 
import torch.optim as optim 

from pathlib import Path

from loguru import logger
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from torch.utils.data import DataLoader, Dataset, random_split

from src.setup.paths import DATA_DIR, TOKENIZERS_DIR, make_tokenizer_paths
from src.feature_pipeline.data_sourcing import languages, allow_full_names


class DataSet():

    def __init__(self, source_lang: str, en_tokenizer_name: str, source_tokenizer_name: str):

        make_tokenizer_paths()
        allow_full_names(source_lang=source_lang)

        self.folder_path = DATA_DIR/f"{source_lang}-en"
        self.tokenizers_path = TOKENIZERS_DIR/f"{source_lang}-en"
        self.num_tokenizers = len(os.listdir(path=self.tokenizers_path))

        self.en_file_name = f"europarl-v7.{source_lang}-en.en"
        self.source_lang_file_name = f"europarl-v7.{source_lang}-en.{source_lang}"

        self.en_text = self._load_text(language="en")   
        self.source_text = self._load_text(language=source_lang)

        self.en_tokenizer = self._tokenize(dataset=self.en_text, tokenizer_name=en_tokenizer_name)
        self.source_tokenizer = self._tokenize(dataset=self.source_text, tokenizer_name=source_tokenizer_name)


    def _load_text(self, language: str) -> list[str]:
        """
        Access and read the text written in the specified language, or its
        English translation.

        Args:
            language [str]: the language whose text is to be loaded.

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


    def _tokenize(self, dataset:list[str], tokenizer_name: str) -> Tokenizer:

        def __build_tokenizer(self):
            """
            Use HuggingFace's word level tokenizer to tokenize a text file,
            and save that tokenizer. 
            """

            # If an unknown word is encountered, the tokenizer will map it to the number which corresponds to "UNK"
            tokenizer = Tokenizer(
                model=WordLevel(unk_token="UNK")
            )   
        
            # Choose the whitespace pretokenizer 
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

            # Perform tokenization
            logger.info("Tokenizing the text file")
            tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)

            tokenizer.save(path=f"{self.tokenizers_path}/{tokenizer_name}")
            logger.success(f"Saved {tokenizer_name}")

        def __get_tokenizer(self, path: str = self.tokenizers_path) -> Tokenizer:
            """
            Get a tokenizer if it has already been saved.

            Args:
                path (str, optional): the path where the tokenizer lives. 
                                    Defaults to f"{TOKENIZERS_DIR}".
            Returns:
                Tokenizer: a downloaded Tokenizer object
            """
            return Tokenizer.from_file(path)

        # Build the tokenizer if it does not already exist
        if not os.path.exists(path=f"{self.tokenizers_path/tokenizer_name}"):

            __build_tokenizer(self)
            logger.success(f"Built the {tokenizer_name}")

        # Get the tokenizer if it is already present
        else:
            __get_tokenizer(self, path=f"{self.tokenizers_path}/{tokenizer_name}")  
            logger.success(f"Fetched the saved {tokenizer_name}")