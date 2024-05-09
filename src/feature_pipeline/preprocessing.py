import os 
import json

import torch

from pathlib import Path

from loguru import logger
from tqdm import tqdm
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
#from torch.utils.data import DataLoader, Dataset, random_split

from src.feature_pipeline.data_sourcing import languages
from src.setup.paths import DATA_DIR, TOKENS_DIR, make_tokenizer_path
from src.feature_pipeline.data_sourcing import languages, allow_full_language_names, download_data


class BilingualData():

    def __init__(self, source_lang: str) -> None:
        
        # Make it so that full source language names are also accepted
        self.source_lang = allow_full_language_names(source_lang=source_lang)

        # Make paths for the tokenizers if they don't already exist
        make_tokenizer_path(source_lang=self.source_lang)

        # The directories where the data and its tokenizers will be saved
        self.folder_path = DATA_DIR/f"{self.source_lang}-en"
        self.tokens_path = TOKENS_DIR/f"{self.source_lang}-en"

        # The names of the files in the dataset
        self.en_text_name = f"europarl-v7.{self.source_lang}-en.en"
        self.source_text_name = f"europarl-v7.{self.source_lang}-en.{self.source_lang}"

        # The raw datasets themselves
        self.en_text = self._load_text(language="en")   
        self.source_text = self._load_text(language=self.source_lang)

        # Tokenize the English and {source language} texts
        self.en_tokens = self._get_tokens(dataset=self.en_text, token_file_name="en_tokens.json")
        self.source_tokens = self._get_tokens(dataset=self.source_text, token_file_name=f"{self.source_lang}_tokens.json")

        self.sos_id = torch.tensor(data=[self.source_tokens["[SOS]"]], dtype=int64)
        self.eos_id = torch.tensor(data=[self.source_tokens["[EOS]"]], dtype=int64)
        self.pad_token = torch.tensor(data=[self.source_tokens["[PAD]"]], dtype=int64)


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
        if language in ["en", "english", "English"]:
            with open(self.folder_path/self.en_text_name) as file:
                lines = file.readlines()

        elif language is self.source_lang:
            with open(self.folder_path/self.source_text_name) as file:
                lines = file.readlines()

        return lines  


    def _tokenize(self, dataset:list[str], token_file_name: str) -> dict:
        """
        Use HuggingFace's word level tokenizer to tokenize the text file, and save 
        the tokens. 
        """

        # If an unknown word is encountered, the tokenizer will map it to the number which corresponds to "UNK"
        tokenizer = Tokenizer(
            model=WordLevel(unk_token="UNK")
        )   
    
        # Choose the whitespace pre-tokenizer 
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # Perform tokenization
        tokenizer.train_from_iterator(iterator=dataset, trainer=trainer)

        # Save the tokens
        tokenizer.save(path=f"{self.tokens_path}/{token_file_name}.json")

        # Get the tokens    
        self._get_tokens(dataset=dataset, token_file_name=token_file_name)


    def _get_tokens(self, dataset: list[str], token_file_name: str) -> dict:
        """
        Tokenize the file, or retrieve the tokens if the tokens already exist
        in their appropriate folder.

        Returns:
            dict : the tokens and their respective IDs
        """

        if not Path(self.tokens_path/token_file_name).exists():
            self._tokenize(dataset=dataset, token_file_name=token_file_name)

        elif Path(self.tokens_path/token_file_name).exists():

            # Get the tokens if it is already present
            with open(self.tokens_path/token_file_name, mode="r") as file:
                tokens = json.load(file)

            return tokens["model"]["vocab"]
        
    
class TransformerInputs():

    def __init__(self, seq_length: int, data: BilingualData):

        self.seq_length = seq_lengths
        self.encoder_input_tokens = data.source_tokens.values()
        self.decoder_input_tokens = data.en_tokens.values()
        
        self.encoder_num_padding_tokens = self.seq_length - len(self.encoder_input_tokens) - 2
        self.decoder_num_padding_tokens = self.seq_length - len(self.decoder_input_tokens) - 1

        self.encoder_input = torch.cat(
            [
                data.sos_id,
                torch.tensor(self.encoder_input_tokens, dtype=torch.int64),
                data.eos_id,
                torch.tensor([data.pad_token] * self.encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        self.decoder_input = torch.cat(
            [
                data.sos_id,
                torch.tensor(self.decoder_input_tokens, dtype=torch.int64),
                data.en_tokens["[EOS]"],
                torch.tensor([data.pad_token] * self.decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )


    def _enough_tokens(self, seq_length: int):

        if self.encoder_num_padding_tokens < 0 or self.decoder_num_padding_tokens < 0:

            raise ValueError("Sentence is too long")