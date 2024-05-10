import os 
import json

import torch
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from sklearn.model_selection import KFold, train_test_split
from optuna import trial

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

        # These tokens have the same IDs in both the source language and English, so it
        # doesn't matter that I'm using the only source language files to get the IDs here.
        self.sos_id = torch.tensor(data=[self.source_tokens["[SOS]"]], dtype=int64)
        self.eos_id = torch.tensor(data=[self.source_tokens["[EOS]"]], dtype=int64)
        self.pad_id = torch.tensor(data=[self.source_tokens["[PAD]"]], dtype=int64)


    def _load_text(self, language: str) -> list[str]:
        """
        Access and read the text written in the specified language, or its
        English translation.

        Args:
            language [str]: the language whose text is to be loaded.

        Returns:
            list[str]: a list which contains the text in either the source 
            language or English.
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

        Args:
            dataset (list[str]): the file which contains our text.
            token_file_name (str): the name of the .json file to be created which 
                                   contains all the tokens and their IDs.

        Returns:
            dict: .json file that contains the tokens and their corresponding IDs
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

    def __init__(self, seq_length: int, data: BilingualData) -> None:
        self.seq_length = seq_length
        self.encoder_input_tokens = data.source_tokens.values()
        self.decoder_input_tokens = data.en_tokens.values()
        
        self.encoder_num_padding_tokens = self.seq_length - len(self.encoder_input_tokens) - 2
        self.decoder_num_padding_tokens = self.seq_length - len(self.decoder_input_tokens) - 1

        self.encoder_input = torch.cat(
            [
                data.sos_id,
                torch.tensor(self.encoder_input_tokens, dtype=torch.int64),
                data.eos_id,
                torch.tensor([data.pad_id] * self.encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        self.decoder_input = torch.cat(
            [
                data.sos_id,
                torch.tensor(self.decoder_input_tokens, dtype=torch.int64),
                torch.tensor([data.pad_id] * self.decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        self.label = torch.cat(
            [
                torch.tensor(self.decoder_input_tokens, dtype=torch.int64),
                data.eos_id, 
                torch.tensor([data.pad_id] * self.decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert self.encoder_input.size(0) == self.seq_length
        assert self.decoder_input.size(0) == self.seq_length
        assert self.label.size(0) == self.seq_length


    def _enough_tokens(self, seq_length: int) -> ValueError:
        """
        A checker that produces an error if the number of input tokens 
        into the encoder or decoder is too high.

        Args:
            seq_length (int): the maximum length of each sentence.

        Raises:
            ValueError: an error message that indicates an excessive 
                        number of input tokens into the encoder or
                        decoder.
        """

        if self.encoder_num_padding_tokens < 0 or self.decoder_num_padding_tokens < 0:

            raise ValueError("Sentence is too long")


    def __get_items(self) -> dict:
        """
        Return the encoder and decoder inputs, which are both of dimension 
        (seq_length, ), as well as the encoder and decoder masks.
        
        We also establish the encoder and decoder masks. The encoder mask 
        includes the elements of the encoder input that are not padding 
        tokens.

        The decoder mask is meant to ensure that each word in the decoder 
        only watches words that come before it. It does this by zeroing out
        the upper triangular part of a matrix.

        The masked encoder and decoder inuts are twice unsqueezed with respect 
        to the first dimension. Doing this adds sequence and batch dimensions
        to the tensors in the mask.

        Returns:
            dict: 
        """

        return {
            "label": self.label,
            "encoder_input": self.encoder_input,
            "decoder_input": self.decoder_input, 
            "encoder_mask": (self.encoder_input != self.pad_id).unsqueeze(dim=0).unsqueeze(dim=0).int(), 
            "decoder_mask": (self.decoder_input != self.pad_id).unsqueeze(dim=0).unsqueeze(dim=0).int() \
                            & self._causal_mask(size=self.decoder_input.size(0))
        }


    def _causal_mask(size: int) -> bool: 
        """
        Make a matrix of ones whose upper triangular part is full of zeros.

        Args:
            size (int): the second and third dimensions of the matrix.

        Returns:
            bool: return all the values above the diagonal, which should be the
                  upper triangular part.
        """
        
        mask = torch.triu(input=torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask == 0
