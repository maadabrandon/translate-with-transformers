import os 
import spacy
import json
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from spacy.util import Doc
from transformers import MBart50Tokenizer

from torchtext.data import Field
from torchtext.datasets import TranslationDataset
from torch.utils.data import Dataset, DataLoader

from src.feature_pipeline.data_sourcing import allow_full_language_names
from src.feature_pipeline.pretrained.spacy_models import download_spacy_model, get_model_name
from src.setup.paths import DATA_DIR, ORIGINAL_DATA_DIR, WORD_lEVEL_TOKENS_DIR, MBART_TOKENS_DIR, SPACY_OBJECTS_DIR

class BilingualData(Dataset):

    def __init__(self, source_lang: str, tokenizer_name: str) -> None:
        
        self.tokenizer_name = tokenizer_name

        # Make it so that full source language names are also accepted
        self.source_lang = allow_full_language_names(language=source_lang)

        # The directories where the data will be saved
        self.data_path = ORIGINAL_DATA_DIR/f"{self.source_lang}-en"

        # Make the directories to the tokens if they don't already exist
        self.tokens_path = self._set_tokens_dir()

        # The paths to the tokens
        self.source_tokens_path = self.tokens_path/f"{self.source_lang}_tokens.json"
        self.en_tokens_path = self.tokens_path/"en_tokens.json"

        # The names of the files in the dataset
        self.en_text_name = f"europarl-v7.{self.source_lang}-en.en"
        self.source_text_name = f"europarl-v7.{self.source_lang}-en.{self.source_lang}"

        # The raw datasets themselves
        self.en_text = self._load_text(language="en")
        self.source_text = self._load_text(language=self.source_lang)

        # Tokenize the text in the source language
        self.source_lang_field = Field(
            tokenize=self._tokenize(text=self.source_text, language=self.source_lang),
            init_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            lower=True
        )   

        # Tokenize the English text
        self.en_field = Field(
            tokenize=self._tokenize(text=self.en_text, language="en"),
            init_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            lower=True
        )

        self.translation_data = TranslationDataset(
            path=f"{self.data_path}/",
            exts=(self.source_text_name, self.en_text_name),
            fields=(self.source_lang_field, self.en_field)
        )

        self.source_lang_field.build_vocab(self.translation_data, min_freq=2)
        self.en_field.build_vocab(self.translation_data, min_freq=2)

        self.source_lang_vocab = self.source_lang_field.vocab
        self.en_vocab = self.en_field.vocab

        self.source_lang_vocab_size = len(self.source_lang_vocab)
        self.en_vocab_size = len(self.en_vocab)


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
            with open(self.data_path/self.en_text_name) as file:
                lines = file.readlines()

        elif language is self.source_lang:
            with open(self.data_path/self.source_text_name) as file:
                lines = file.readlines()

        return lines  


    def _set_tokens_dir(self) -> Path:
        """
        Create the directory houses the json files that contain the tokens and IDs for each source 
        language/English combination, and return its path.
        
        Returns:
            Path: the path to the tokens in question.
        """
        tokenizers_and_paths = {
            "wordlevel" : WORD_lEVEL_TOKENS_DIR/f"{self.source_lang}-en",
            "mbart" : MBART_TOKENS_DIR/f"{self.source_lang}-en" 
        }

        if self.tokenizer_name.lower() in tokenizers_and_paths.keys():
            tokens_path = tokenizers_and_paths[self.tokenizer_name.lower()]

            if not Path(tokens_path).exists():
                os.mkdir(tokens_path)
                
            return tokens_path


    def _tokenize(self, text: list[str], language: str) -> dict:
        """
        Does one of the following:
        -    Use either HuggingFace's word level tokenizer to tokenize the text file, and save 
             the tokens, and their IDs.
        -    Use the MBart50Tokenizer to tokenize the sentences in the text file, and save 

        Args:
            text (list[str]): the file which contains our text.

            language (str): the language in which the text to be tokenized is written.
                            This argument only kicks in if MBart's tokenizer is being used.

        Returns:
            dict: .json file that contains the tokens and their corresponding IDs
        """
        token_file_name = f"{language}_tokens.json"
        if not Path(self.tokens_path/token_file_name).exists():

            logger.info(f"Initialising the {self.tokenizer_name} tokenizer")
            
            if "wordlevel" in self.tokenizer_name.lower():

                # If an unknown word is encountered, the tokenizer call it "UNK", and tokenize it.
                tokenizer = Tokenizer(
                    model=WordLevel(unk_token="<UNK>")
                )   
            
                # Choose the whitespace pre-tokenizer 
                tokenizer.pre_tokenizer = Whitespace()
                trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"], min_frequency=2)

                # Perform tokenization
                tokenizer.train_from_iterator(iterator=text, trainer=trainer)

                # Save the tokens
                tokenizer.save(path=f"{self.tokens_path}/{token_file_name}")

            elif "mbart" in self.tokenizer_name.lower():

                tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
                tokenizer.add_special_tokens(
                    special_tokens_dict={
                        "unk_token": "<UNK>",
                        "bos_token": "<BOS>", 
                        "eos_token": "<EOS>",
                        "pad_token": "<PAD>"
                    }
                )

                sentences = self._segment_sentences(text=text, language=language)

                logger.info("Tokenizing segmented text...")
                encoded_text = tokenizer(text=sentences, padding=True, truncation=True)

                tokenized_sentences = []
                for i, sentence in enumerate(sentences):
                    tokenized_sentences.append(
                        {
                            "input_sentence": sentence,
                            "tokens": tokenizer.convert_ids_to_tokens(encoded_text["input_ids"][i]),
                            "ids": encoded_text["input_ids"][i]
                        }
                    )

                logger.info("Writing tokens and IDs to a json file")
                with open(f"{self.tokens_path}/{token_file_name}", mode="w") as file:
                    json.dump(tokenized_sentences, fp=file)

            else:
                raise NotImplementedError("Requested tokenizer has not been implemented")

        else:
            logger.success(f"The {language} text has already been tokenized")


    def _segment_sentences(self, text: list[str], language: str) -> list[str]:
        """
        Load a pre-trained spacy model that will be used to iterate over the sentences
        in the text, and return a list contatining all these sentences.

        Args:
            text (list[str]): the raw text to be segmented into sentences
            language (str): the language whose pretrained model will have to be
                            loaded.

        Returns:
            list[str]: a list containing the sentences
        """
        # Download spacy model if necessary
        model_name = get_model_name(language=language)

        # Download spacy model if necessary
        if not spacy.util.is_package(name=model_name):
            download_spacy_model(source_lang=language)

        # Load spacy model
        spacy_model = spacy.load(name=model_name)
        text_string = self._make_text_into_one_string(text=text)

        def __process_text_in_chunks(text_string: str, chunk_size: int, save: bool = True) -> list[Doc]:
            """
            Use the spacy model to process chunks of the text, and return the processed file as a list
            of the spacy Doc files (produced by the processing task). This output will be saved if 
            requested.

            Args:
                text_string (str): the string which contains the full text to be processed by spacy
                chunk_size (int): the number of characters to be processed at a time
                save (bool): whether to save the list of spacy Doc files that will be generated

            Returns:
                list[Doc]: a list of Doc files produced by the processing
            """
            chunks = [text_string[i: i+chunk_size] for i in range(0, len(text_string), chunk_size)]
            logger.info(f"Using spacy to process the {language} text...")
            processed_chunks = [spacy_model(chunk) for chunk in tqdm(chunks)]

            if save:
                __save_processed_chunks(
                    file_name=f"processed_{language}_text.spacy",
                    processed_chunks=processed_chunks
                )

            return processed_chunks

        def __save_processed_chunks(file_name: str, processed_chunks: list[Doc]):
            """
            Use spacy's default serialization tools to save the list of Doc files that spacy 
            produced during the processing stage.

            Args:
                file_name (str): the intended name of the file to be saved.

                processed_chunks (list[Doc]): the list of Doc files that results from spacy 
                                              processing the 
            """
            # Serialize the doc objects in the processed chunks
            serialized_chunks = [doc_chunk.to_bytes() for doc_chunk in processed_chunks]

            # Convert each doc file 
            with open(SPACY_OBJECTS_DIR/file_name, mode="wb") as file:
                for doc_chunk in processed_chunks:
                    file.write(doc_chunk.to_bytes())
                    file.write(b'\n\n') # Use double lines to separate doc objects

        def __segment_sentences_in_the_chunks(processed_chunks: list[Doc]) -> list:
            sentences = []
            for chunk in processed_chunks:
                sentences.extend(
                    [sentence.text for sentence in chunk.sents]
                )
            return sentences 

        # Use spacy loader to process the text
        processed_text = __process_text_in_chunks(
            text_string=text_string, 
            chunk_size=spacy_model.max_length,
            save=True
        )
        
        # Segment string into sentences 
        sentences = __segment_sentences_in_the_chunks(processed_chunks=processed_text)
        return sentences

    
    def _make_text_into_one_string(self, text: list[str]):

        text_string = ""
        text_pieces = tqdm(text)
        text_name = self.source_text_name if text is self.source_text else self.en_text_name
        text_pieces.set_description(desc=f"Putting {text_name} into a single string")

        for piece in text_pieces:
            text_string += piece
        return text_string


    def _retrieve_tokens(self) -> tuple[dict, dict]|list :

        with open(self.source_tokens_path, mode="r") as file1, open(self.en_tokens_path, mode="r") as file2:
            source_tokens = json.load(file1)
            en_tokens = json.load(file2)
        
        if "word_level" in self.tokenizer_name.lower():
            return source_tokens["model"]["vocab"], en_tokens["model"]["vocab"]

        elif "mbart" in self.tokenizer_name.lower():
            # Returning the contents of the files for now. I won't be sure whether ny parsing logic will be 
            # necessary until I've seen how the MBart model works
            return source_tokens, en_tokens
            

class DataSplit():

    def __init__(self, source_lang: str, train_size: float, val_size: float) -> None:
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - self.train_size - self.val_size

        self.source_lang = allow_full_language_names(source_lang=source_lang)
        self.data = BilingualData(source_lang=self.source_lang)

        self.source_train_size = int(len(self.data.source_text) * self.train_size)
        self.source_val_size = int(len(self.data.source_text) * self.val_size)
        self.source_test_size = int(len(self.data.source_text) * self.test_size)

        self.en_train_size = int(len(self.data.en_text) * self.train_size)
        self.en_val_size = int(len(self.data.en_text) * self.val_size)
        self.en_test_size = int(len(self.data.en_text) * self.test_size)

        self.source_train_indices = list(range(self.source_train_size))
        self.source_val_indices = list(range(self.source_val_size))
        self.source_test_indices = list(range(self.source_test_size))

        self.en_train_indices = list(range(self.en_train_size))
        self.en_val_indices = list(range(self.en_val_size))
        self.en_test_indices = list(range(self.en_test_size))

        self.path_to_split_data = self._make_path_to_split_data()
        self.num_splits_made = len(os.listdir(self.path_to_split_data))


    def _make_path_to_split_data(self) -> str:

        split_data_path = DATA_DIR/"split_data"/f"{self.source_lang}-en"

        if not Path(split_data_path).exists():
            os.mkdir(split_data_path)

        return split_data_path


    def _split_data(self):
        """
        We split the source and target datasets using the pre-defined proportions 
        of the training, validation and test sets, and then save these split files.
        """
        logger.info("Creating splits for each language...")

        self.source_train_data = [self.data.source_text[i] for i in self.source_train_indices]
        self.en_train_data = [self.data.en_text[i] for i in self.en_train_indices]

        self.source_val_data = [self.data.source_text[i] for i in self.source_val_indices]
        self.en_val_data = [self.data.en_text[i] for i in self.en_val_indices]
        
        self.source_test_data = [self.data.source_text[i] for i in self.source_test_indices]
        self.en_test_data = [self.data.en_text[i] for i in self.en_test_indices]

        self._save_all_split_data()


    def _save_all_split_data(self):
        """
        Save the training, validation, and test data for the source and 
        target languages.
        """
        options = {
            ("train", self.source_lang): self.source_train_data,
            ("val", self.source_lang): self.source_val_data,
            ("test", self.source_lang): self.source_test_data,
            ("train", "en"): self.en_train_data,
            ("val", "en"): self.en_val_data,
            ("test", "en"): self.en_test_data,
        }

        for option in tqdm(options.keys()):
            
            if self.source_lang in option:
                logger.info(f"Saving raw {option[0]}ing data for the {self.source_lang} version")
            else:
                logger.info(f"Saving raw {option[0]} data for the en version")
    
            self._save_file_as_txt(
                data_split=options[option],
                file_name=f"{option[0]}_europarl-v7.{self.source_lang}-en.{option[1]}"
            )   


    def _save_file_as_txt(self, data_split: list[str], file_name: str):
        
        if f"{file_name}" not in os.listdir(self.path_to_split_data):
            logger.info("Saving split data files for data loader creation...")

            with open(self.path_to_split_data/file_name, mode="w", encoding="utf-8") as text:
                split_pieces = tqdm(data_split)

                for strings in split_pieces:
                    split_pieces.set_description("Writing the strings to the file...")
                    text.write(strings)
        else:
            logger.success(f"{file_name} has already been saved")
    
    
    def _make_translation_dataset(self) -> tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
        """
        Create the translation dataset object by fetching the saved training, validation, and testing data
        """
        return TranslationDataset.splits(
            fields=(self.data.source_lang_field, self.data.en_field),
            exts=(self.data.source_text_name, self.data.en_text_name),
            path=self.path_to_split_data,
            train="train_",
            validation="val_",
            test="test_" 
        )


    def _make_data_loaders(self, batch_size: int, split: str) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split the data into training, validation, testing sets, generate their corresponding data loaders,
        and return them.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: the training, validation, and testing dataloaders.
            split   (str): whether the training, validation, or test dataloader is to be produced.
        """
        # Split the data, and save the splits as separate text files
        self._split_data()

        # Get the training, validation, and test data as objects of Pytorch's Dataset class
        train_data, val_data, test_data = self._make_translation_dataset()

        # Using a batch size of 20 for now
        train_dataloader =  DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        # I am using yield statements here because returning all three dataloader results in an error 
        # when this function is called to get the training dataloader during the training loop.
        if "train" in split:
            yield train_dataloader

        elif "val" in split:
            yield val_dataloader

        elif "test" in split:
            yield test_dataloader


if __name__ == "__main__":
    BilingualData(source_lang="de", tokenizer_name="mbart")
