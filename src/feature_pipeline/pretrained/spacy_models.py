import subprocess
from loguru import logger
from src.feature_pipeline.data_sourcing import allow_full_language_names

def download_spacy_model(language: str):

    if language.lower() in ["en", "eng", "english"]:
        
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_news_sm"],
            check=True
        )

    else:
        # Check whether the language is among the available source languages
        source_lang = allow_full_language_names(source_lang=language)

        try:
            # Attempt to download a spacy model for that language
            subprocess.run(
                ["python", "-m", "spacy", "download", f"{source_lang}_core_news_sm"],
                check=True
            )

        except: 
            logger.error("There is no spacy model available for the requested source language.")   
