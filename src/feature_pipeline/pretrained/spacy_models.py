import subprocess
from loguru import logger
from src.feature_pipeline.data_sourcing import allow_full_language_names


def get_model_name(language: str) -> str:
    model_name = "en_core_web_sm" if language == "en" else f"{language}_core_news_sm"
    return model_name


def download_spacy_model(language: str) -> None:

    # Check whether the language is among the available source languages
    language = allow_full_language_names(language=language)
    model_name = get_model_name(language=language)

    try:
        # Attempt to download a spacy model for that language
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name],
            check=True
        )

    except: 
        logger.error("There is no spacy model available for the requested source language.")   
        