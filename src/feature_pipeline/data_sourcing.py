"""
This module contains the code for fetching data. It contains 
more functions than I had intended to write because of the 
need to take care of the possibility of things like missing data.
"""

import os
import tarfile
import requests

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from src.setup.paths import ORIGINAL_DATA_DIR, make_fundamental_paths

# The languages for which data is available
source_languages = {
    "bulgarian": "bg",
    "czech": "cs",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "spanish": "es",
    "estonian": "et",
    "finnish": "fi",
    "italian": "it",
    "lithuanian": "lt",
    "latvian": "lv",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "slovak": "sk",
    "slovenian": "sl",
    "slovene": "sl",
    "swedish": "sv"
}


class FileDownload:
    """
    A small class do encapsulate the use of the tqdm library to make progress bars
    for downloads (I like visuals).
    """
    def __init__(self, url: str, target_path: Path, chunk_size: int = 1024):
        """
        Get the HTTP response, requesting the value of the "content-length" key of the HTTP header,
        specifically asking that the zero be returned if there is no value.

        :param url: the url where the desired object is.
        :param chunk_size: the
        """
        self.response = requests.get(url)
        self.target_path = target_path
        self.chunk_size = chunk_size  # the size of each chunk in bytes
        self.total_size = int(self.response.headers.get("content-length", 0))
        self.bar = tqdm(total=self.total_size, unit="MiB", unit_scale=True, desc="Downloading...")

    def write(self):
        with open(self.target_path, mode="wb") as file:
            for piece in self.response.iter_content(chunk_size=self.chunk_size):
                self.bar.update(len(piece))
                file.write(piece)
        self.bar.close()
        logger.success("Downloaded!")


def download_data(source_lang: str, keep_tarball: bool | None = True):
    """
    Check whether the folder/tarball containing the data already exists, and take
    appropriate action as the case may be. If neither of them exists, the function will
    fetch the tarball containing the requested data and extract it into a folder with the
    same name.
    
    Args:
        source_lang: The "source_lang" can be one of two things. It must either be the
                     full name of the source language, or a two letter abbreviation of 
                     that name. For example, for French, Spanish, Italian, and Greek, 
                     we will be using "FR", "ES", "IT", and "EL" respectively.
                     Capitalisation of names or abbreviations is allowed.

        keep_tarball: a boolean that allows us to specify whether we want to
                      delete the tarball (the initial download) after extraction.
    """

    folder_name = f"{source_lang.lower()}-en"
    archive_name = f"{folder_name}.tgz"

    tarball_path = ORIGINAL_DATA_DIR / archive_name
    destination_path = ORIGINAL_DATA_DIR / folder_name

    if available_language(source_lang=source_lang):
        source_lang = allow_full_language_names(lang=source_lang)

        logger.info("Checking for the presence of folders and previously downloaded tarballs")
        if data_folder_exists(source_lang=source_lang) and tarball_exists(source_lang=source_lang):
            logger.info("Checking for missing files...")

            if missing_data_files(path=destination_path, source_lang=source_lang):
                extract_missing_files(
                    tarball_path=tarball_path,
                    destination_path=destination_path,
                    source_lang=source_lang
                )

            elif both_data_files_exist(source_lang=source_lang, path=destination_path):
                logger.success("Both the source language and English files are present")

            # Delete tarball if demanded
            if not keep_tarball:
                os.remove(path=tarball_path)

            logger.success(f"The folder {folder_name} already exists")

        elif data_folder_exists(source_lang=source_lang) and not tarball_exists(source_lang=source_lang):
            logger.success(f"The folder {folder_name} already exists")

            logger.info("Checking for missing files...")
            if missing_data_files(path=destination_path, source_lang=source_lang):

                # Get the tarball, and extract the missing file(s) from it
                get_tarball(
                    source_lang=source_lang,
                    archive_name=archive_name,
                    destination_path=destination_path,
                    tarball_path=tarball_path,
                    keep_tarball=False
                )

            elif both_data_files_exist(source_lang=source_lang, path=destination_path):
                logger.success("Both data files are present. No need for any downloads")

            else:
                logger.warning("There are more files in the folder than there should be. Please investigate")

        elif not data_folder_exists(source_lang=source_lang) and tarball_exists(source_lang=source_lang):
            # The folder does not exist, but the tarball does (and needs to be extracted)
            fully_extract_tarball(
                archive_path=tarball_path,
                destination_path=destination_path,
                keep_tarball=keep_tarball
            )

        # The data needs to be downloaded from scratch
        else:
            get_tarball(
                source_lang=source_lang,
                archive_name=archive_name,
                destination_path=destination_path,
                tarball_path=tarball_path,
                keep_tarball=keep_tarball
            )
    else:
        raise Exception("Data in the requested source language is unavailable")


def get_tarball(
        source_lang: str,
        archive_name: str,
        destination_path: Path,
        tarball_path: Path,
        keep_tarball: bool
):
    """
    We download the tarball. If the destination directory doesn't exist, it will be created 
    before all the contents of the tarball are extracted into it.

    If the directory already exists, then the extraction will depend on whether
    there are any missing files in the folder or not. If there are, those missing files
    alone will be extracted into the folder.

    Args:
        source_lang (str): the language from which we will be translating to English
        archive_name (str): the name of the tarball to be downloaded
        destination_path (Path): the directory into which the contents of the tarball will be extracted
        tarball_path (Path): the directory where the tarball resides
        keep_tarball (bool): whether to keep the tarball after extracting the contents

    Raises:
        Exception: an exception would be raised in one of the following two situations:
                  - the URL is unavailable
                  - the download could not be completed due to an exception that occurred
                    during the handling of the HTTP request
    """
    url = f"https://www.statmt.org/europarl/v7/{archive_name}"

    try:
        logger.info(f"Downloading the tarball for {source_lang}-en ...")
        download = FileDownload(url=url, target_path=tarball_path)

        if download.response.status_code == 200:
            # Save the download with an accompanying progress bar
            download.write()

            # Provide the final destination for the extracted files, since it doesn't exist
            if not Path(destination_path).exists():
                os.mkdir(destination_path)

                logger.info("Extracting the contents of the tarball...")
                fully_extract_tarball(
                    archive_path=tarball_path,
                    destination_path=destination_path,
                    keep_tarball=keep_tarball
                )

            else:
                logger.success("Destination folder exists")

                if missing_data_files(path=destination_path, source_lang=source_lang):
                    logger.info("Extracting the missing file...")
                    extract_missing_files(
                        tarball_path=tarball_path,
                        destination_path=destination_path,
                        source_lang=source_lang
                    )

                logger.success("Done!")
        else:
            raise Exception(f"{url} is not available")

    except requests.RequestException as issue:
        logger.error(issue)
        raise


def available_language(source_lang: str) -> bool:
    """
    Check whether the requested source language (again full names and abbreviations are both
    acceptable) is among the languages for which we have data.

    Returns:
        bool: whether the language is available.
    """
    if source_lang.lower() not in source_languages.keys() and source_lang.lower() not in source_languages.values():
        return False
    else:
        return True


def data_folder_exists(source_lang: str) -> bool:
    """
    Checks for the existence of a local data folder for the language.

    Returns:
        bool: whether it exists or not
    """
    folder_name = f"{source_lang.lower()}-en"

    return True if Path(ORIGINAL_DATA_DIR / folder_name).exists() else False


def tarball_exists(source_lang: str) -> bool:
    """
    Checks for the existence of a downloaded tarball for the language.

    Returns:
        bool: whether it exists or not
    """
    archive_name = f"{source_lang.lower()}-en.tgz"

    return True if Path(ORIGINAL_DATA_DIR / archive_name).exists() else False


def fully_extract_tarball(archive_path: Path, destination_path: Path, keep_tarball: bool = True):
    """
    Extract the downloaded tarball to the named path, and delete it if instructed.

    Args:
        archive_path (Path): the path of the tarball
        destination_path (Path): the path of the destination folder
        keep_tarball (bool, optional): whether the tarball should be deleted after extraction
    """
    # Extract the files
    with tarfile.open(archive_path) as archive:
        archive.extractall(path=destination_path)
        archive.close()

    if not keep_tarball:
        logger.info("Deleting the tarball...")
        os.remove(path=archive_path)


def missing_data_files(path: Path, source_lang: str) -> bool:
    """
    Looks into a given data folder, and determines whether:
        - both data files exist
        - whether there is at least one missing file
        - whether there are more files than there should be 

    Returns:
        bool: True if there are missing files.
              False if there are no missing files, and if 
              there are more files than there should be.
    """
    num_files = len(os.listdir(path=path))

    if both_data_files_exist(path=path, source_lang=source_lang):
        return False
    elif num_files < 2:
        return True
    else:
        logger.warning("There are more files in the folder than there should be. Please investigate")
        return False


def both_data_files_exist(path: Path, source_lang: str) -> bool:
    """
    Check whether both of the data files are in the folder.

    Returns:
        bool: True if both exist
    """

    contents = os.listdir(path=path)
    en_file_name = f"europarl-v7.{source_lang}-en.en"
    source_lang_file_name = f"europarl-v7.{source_lang}-en.{source_lang}"

    if source_lang_file_name in contents and en_file_name in contents:
        logger.success("All data files intact")
        return True


def get_missing_file_names(path: Path, source_lang: str) -> str | tuple | None:
    """
    Each data folder contains two files: an english file, and a file in the source
    language. The function checks whether any of these files is in the relevant data 
    folder.

    Args:
        path (Path): the path of the data folder being searched
        source_lang (str): the language from which we will be translating

    Returns:
        str: the name of the missing file
        tuple: the names of both files in case they are missing 
    """
    contents = os.listdir(path=path)
    en_file_name = f"europarl-v7.{source_lang}-en.en"
    source_lang_file_name = f"europarl-v7.{source_lang}-en.{source_lang}"

    if source_lang_file_name in contents and en_file_name in contents:
        logger.success("There are no missing files")

    elif source_lang_file_name in contents and en_file_name not in contents:
        logger.warning(f"{en_file_name} is missing")
        return en_file_name

    elif source_lang_file_name not in contents and en_file_name in contents:
        logger.warning(f"{source_lang_file_name} is missing")
        return source_lang_file_name

    elif source_lang_file_name not in contents and en_file_name not in contents:
        return source_lang_file_name, en_file_name


def extract_missing_files(tarball_path: Path, destination_path: Path, source_lang: str):
    """
    Get the name of the missing file in a given folder, and extract those files from
    a tarball.

    Args:
        tarball_path (Path): the path where the tarball resides 
        destination_path (Path): the target directory where the files are to be extracted
        source_lang (str): the source language 
    """
    logger.info("Finding missing files...")
    missing_file_names = get_missing_file_names(path=destination_path, source_lang=source_lang)

    logger.info("Extracting missing files from tarball...")

    for i in range(len(missing_file_names)):
        with tarfile.open(tarball_path, mode="r") as archive:
            archive.extract(member=missing_file_names[i], path=destination_path)


def allow_full_language_names(lang: str) -> str:
    """
    Ensure that if the full name of the source language or the target language (English) 
    is entered, it will be converted into its abbreviated form for later use elsewhere 
    in the code. 
    """
    if lang.lower() in source_languages.keys():
        return source_languages[lang.lower()]
    elif lang.lower() in source_languages.values():
        return lang.lower()
    elif lang.lower() in ["en", "eng", "english"]:
        return "en"
    else:
        raise NotImplementedError("Data in the requested source language is unavailable")


if __name__ == "__main__":

    make_fundamental_paths()
    for language in source_languages.keys():
        download_data(
            source_lang=allow_full_language_names(lang=language),
            keep_tarball=False
        )
