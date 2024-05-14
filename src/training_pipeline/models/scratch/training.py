from loguru import logger
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop

from src.training_pipeline.models.scratch.components import Transformer
from src.training_pipeline.models.scratch.construction import build_transformer
from src.feature_pipeline.preprocessing import make_data_loaders, BilingualData, DataSplit, TransformerInputs


def get_optimizer(
    model_fn: Transformer,
    learning_rate: float,
    optimizer_name: str|None, 
    weight_decay: float|None, 
    momentum: float|None
) -> Adam|SGD|RMSprop:
    """
    Returns the required optimizer function, based on the name of the requested
    optimizer. In doing so, different spellings of the provided name are 
    accepted.

    Args: 
        model_fn: the (transformer) model that is to be trained

        optimizer_name: the function that will be used to search for the
                        global minimum of the loss function.

        learning_rate: the learning rate that is optimizer is using for 
                       its search.

        weight_decay: a regularization term that reduces the network's weights

        momentum: the momentum coefficient used during stochastic gradient descent (SGD)
        
    Raises:
        NotImplementedError: The requested optimizer has not been implemented

    Returns:
        Adam, SGD, & RMSprop: the optimizer that will be returned.
    """

    optimizers_and_likely_spellings = {
        ("adam", "Adam"): Adam(params=Transformer.parameters(), lr=learning_rate, weight_decay=weight_decay),
        ("sgd", "SGD"): SGD(params=model_fn.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum),
        ("rms", "rmsprop", "RMSprop", "RMSProp"): RMSprop(params=model_fn.parameters(), weight_decay=weight_decay, momentum=momentum)
    }

    optimizer_with_each_spelling = {
        spelling: optimizer for spellings, function in optimizers_and_likely_spellings.items() for spelling in spellings
    }

    if optimizer_name in optimizer_with_each_spelling.keys():
        return optimizer_with_each_spelling[optimizer_name]

    # Make Adam the default if no optimizer is specified
    elif optimizer_name is None:
        return optimizer_with_each_spelling["Adam"]

    else:
        raise NotImplementedError("We only accept the Adam, SGD, and RMS optimizers")


def run_training_loop(
    source_lang: str,
    data_loaders: tuple[DataLoader, DataLoader, DataLoader],
    model_fn: Transformer,
    criterion: callable,
    optimizer: Adam|SGD|RMSprop,
    num_epochs: int,
    batch_size: int,
    save: bool
):
    logger.info("Collecting training data...")

    data = BilingualData(source_lang=source_lang) 
    model_inputs = TransformerInputs(seq_length=30, data=data)
    data_split = DataSplit(source_lang=source_lang, train_size=0.7, val_size=0.2)

    train_loader = data_split._make_data_loaders(source_lang=source_lang)[0]
    train_iterator = iter(train_loader)

    logger.info("Setting training device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fn.to(device=device)

    logger.info("Training...")
    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}")

        # Put the model in training mode
        model_fn.train()

        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch+1}")

        for batch in batch_iterator:

            # Prepare transformer inputs 
            encoder_input = model_inputs.__getitem__()["encoder_input"]
            decoder_input = model_inputs.__getitem__()["decoder_input"]