from loguru import logger
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop

from src.training_pipeline.models.scratch.components import Transformer
from src.training_pipeline.models.scratch.construction import build_transformer
from src.feature_pipeline.preprocessing import make_data_loaders, TransformerInputs


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

