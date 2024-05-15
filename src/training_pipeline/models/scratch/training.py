import torch

from tqdm import tqdm 
from loguru import logger

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torch.optim import Adam, SGD, RMSprop
from torch.utils.tensorboard import SummaryWriter

from src.training_pipeline.models.scratch.components import Transformer
from src.training_pipeline.models.scratch.construction import build_transformer
from src.feature_pipeline.preprocessing import BilingualData, DataSplit, TransformerInputs


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
        optimizer_name: the function that will be used to search for the
                        global minimum of the loss function.

        learning_rate: the learning rate that is optimizer is using for its search.

        weight_decay: a regularization term that reduces the network's weights

        momentum: the momentum coefficient used during stochastic gradient descent (SGD)

    Raises:
        NotImplementedError: The requested optimizer has not been implemented

    Returns:
        Adam, SGD, & RMSprop: the optimizer that will be returned.
    """
    optimizers_and_likely_spellings = {
        ("adam", "Adam"): Adam(params=model_fn.parameters(), lr=learning_rate, weight_decay=weight_decay),
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
    num_epochs: int,
    batch_size: int,
    save: bool
):
    global_step = 0
    logger.info("Collecting training data...")

    # Initialise the relevant data object
    data = BilingualData(source_lang=source_lang) 
    
    # Instantiate the transformer inputs and split the data as we see fit
    model_inputs = TransformerInputs(seq_length=30, data=data)
    data_split = DataSplit(source_lang=source_lang, train_size=0.7, val_size=0.2)

    # Get the training dataloader
    train_loader = data_split._make_data_loaders(source_lang=source_lang)[0]
    train_iterator = iter(train_loader)

    logger.info("Setting training device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allow retrieval of the model and retrieve it.
    def get_model():

        return build_transformer(
            source_vocab_size=data.source_lang_vocab_size, 
            target_vocab_size=data.en_vocab_size, 
            source_seq_length=350,
            target_seq_length=350,
            dropout=0.2,
            d_model=512,
            d_ff=512,
            num_blocks=6,
            heads=5
    )

    model_fn = get_model().to(device=device)

    # Instantiate the loss function and the optimizer 
    loss_fn = CrossEntropyLoss(ignore_index=model_inputs.encoder_input_tokens["<PAD>"], label_smoothing=0.1)
    optimizer = get_optimizer(model_fn=model_fn, learning_rate=0.01, optimizer_name="adam", weight_decay=0.001, momentum=None)
    loss_fn = loss_fn.to(device=device)

    logger.info("Training...")
    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}")

        # Put the model in training mode
        model_fn.train()

        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch+1}")
        for batch in batch_iterator:

            # Prepare transformer inputs and labels
            encoder_input = model_inputs.__getitem__()["encoder_input"].to(device)
            decoder_input = model_inputs.__getitem__()["decoder_input"].to(device)
            encoder_mask = model_inputs.__getitem__()["encoder_mask"].to(device)
            decoder_mask = model_inputs.__getitem__()["decoder_mask"].to(device)
            label = model_inputs.__getitem__()["label"].to(device)

            # Run the inputs through the transformer
            encoder_output = model_fn._encode(input=encoder_input, source_mask=encoder_mask)

            decoder_output = model_fn._decode(
                encoder_output=encoder_output, 
                source_mask=encoder_mask, 
                target=decoder_input,
                target_mask=decoder_mask
            )

            projected_output = model_fn._project(x=decoder_output)

            # Calculate the cross-entropy loss using the projected output and the 
            training_loss = loss_fn(
                projected_output.view(-1, data.en_vocab_size), label.view(-1)
            )

            # Update progress bar with the computed loss
            batch_iterator.set_postfix(ordered_dict={f"Training Loss": f"{training_loss.item():6.3f}"})

            # Log the loss 
            SummaryWriter.add_scalar(tag="train_loss", scalar_value=training_loss.item(), global_step=True)
            SummaryWriter.flush()

            training_loss.backward()

            optimizer.step()
            optimizer.zero_grad()            

            global_step += 1

        if save:

            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state": model_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step
                }
            )


if __name__ == "__main__":

    run_training_loop(
        source_lang="de",
        num_epochs=20,
        batch_size=20, 
        save=True
    )