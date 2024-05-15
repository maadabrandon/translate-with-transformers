"""
This module contains code that assembles the encoder and decoder blocks
that we then use to construct the transformer.

I chose to separate it from the internals of the transformer for enhanced 
readability.
"""

from torch.nn import init, ModuleList

from src.training_pipeline.models.scratch.components import (
    InputEmbedding, PositionalEncoding, MultiHeadAttention, FeedForwardBlock, EncoderBlock, DecoderBlock, Decoder,
    Encoder, ProjectionLayer, Transformer
)


def assemble_encoder_blocks(d_model: int, d_ff: int, dropout: float, num_blocks: int, heads: int) -> ModuleList:
    """
    This is where we put together the all the components that make up each encoder block, and
    put num_blocks of these blocks into a list that is then returned.

    Returns:
        ModuleList: all the encoder blocks that will make up the encoder.
    """

    # Create encoder blocks
    encoder_blocks = ModuleList()
    for _ in range(num_blocks):

        # The two sub-layers of each encoder block
        encoder_self_attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout=dropout)
        feed_forward = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Put these elements together into each encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention, feed_forward=feed_forward, dropout=dropout)
        encoder_blocks.append(encoder_block)

    return encoder_blocks


def assemble_decoder_blocks(d_model: int, dropout: float, d_ff: int, num_blocks: int, heads: int) -> ModuleList:
    """
    We put together the all the components that make up each decoder block, and
    put num_blocks of these blocks into a list that is then returned.

    Returns:
        ModuleList: all the decoder blocks that the decoder will consist of.
    """

    decoder_blocks = ModuleList()

    for _ in range(num_blocks):
        # Set up the components of the decoder block
        decoder_self_attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout=dropout)
        decoder_cross_attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout=dropout)
        feed_forward = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Create each decoder block
        decoder_block = DecoderBlock(
            dropout=dropout,
            feed_forward=feed_forward,
            self_attention_block=decoder_self_attention,
            cross_attention_block=decoder_cross_attention
        )

        decoder_blocks.append(decoder_block)

    return decoder_blocks


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_seq_length: int,
    target_seq_length: int,
    dropout: float,
    d_model: int,
    d_ff: int,
    num_blocks: int,
    heads: int
    ) -> Transformer:
    """
    Put all the components of the transformer together, with the help of 
    the helper functions that we used to create the encoder and decoder. 

    Args:
        source_vocab_size (int): _description_
        target_vocab_size (int): _description_
        source_seq_length (int): _description_
        target_seq_length (int): _description_
        dropout (float): _description_
        d_model (int): _description_
        d_ff (int): _description_
        num_blocks (int): _description_
        heads (int): _description_

    Returns:
        Transformer: the constructed transformer model.
    """

    source_embedding = InputEmbedding(d_model=d_model, vocab_size=source_vocab_size)
    target_embedding = InputEmbedding(d_model=d_model, vocab_size=target_vocab_size)

    # Initialise positional encoding layers
    source_position = PositionalEncoding(d_model=d_model, seq_length=source_seq_length, dropout=dropout)
    target_position = PositionalEncoding(d_model=d_model, seq_length=target_seq_length, dropout=dropout)

    # Assemble the encoder and decoder blocks
    encoder_blocks = assemble_encoder_blocks(d_model=d_model, d_ff=d_ff, num_blocks=num_blocks, heads=heads)
    decoder_blocks = assemble_decoder_blocks(d_model=d_model, d_ff=d_ff, num_blocks=num_blocks, heads=heads)

    # Initialise the encoder and decoder
    encoder = Encoder(layers=encoder_blocks)
    decoder = Decoder(layers=decoder_blocks)

    # Initialise the projection layer
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=target_vocab_size)

    # Make the transformer from these components
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        source_embed=source_embedding,
        target_embed=target_embedding,
        source_position=source_position,
        target_position=target_position,
        projection_layer=projection_layer

    )

    # Using Xavier initialization to initialize the parameters of the transformer. This avoids random
    # initialization of weight matrices and scales them to avoid exploding and vanishing gradients.
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
             init.xavier_uniform(parameter)
    
    return transformer
     