from src.training_pipeline.models.scratch.components import (
    InputEmbedding, PositionalEncoding, MultiHeadAttention, FeedForwardBlock, EncoderBlock, DecoderBlock,
    Decoder, Encoder, VocabProjectionLayer, Transformer
)

from torch.nn import init, Module


def assemble_encoder_blocks(d_model: int, dropout: float, d_ff: int, n: int, h: int) -> list[Module]:
    """
    This is where we put together the all the components that make up each encoder block, and
    put n of these blocks into a list that is then returned.

    Returns:
        list[Module]: all of the encoder blocks that will make up the encoder.
    """

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(n):

        # The two sublayers of each encoder block
        encoder_self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        feed_forward = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Put these elements together into each encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention, feed_forward=feed_forward, dropout=dropout)
        encoder_blocks.append(encoder_block)

    return encoder_blocks


def assemble_decoder_blocks(d_model: int, dropout: float, d_ff: int, n: int, h: int) -> list[Module]:
    """
    We put together the all the components that make up each decoder block, and
    put n of these blocks into a list that is then returned.

    Returns:
        list[Module]: all of the decoder blocks that the decoder will consist of.
    """

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(n):

        # Setup the components of the decoder block
        decoder_self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        decoder_cross_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
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
    encoder_blocks: list,
    decoder_blocks: list,
    source_vocab_size: int,
    target_vocab_size: int,
    source_seq_length: int,
    target_seq_length: int,
    dropout: float,
    d_model: int,
    d_ff: int,
    n: int,
    h: int
    ) -> Transformer:

    """
    Put all the components of the transformer together, with the help of 
    the helper functions that we used to create the encoder and decoder. 
    """

    source_embedding = InputEmbedding(d_model=d_model, vocab_size=source_vocab_size)
    target_embedding = InputEmbedding(d_model=d_model, vocab_size=target_vocab_size)

    # Initialise positional encoding layers
    source_position = PositionalEncoding(d_model=d_model, seq_length=source_seq_length, dropout=dropout)
    target_position = PositionalEncoding(d_model=d_model, seq_length=target_seq_length, dropout=dropout)

    # Assemble the encoder and decoder blocks
    encoder_blocks = assemble_encoder_blocks(d_model=d_model, d_ff=d_ff, n=n, h=h)
    decoder_blocks = assemble_decoder_blocks(d_model=d_model, d_ff=d_ff, n=n, h=h)

    # Initialise the encoder and decoder
    encoder = Encoder(layers=encoder_blocks)
    decoder = Decoder(layers=decoder_blocks)

    # Initialise the projection layer
    projection_layer = VocabProjectionLayer(d_model=d_model, vocab_size=target_vocab_size)

    # Makse the transformer
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        source_embed=source_embedding,
        target_embed=target_embedding,
        source_position=source_position,
        target_position=target_position,
        projection_layer=projection_layer

    )

    # Xavier inititialization to initialize the parameters of the transformer.
    # This avoids random initialization of weight matrices and scales them to avoid
    # exploding and vanishing gradients.
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
             init.xavier_uniform(parameter)

