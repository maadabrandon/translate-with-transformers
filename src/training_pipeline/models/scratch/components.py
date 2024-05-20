"""
In this module, we detail the components of the transformer based on
the paper which introduced the architecture: "Attention is All You Need"
"""
import torch
from math import sqrt, log
from torch.nn import Linear, Module, Dropout, Embedding, Parameter, ModuleList


class InputEmbedding(Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In "Attention is All You Need", the authors multiplied the
        weights in the weight matrix produced by the embedding layers
        by the square root of the embedding dimension.

        Returns:
            torch.Tensor: the scaled weights from the embedding layer.
        """
        return self.embedding(x) * sqrt(self.d_model)


class PositionalEncoding(Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = Dropout(p=dropout)
        self.encoding = self._encode(seq_length=seq_length, d_model=d_model)

    def _encode(self, seq_length: int, d_model: int) -> torch.Tensor:

        # Make a matrix with the given shape
        positional_encoding = torch.zeros(seq_length, d_model)

        # Make vector of length (seq_length, 1)
        position = torch.arange(start=0, end=seq_length, dtype=torch.float).unsqueeze(1)

        divisor_term = torch.exp(
            (torch.arange(start=0, end=d_model, step=2).float() / d_model) * -log(10000.0)
        )

        # Apply the sine and cosine to even and odd positions respectively
        positional_encoding[:, 0::2] = torch.sin(position * divisor_term)
        positional_encoding[:, 1::2] = torch.cos(position * divisor_term)

        # Reshape the tensor so that it has shape (1, seq_length, d_model)
        positional_encoding.unsqueeze(0)

        # Save the encoding as a buffer. A buffer is a named tensor whose value
        # has no impact on gradients. In other words, buffers are not learned.
        self.register_buffer(name="positional_encoding", tensor=positional_encoding)

        return positional_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure that autograd does not consider this tensor when adjusting gradients.
        x += self.encoding[:, :x.shape[1], :].requires_grad_(mode=False)
        return self.dropout.forward(x)


class LayerNormalization(Module):
    """
    This layer performs normalisation of tensor elements along the embedding
    (feature) dimension.
    """

    def __init__(self, epsilon: float = 10 ** -6) -> None:
        super().__init__()

        # Added to the variance during normalisation to avoid division by zero
        self.epsilon = epsilon

        # These parameters (multiplicative and additive respectively) introduce some
        # fluctuations so that the values do not necessarily have to be restricted to
        # [0,1]. The model will tune these parameters when it deems it necessary.
        self.gamma = Parameter(data=torch.ones(1))
        self.bias = Parameter(data=torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        # Calculate the standard normal values of x
        z_score = (x - mean) / (sqrt(var + self.epsilon))

        return self.gamma * z_score + self.bias


class FeedForwardBlock(Module):
    """
    This is a layer that performs two linear transformations with a ReLU in
    between. There are two bias terms included in each transformation, so
    even though Pytorch sets bias=True by default in Linear layers, I
    made that explicit just for emphasis.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = Linear(in_features=d_model, out_features=d_ff, bias=True)
        self.dropout = Dropout(p=dropout)
        self.linear2 = Linear(in_features=d_ff, out_features=d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The linear transformations makes the following dimensional changes:
        (Batch, seq_length, d_model) --> (Batch, seq_length, d_ff) --> (Batch, seq_length, d_model)
        """
        return self.linear2(
            self.dropout(
                p=torch.relu(self.linear1(x))
            )
        )


class MultiHeadAttention(Module):

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.dropout = Dropout(p=dropout)
        self.attention_scores = None

        assert self.d_model % self.heads == 0, \
            f"Size of the embedding dimension ({d_model}) not divisible by the number of heads ({heads})"

        # Define matrices used to transform the query, key, and values respectively
        self.w_k = Linear(in_features=d_model, out_features=d_model)
        self.w_q = Linear(in_features=d_model, out_features=d_model)
        self.w_v = Linear(in_features=d_model, out_features=d_model)
        self.w_o = Linear(in_features=heads * self.d_k, out_features=d_model)

    @staticmethod
    def compute_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: Dropout,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the self-attention scores using the scaled dot-product attention formula.
        """
        # The size of the embedding (feature) dimension of the query
        d_k = query.shape[-1]

        # There is a dimensional change: (Batch, seq_length, d_k) --> (Batch, seq_length, seq_length)
        attention_scores = (query @ key.transpose(dim0=-2, dim1=-1)) / sqrt(d_k)
        attention_scores.masked_fill_(mask = mask, value=-1e9) if mask is not None else None
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask) -> torch.Tensor:
        """
        The multi-head attention object first performs a linear transformation on
        the query, key, and value matrices. Then the necessary dimensional changes
        on each of these matrices are performed to split the matrices into heads,
        so that each head has access to the full sequence (sentence) but to a
        different aspect of the meaning of each word.

        The scaled dot-product attention scores are then computed before putting
        the resulting tensor in its original shape.

        Args:
            q (torch.Tensor): the query tensor
            k (torch.Tensor): the key tensor
            v (torch.Tensor): the value tensor
            mask (torch.Tensor): 

        Returns:
            torch.Tensor: _description_
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # We now split the matrices in such a way that each head views the
        # (Batch, seq_length, d_model) --> (Batch, seq_length, heads, d_k) --> (Batch, heads, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        # Compute self-attention scores
        x, self.attention_scores = MultiHeadAttention.compute_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            mask=mask
        )

        # Restore the x to its original shape.
        # (Batch, heads, seq_length, d_k) --> (Batch, seq_length, heads, d_k) --> (Batch, seq_length, d_model)
        x = x.transpose(dim0=1, dim1=2).contiguous().view(x.shape[0], x.shape[1], self.heads * self.d_k)

        return self.w_o(x)


class SkipConnection(Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = Dropout(p=dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: Module) -> torch.Tensor:
        """
        Implement a skip connection by adding the normalised
        output of a given sublayer to the input data.

        Args:
            x (torch.Tensor): input tensor
            sublayer (Module): the next sub-layer

        Returns:
            torch.Tensor: the output of the skip connection.
        """
        return x + self.dropout(
            self.norm(
                sublayer(x)
            )
        )


class EncoderBlock(Module):

    def __init__(self, dropout: float, feed_forward: FeedForwardBlock, self_attention_block: MultiHeadAttention) -> None:
        super().__init__()
        self.feed_forward = feed_forward
        self.self_attention_block = self_attention_block

        # There are two skip connections in each encoder block
        self.skip_connections = ModuleList(
            modules=[SkipConnection(dropout=dropout) for _ in [0, 1]]
        )

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:

        # Apply the first skip connection to the input tensor x with the sublayer being the self-attention.
        x = self.skip_connections[0].forward(
            x=x,
            sublayer=self.self_attention_block.forward(q=x, k=x, v=x, mask=source_mask)
        )

        # Apply the second skip connection, where the sublayer in question is a feedforward block.
        x = self.skip_connections[1].forward(
            x=x,
            sublayer=self.feed_forward.forward(x=x)
        )

        return x


class Encoder(Module):

    def __init__(self, layers: ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask):

        for layer in self.layers:
            x = layer.forward(x=x, source_mask=mask)

        return self.norm(x)


class DecoderBlock(Module):

    def __init__(
        self,
        dropout: Dropout,
        feed_forward: FeedForwardBlock,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention
    ) -> None:

        super().__init__()
        self.feed_forward = feed_forward
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block

        # There are three skip connections in each decoder block
        self.skip_connections = ModuleList(
            [SkipConnection(dropout=dropout) for _ in range(3)]
        )

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        source_mask: torch.Tensor, 
        target_mask: torch.Tensor
        ):
        # First skip connection
        x = self.skip_connections[0].forward(
            x=x,
            sublayer=self.self_attention_block.forward(q=x, k=x, v=x, mask=target_mask)
        )

        # Second skip connection, with a cross-attention sublayer
        x = self.skip_connections[1].forward(
            x=x,
            sublayer=self.cross_attention_block.forward(q=x, k=encoder_output, v=encoder_output, mask=source_mask)
        )

        # The final skip connection has a feedforward sublayer
        x = self.skip_connections[2].forward(
            x=x,
            sublayer=self.feed_forward
        )
        return x


class Decoder(Module):

    def __init__(self, layers: ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        source_mask: torch.Tensor, 
        target_mask:torch.Tensor
        ):
    
        for layer in self.layers:
            x = layer.forward(
                x=x,
                encoder_output=encoder_output,
                source_mask=source_mask,
                target_mask=target_mask
            )

        return self.norm(x)


class ProjectionLayer(Module):
    """
    This is a linear map that acts on the embedding (feature) dimension and
    transforms its input tensor so that its feature dimension corresponds to
    the number of words in the entire dataset.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection_layer = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the final dimensional change from
        (Batch, seq_length, d_model) --> (Batch, seq_length, vocab_size)

        Args:
            x (torch.Tensor): the output of the decoder block

        Returns:
            torch.Tensor: the log softmax of the transformed tensor.
        """
        return torch.log_softmax(
            self.projection_layer(x), dim=-1
        )


class Transformer(Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embedding: InputEmbedding,
        target_embedding: InputEmbedding,
        source_position: PositionalEncoding,
        target_position: PositionalEncoding,
        projection_layer: ProjectionLayer
        ) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.projection_layer = projection_layer


    def _encode(self, input: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:

        # Apply the input embedding and positional encoding to the data before it enters the encoder blocks
        source = self.source_position(
            self.source_embedding.forward(input)
        )

        return self.encoder.forward(x=source, mask=source_mask)


    def _decode(
        self, 
        encoder_output: torch.Tensor, 
        source_mask: torch.Tensor, 
        target: torch.Tensor,
        target_mask: torch.Tensor
        ) -> torch.Tensor:

        # Apply the output embedding to the input data
        target = self.target_embedding.forward(x=target)

        # Apply the positional embedding to the data before it enters the decoder blocks
        target = self.target_position.forward(x=target)

        return self.decoder.forward(
            x=target,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask
        )

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer.forward(x=x)
