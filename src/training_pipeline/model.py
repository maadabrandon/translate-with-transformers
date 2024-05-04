import math
import torch

from torch.nn import Sequential, Embedding, Module


class InputEmbeddings(Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: tensor.Tensor) -> torch.Tensor:
        """
        In "Attention is All You Need", the authors multiplied the 
        weights in the weight matrix produced by the embedding layers
        by the square root of the embedding dimension.

        Returns:
            Tensor: the scaled weights from the embedding layer.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(Module):

    def __init__(self, d_model: int, sequence_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_lengths
        self.dropout = dropout
        self.encoding = self._encode(sequence_length=sequence_length, d_model=d_model)

    def _encode(self, sequence_length: int, d_model: int) -> torch.Tensor:

        # Make a matrix with the given shape 
        positional_encoding = torch.zeros(sequence_length, d_model)
    
        # Make vector of length (sequence_length, 1)
        position = torch.arange(start=0, end=sequence_length, dtype = torch.float).unsqueeze(1)
        
        divisor_term = torch.exp(
            (torch.arange(start=0, end=d_model, step=2).float()/d_model) * -math.log(10000.0)
        )

        # Apply the sine and cosine to even and odd positions respectively
        positional_encoding[:, 0::2] = torch.sin(position * divisor_term)
        positional_encoding[:, 1::2] = torch.cos(position * divisor_term)

        # Reshape the tensor so that it has shape (1, sequence_length, d_model)
        positional_encoding.unsqueeze(9)

        # Save the encoding as a buffer. A buffer is a named tensor whose value 
        # has no impact on gradients. In other words, buffers are not learned.
        self.register_buffer(name="positional_encoding", tensor=positional_encoding)

        return positional_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Ensure that autograd does not consider this tensor when adjusting gradients. 
        x += self.encoding[:, :x.shape[1], :].requires_grad_(mode=False)

