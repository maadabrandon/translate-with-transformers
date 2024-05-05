"""
In this module, we construct a transformer based on the paper
which introduced the architecture: "Attention is All You Need"
"""
import torch

from math import sqrt, log
from torch.nn import Linear, Sequential, Module, Dropout, Embedding, Parameter   


class InputEmbedding(Module):

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
        return self.embedding(x) * sqrt(self.d_model)


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
            (torch.arange(start=0, end=d_model, step=2).float()/d_model) * -log(10000.0)
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

        return self.dropout(x)


class LayerNormalization(Module):
    """
    This layer performs normalisation of tensor elements along the embedding 
    (feature) dimension 
    """
    def __init__(self, epsilon: float = 10**-6):
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
        z_score = (x-mean)/(sqrt(var + self.epsilon)) 

        return self.gamma * z_score + self.bias


class FeedForward(Module):
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
        The linear transformations makes the following dimensional changes 
        (Batch, sequence_length, d_model) --> (Batch, sequence_length, d_ff) --> (Batch, sequence_length, d_model) 
        """
        return self.linear2(
            self.dropout(
                p=torch.relu(self.linear1(x))
            )
        )

        