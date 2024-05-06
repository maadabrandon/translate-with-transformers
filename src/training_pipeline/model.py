"""
In this module, we construct a transformer based on the paper
which introduced the architecture: "Attention is All You Need"
"""
from math import sqrt, log

from torch import Tensor, relu
from torch.nn import Linear, Sequential, Module, Dropout, Embedding, Parameter   


class InputEmbedding(Module):

    def __init__(self, d_model: int, vocab_size: int) -> Tensor:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        In "Attention is All You Need", the authors multiplied the 
        weights in the weight matrix produced by the embedding layers
        by the square root of the embedding dimension.

        Returns:
            Tensor: the scaled weights from the embedding layer.
        """
        return self.embedding(x) * sqrt(self.d_model)


class PositionalEncoding(Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> Tensor:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_lengths
        self.dropout = dropout
        self.encoding = self._encode(seq_length=seq_length, d_model=d_model)

    def _encode(self, seq_length: int, d_model: int) -> Tensor:

        # Make a matrix with the given shape 
        positional_encoding = torch.zeros(seq_length, d_model)
    
        # Make vector of length (seq_length, 1)
        position = torch.arange(start=0, end=seq_length, dtype = torch.float).unsqueeze(1)
        
        divisor_term = torch.exp(
            (torch.arange(start=0, end=d_model, step=2).float()/d_model) * -log(10000.0)
        )

        # Apply the sine and cosine to even and odd positions respectively
        positional_encoding[:, 0::2] = torch.sin(position * divisor_term)
        positional_encoding[:, 1::2] = torch.cos(position * divisor_term)

        # Reshape the tensor so that it has shape (1, seq_length, d_model)
        positional_encoding.unsqueeze(9)

        # Save the encoding as a buffer. A buffer is a named tensor whose value 
        # has no impact on gradients. In other words, buffers are not learned.
        self.register_buffer(name="positional_encoding", tensor=positional_encoding)

        return positional_encoding

    def forward(self, x: Tensor) -> Tensor:

        # Ensure that autograd does not consider this tensor when adjusting gradients. 
        x += self.encoding[:, :x.shape[1], :].requires_grad_(mode=False)

        return self.dropout(x)


class LayerNormalization(Module):
    """
    This layer performs normalisation of tensor elements along the embedding 
    (feature) dimension.
    """
    def __init__(self, epsilon: float = 10**-6) -> Tensor:
        super().__init__()

        # Added to the variance during normalisation to avoid division by zero
        self.epsilon = epsilon

        # These parameters (multiplicative and additive respectively) introduce some 
        # fluctuations so that the values do not necessarily have to be restricted to 
        # [0,1]. The model will tune these parameters when it deems it necessary.
        self.gamma = Parameter(data=torch.ones(1))
        self.bias = Parameter(data=torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        """
        The linear transformations makes the following dimensional changes:
        (Batch, seq_length, d_model) --> (Batch, seq_length, d_ff) --> (Batch, seq_length, d_model) 
        """
        return self.linear2(
            self.dropout(
                p=relu(self.linear1(x))
            )
        )


class MultiHeadAttention(Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.dropout = Dropout(p=dropout)
        self.attention_scores = None

        assert d_model % h == 0, f"Size of the embedding dimension ({d_model}) not divisible by the number of heads ({h})"
        
        # Define matrices used to transform the query, key, and values respectively
        self.w_k = Linear(in_features=d_model, out_features=d_model)
        self.w_q = Linear(in_features=d_model, out_features=d_model)
        self.w_v = Linear(in_features=d_model, out_features=d_model)
        self.w_o = Linear(in_features=h*self.d_k, out_features=d_model)

    @staticmethod
    def compute_attention(query: Tensor, key: Tensor, value: Tensor, dropout: Dropout, mask) -> tuple[Tensor]:
        """
        Compute the self-attention scores using the scaled dot-product 
        attention formula.
        """
        d_k = query.shape[-1]

        # There is a dimensional change: (Batch, seq_length, d_k) --> (Batch, seq_length, seq_length)
        attention_scores = (query @ key.transpose(dim0=-2, dim1=-1))/sqrt(d_k)
        attention_scores.masked_fill_(mask==0, value=-1e9) if mask is not None else None
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        """
        The multi-head attention object first performs a linear transformation on 
        the query, key, and value matrices. Then the necessary dimensional changes
        on each of these matrices are performed to split the matrices into heads, 
        so that each head has access to the full sequence (sentence) but to a 
        different aspect of the meaning of each word.
        
        The scaled dot-product attention scores are then computed before putting
        the resulting tensor in its original shape.
        
        Args:
            q (Tensor): the query tensor
            k (Tensor): the key tensor
            v (Tensor): the value tensor
            mask (_type_): _description_

        Returns:
            Tensor: _description_
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # We now split the matrices in such a way that each head views the 
        # (Batch, seq_length, d_model) --> (Batch, seq_length, h, d_k) --> (Batch, h, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Compute self-attention scores
        x, self.attention_scores = MultiHeadAttention.compute_attention(
            query=query, 
            key=key, 
            value=value, 
            dropout=self.dropout,
            mask=mask
        )
        
        # Restore the x to its original shape.
        # (Batch, h, seq_length, d_k) --> (Batch, seq_length, h, d_k) --> (Batch, seq_length, d_model)
        x = x.transpose(dim0=1,dim1=2).contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)

        return self.w_o(x)


class SkipConnection(Module):

    def __init__(self, dropout: float) -> None:
        super.__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, sublayer: Module) -> Tensor:
        """
        Implement a skip connection by adding the normalised
        output of a given sublayer to the input data.

        Args:
            x (Tensor): input tensor
            sublayer (Module): the next sub-layer

        Returns:
            Tensor: the output of the skip connection
        """

        return x + self.dropout(
            self.norm(sublayer(x))
        )