import torch
from src.feature_pipeline.preprocessing import BilingualData
    
class TransformerInputs():

    def __init__(self, seq_length: int, data: BilingualData) -> None:
        self.seq_length = seq_length
        self.encoder_input_tokens = data._retrieve_tokens()[0]
        self.decoder_input_tokens = data._retrieve_tokens()[1]
        
        self.sos_id_tensor = torch.tensor([self.encoder_input_tokens["<SOS>"]], dtype=torch.int64).unsqueeze(dim=0)
        self.eos_id_tensor = torch.tensor([self.encoder_input_tokens["<EOS>"]], dtype=torch.int64).unsqueeze(dim=0)
        self.pad_id_tensor = torch.tensor([self.encoder_input_tokens["<PAD>"]], dtype=torch.int64).unsqueeze(dim=0)

        self.encoder_num_padding_tokens = self.seq_length - len(self.encoder_input_tokens) - 2
        self.decoder_num_padding_tokens = self.seq_length - len(self.decoder_input_tokens) - 1

        self.encoder_input = torch.cat(
            [
                self.sos_id_tensor,
                torch.tensor([list(self.encoder_input_tokens.values())], dtype=torch.int64).unsqueeze(1),
                self.eos_id_tensor,
                torch.tensor([self.encoder_input_tokens["<PAD>"]] * self.encoder_num_padding_tokens, dtype=torch.int64).unsqueeze(dim=0)
            ], dim=0
        )

        self.decoder_input = torch.cat(
            [   
                self.sos_id_tensor,
                torch.tensor([list(self.decoder_input_tokens.values())], dtype=torch.int64),
                torch.tensor([self.decoder_input_tokens["<PAD>"]] * self.decoder_num_padding_tokens, dtype=torch.int64).unsqueeze(dim=0)
            ], dim=0
        )

        self.label = torch.cat(
            [   
                torch.tensor([list(self.decoder_input_tokens.values())], dtype=torch.int64),
                self.eos_id_tensor, 
                torch.tensor([self.encoder_input_tokens["<PAD>"]] * self.decoder_num_padding_tokens, dtype=torch.int64).unsqueeze(dim=0)
            ], dim=0
        )

    def _enough_tokens(self, seq_length: int) -> ValueError:
        """
        A checker that produces an error if the number of input tokens 
        into the encoder or decoder is too high.

        Args:
            seq_length (int): the maximum length of each sentence.

        Raises:
            ValueError: an error message that indicates an excessive 
                        number of input tokens into the encoder or
                        decoder.
        """

        if self.encoder_num_padding_tokens < 0 or self.decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")


    def __getitem__(self) -> dict:
        """
        Return the encoder and decoder inputs, which are both of dimension 
        (seq_length, ), as well as the encoder and decoder masks.
        
        We also establish the encoder and decoder masks. The encoder mask 
        includes the elements of the encoder input that are not padding 
        tokens.

        The decoder mask is meant to ensure that each word in the decoder 
        only watches words that come before it. It does this by zeroing out
        the upper triangular part of a matrix.

        The masked encoder and decoder inuts are twice unsqueezed with respect 
        to the first dimension. Doing this adds sequence and batch dimensions
        to the tensors in the mask.
        """
        return {
            "label": self.label,
            "encoder_input": self.encoder_input,
            "decoder_input": self.decoder_input, 
            "encoder_mask": (self.encoder_input != self.pad_id_tensor).unsqueeze(dim=0).unsqueeze(dim=0).int(), 
            "decoder_mask": (self.decoder_input != self.pad_id_tensor).unsqueeze(dim=0).unsqueeze(dim=0).int() \
                            & self._causal_mask(size=self.decoder_input.size(dim=0))
        }


    def _print_sizes(self):
        """
        A temporary method to be used to diagnose a dimensional
        issue with the tensors in the encoder input
        """
        for tensor in [
                self.sos_id_tensor,
                torch.tensor([list(self.encoder_input_tokens.values())], dtype=torch.int64).unsqueeze(1),
                self.eos_id_tensor,
                torch.tensor([self.encoder_input_tokens["<PAD>"]] * self.encoder_num_padding_tokens, dtype=torch.int64).unsqueeze(0)
            ]:

            print(tensor.size())
            

    def _causal_mask(size: int) -> torch.Tensor: 
        """
        Make a matrix of ones whose upper triangular part is full of zeros.

        Args:
            size (int): the second and third dimensions of the matrix.

        Returns:
            torch.Tensor: return all the values above the diagonal, which should be the
                  upper triangular part.
        """
        mask = torch.triu(input=torch.ones(1, size, size), diagonal=1).type(torch.int64)
        return mask == 0


if __name__ == "__main__":

    TransformerInputs(
        seq_length=31000, data=BilingualData(source_lang="de")
    )._print_sizes() 
    