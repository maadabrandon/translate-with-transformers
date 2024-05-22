from unittest import TestCase
from src.feature_pipeline.preprocessing import BilingualData
from src.feature_pipeline.scratch.model_inputs import TransformerInputs

class TestModelInputSizes(TestCase):

    def __init__(self, source_lang: str, seq_length: int):
        self.source_lang = source_lang
        self.seq_length = seq_length

        self.inputs = TransformerInputs(
            seq_length=self.seq_length,
            data=BilingualData(source_lang=self.source_lang)
        )
        
    def _test_encoder_input_size(self):
        assert self.inputs.encoder_input.size(0) == self.inputs.seq_length

    def _test_decoder_input_size(self):
        assert self.inputs.decoder_input.size(0) == self.inputs.seq_length

    def _test_label_size(self):
        assert self.inputs.label.size(0) == self.inputs.seq_length
