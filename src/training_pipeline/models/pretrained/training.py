from transformers import MBartForConditionalGeneration
from transformers import Trainer, TrainingArguments

from src.setup.paths import PRETRAINED_DIR
from src.feature_pipeline.preprocessing import DataSplit

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")


training_args = TrainingArguments(
    output_dir=PRETRAINED_DIR/"artifacts",
    per_device_eval_batch_size=30,
    num_train_epochs=20,
    save_steps=5,
    do_train=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=
)


