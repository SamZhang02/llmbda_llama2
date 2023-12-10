from transformers import TrainingArguments
from trl import SFTTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from datasets import load_dataset

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("loading model...")
model = AutoModelForCausalLMWithValueHead.from_pretrained("meta-llama/Llama-2-7b-hf", peft_config=lora_config, load_in_4bit=True)
print("model loaded.")

print("loading dataset...")
dataset = load_dataset("saamenerve/mcgill-requisites", split="train")
print("dataset loaded.")

training_args = TrainingArguments(
    output_dir="./models/llaba-2-7b-mcgill-requisites",
    push_to_hub=True
)

print("setting up trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="instruction",
    max_seq_length=4096,
    args=training_args
)
print("trainer set...")

print("start training...")
trainer.train()

