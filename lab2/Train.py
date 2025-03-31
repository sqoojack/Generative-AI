from Dataset import training_dataset, model_name, tokenizer
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, default_data_collator

""" Configure 4-bit quantization """
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name,    # load the pretrained Llama model in 4-bit
                                            quantization_config=quant_config,
                                            device_map="auto")

model.gradient_checkpointing_enable()   # enable gradient checkpointing for memory efficiency
model = prepare_model_for_kbit_training(model)  # needed before adding LoRA

""" Set up LoRA configuration (QLoRA style: target all linear layers) """
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules="all-linear",    # apply LORA to all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM"   # for decoder-only language modeling
)

# apply the LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

""" Define training arguments (tune epochs and LR as needed) """
training_args = TrainingArguments(
    output_dir="model_checkpoint",
    per_device_train_batch_size=1,  # batch size
    gradient_accumulation_steps=8,  # to simulate larger batch
    num_train_epochs=2,
    learning_rate=0.001,
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    evaluation_strategy="no"    # because we will do custom evaluation later
)

""" Create trainer with our data """
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    data_collator=default_data_collator
)

trainer.train()     # start training
model.save_pretrained("model_checkpoint")
tokenizer.save_pretrained("model_checkpoint")
