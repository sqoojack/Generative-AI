import json
from transformers import AutoTokenizer
import evaluate
import torch
from torch.utils.data import Dataset

Max_len = 1024 # define a maximum sequence length for truncation(截斷) (Llama context length, e.g. 1024 tokens)
train_file = "data/train.json"
test_file = "data/test.json"

""" Load the training data """
train_data = []
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        example = json.load(line)
        train_data.append({
            "paper_id": example["paper_id"],
            "introduction": example["introduction"].strip(),
            "abstract": example["abstract"].strip()
        })
        
""" Load the test data """
test_data = []
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        example = json.load(line)
        test_data.append({
            "paper_id": example["paper_id"],
            "introduction": example["introduction"].strip()     # no abstract in testing data
        })
        
print(f"Successfully Loaded {len(train_data)} training examples and {len(test_data)} test examples.")

""" Initialize the tokenizer for Llama (using a pre-trained Llama tokenizer) """
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

""" tokenize the training data with prompt formatting """
train_codings = {"input_ids": [], "attention_mask": [], "labels": []}
for examples in train_data:
    intro = example["introduction"]
    abstract = example["abstract"]
    prompt_text = f"Introduction:\n{intro}\nAbstract:\n"    # tell the model the place where the introduction and abstract
    full_text = prompt_text + abstract
    
    tokens = tokenizer(full_text, max_length=Max_len, truncation=True, padding="max_length")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    
    """ create labels, and set token of introduction part to -100. In order to ignore loss when caculating loss """
    labels = input_ids.copy()
    prompt_tokens = tokenizer(prompt_text, max_length=Max_len, trunation=True)
    prompt_length = len(prompt_tokens["input_ids"])
    
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    
    train_codings["input_ids"].append(input_ids)
    train_codings["attention_mask"].append(attention_mask)
    train_codings["labels"].append(labels)

""" Convert list to tensors for PyTorch and create a Dataset object """
class SummarizationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
    
training_dataset = SummarizationDataset(train_codings)