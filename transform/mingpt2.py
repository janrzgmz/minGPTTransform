import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer

from mingpt.model import GPT
from transform.model_tce import GPT as GPT_tce
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

from itertools import chain

import math
import matplotlib.pyplot as plt
import random

# Seeds & Hyperparameters
set_seed(3407)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

model_type = 'gpt2-medium' # 345M parameters
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

block_size = 1024
learning_rate = 3e-4
max_iters = 40001
batch_size = 2
num_workers = 0
gradient_accumulation_steps = 4

model = GPT.from_pretrained(model_type)
model.to(device)
model.eval()

# Example
def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '':
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']
    x = x.expand(num_samples, -1)
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)

generate(prompt='Artificial intelligence in modern age', num_samples=10, steps=20)

# Load dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42) # 90% training, 10% validation
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': ''})

print('train_dataset length: ', len(train_dataset))
print('val_dataset length: ', len(val_dataset))
print('vocab_size: ', tokenizer.vocab_size)
print("train_dataset example:\n", train_dataset[0]["text"][:500])

# Tokenize dataset
tokenizer.model_max_length = int(1e9)

def tokenize_function(examples):
    return tokenizer([t + tokenizer.eos_token for t in examples["text"]], add_special_tokens=False,  return_attention_mask=False)

def group_texts_for_minGPT(examples):
  concatenated = list(chain.from_iterable(examples["input_ids"]))
  window = block_size + 1
  total_length = (len(concatenated) // window) * window
  if total_length == 0:
      return {"input_ids": [], "labels": []}
  chunks = [concatenated[i:i+window] for i in range(0, total_length, window)]
  inputs = [c[:-1] for c in chunks]
  labels = [c[1:]  for c in chunks]
  return {"input_ids": inputs, "labels": labels}

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['id', 'url', 'title', "text"])
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['id', 'url', 'title', "text"])

lm_train = tokenized_train_dataset.map(group_texts_for_minGPT, batched=True)
lm_val = tokenized_val_dataset.map(group_texts_for_minGPT, batched=True)

lm_train.set_format(type="torch", columns=["input_ids", "labels"])
lm_val.set_format(type="torch", columns=["input_ids", "labels"])

class wikiDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        x = item["input_ids"]
        y = item["labels"]
        return x, y

# print an example instance of the dataset
train_dataset = wikiDataset(lm_train)
val_dataset = wikiDataset(lm_val)

x, y = train_dataset[0]
a, b = val_dataset[0]

print('x:\n', x)
print('y:\n', y)
print('a:\n', a)
print('b:\n', b)

indices = random.sample(range(len(val_dataset)), 5000)
val_subset = Subset(val_dataset, indices)

val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,  batch_size=batch_size, num_workers=num_workers)
val_loader_small = DataLoader(val_subset, shuffle=False, pin_memory=True,  batch_size=batch_size, num_workers=num_workers)

# Determine val_loss and perplexity during training
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    if was_training:
      model.train()
    return avg_loss, ppl

# Generate text during training
def generate_with_model(model, tokenizer, prompt, steps=50, num_samples=1):
    was_training = model.training
    model.eval()
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids'].expand(num_samples, -1)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=steps, do_sample=True, top_k=40)
    outputs = []
    for i in range(num_samples):
        outputs.append(tokenizer.decode(y[i].cpu().squeeze()))
    if was_training:
        model.train()
    return outputs

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = model_type
model_config.vocab_size = tokenizer.vocab_size
model_config.block_size = block_size
model = GPT(model_config)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = learning_rate
train_config.max_iters = max_iters
train_config.batch_size = batch_size
train_config.num_workers = num_workers
train_config.gradient_accumulation_steps = gradient_accumulation_steps

trainer = Trainer(train_config, model, train_dataset)

train_losses_mingpt = []

def batch_end_callback(trainer):
    log_interval_updates = 100
    val_interval_updates = 1000

    log_interval = log_interval_updates * trainer.config.gradient_accumulation_steps
    val_interval = val_interval_updates * trainer.config.gradient_accumulation_steps

    if (trainer.iter_num + 1) % trainer.config.gradient_accumulation_steps == 0:
        if trainer.iter_num % log_interval == 0:
            train_loss = trainer.loss.item()
            train_losses_mingpt.append(train_loss)
            perplexity = math.exp(train_loss) if train_loss < 20 else float('inf')
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {train_loss:.5f}, perplexity {perplexity:.2f}")
        if trainer.iter_num > 0 and trainer.iter_num % val_interval == 0:
            # val_loss and val_perplexity
            val_loss, val_ppl = evaluate(trainer.model, val_loader_small)
            print("-"*100)
            print(f"[Validation, minGPT] iter {trainer.iter_num}: loss {val_loss:.5f}, perplexity {val_ppl:.2f}")
            # generate text
            prompt = 'Artificial intelligence in modern age'
            samples = generate_with_model(trainer.model, tokenizer, prompt, steps=50, num_samples=1)
            print(f"[Text generation, minGPT] iter {trainer.iter_num}, prompt: {prompt}")
            print('Generation:\n', samples[0])
            print("-"*100)

trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# create a GPT instance
model_config_tce = GPT_tce.get_default_config()
model_config_tce.model_type = model_type
model_config_tce.vocab_size = tokenizer.vocab_size
model_config_tce.block_size = block_size
model_tce = GPT_tce(model_config_tce)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = learning_rate
train_config.max_iters = max_iters
train_config.batch_size = batch_size
train_config.num_workers = num_workers
train_config.gradient_accumulation_steps = gradient_accumulation_steps

trainer_tce = Trainer(train_config, model_tce, train_dataset)

train_losses_mingpt_tce = []

def batch_end_callback_tce(trainer):
    log_interval_updates = 100
    val_interval_updates = 1000

    log_interval = log_interval_updates * trainer.config.gradient_accumulation_steps
    val_interval = val_interval_updates * trainer.config.gradient_accumulation_steps

    if (trainer.iter_num + 1) % trainer.config.gradient_accumulation_steps == 0:
        if trainer.iter_num % log_interval == 0:
            train_loss = trainer.loss.item()
            train_losses_mingpt_tce.append(train_loss)
            perplexity = math.exp(train_loss) if train_loss < 20 else float('inf')
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {train_loss:.5f}, perplexity {perplexity:.2f}")
        if trainer.iter_num > 0 and trainer.iter_num % val_interval == 0:
            # val_loss and val_perplexity
            val_loss, val_ppl = evaluate(trainer.model, val_loader_small)
            print("-"*100)
            print(f"[Validation, minGPT_tce] iter {trainer.iter_num}: loss {val_loss:.5f}, perplexity {val_ppl:.2f}")
            # generate text
            prompt = 'Artificial intelligence in modern age'
            samples = generate_with_model(trainer.model, tokenizer, prompt, steps=50, num_samples=1)
            print(f"[Text generation, minGPT_tce] iter {trainer.iter_num}, prompt: {prompt}")
            print('Generation:\n', samples[0])
            print("-"*100)

trainer_tce.set_callback('on_batch_end', batch_end_callback_tce)

trainer_tce.run()

plt.figure(figsize=(10,6))
plt.plot(train_losses_mingpt, label="minGPT")
plt.plot(train_losses_mingpt_tce, label="minGPT_tce")
plt.xlabel("Checkpoint (every 100 iterations)")
plt.ylabel("Training loss")
plt.title("Training loss evolution")
plt.legend()
plt.grid(True)
plt.show()