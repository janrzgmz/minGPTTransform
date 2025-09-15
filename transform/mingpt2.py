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
import glob
import re
from copy import deepcopy

# Seeds & Hyperparameters
set_seed(3407)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

model_type = 'gpt2-medium' # 345M parameters
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

block_size = 512
learning_rate = 3e-5
max_iters = 40000
batch_size = 2
num_workers = 0
gradient_accumulation_steps = 8

pretrained_model = GPT.from_pretrained(model_type)
pretrained_model.to(device)
pretrained_model.eval()

# Example
def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '':
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']
    x = x.expand(num_samples, -1)
    y = pretrained_model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
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
vocab_size = len(tokenizer)

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

rng = random.Random(3407) 
indices = rng.sample(range(len(val_dataset)), 2000)
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

# Directory where to save all checkpoints and models
save_dir = "/home/jan/gptTransform/minGPTTransform/savedModel"
os.makedirs(save_dir, exist_ok=True)

# Save checkpoint
def _to_serializable_config(cfg):
    # Intenta convertir a dict; si no, devuelve None para omitirlo
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return deepcopy(cfg)
    # objetos tipo SimpleNamespace / dataclass / configs con __dict__
    if hasattr(cfg, "__dict__"):
        return deepcopy(cfg.__dict__)
    # algunos configs de minGPT tienen método get_default_config() → dict-like
    if hasattr(cfg, "to_dict"):
        return deepcopy(cfg.to_dict())
    return None  # evita pickle de objetos raros

def safe_torch_save(payload, path):
    ckpt_dir = os.path.dirname(path)
    if ckpt_dir and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    # opcional: compatibilidad antigua
    torch.save(payload, path)

"""
# Load checkpoints
def parse_step_from_ckpt(path):
    match = re.search(r"ckpt_step(\d+)\.pt", os.path.basename(path))
    return int(match.group(1)) if match else -1

def save_checkpoint(trainer, ckpt_dir, train_losses, train_steps, best_val_loss=None, current_val_loss=None, k_keep=3):
    os.makedirs(ckpt_dir, exist_ok=True)
    step_num = trainer.step_num
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step_num}.pt")

    # Save everything
    torch.save({
        'config': trainer.config,  # for reproducibility
        'step_num': step_num,
        'iter_num': trainer.iter_num,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_losses': train_losses,
        'train_steps': train_steps,
    }, ckpt_path)
    print(f"[Checkpoint] Saved at {ckpt_path}")

    # Save best model (according to validation)
    if current_val_loss is not None:
        if best_val_loss is None or current_val_loss < best_val_loss:
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(trainer.model.state_dict(), best_path)
            print(f"[Checkpoint] New best model saved at {best_path} (val_loss={current_val_loss:.5f})")
            best_val_loss = current_val_loss

    # Keep only the last k_keep
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt"))
    ckpts_sorted = sorted(ckpts, key=parse_step_from_ckpt)
    if len(ckpts_sorted) > k_keep:
        for ckpt_to_remove in ckpts_sorted[:-k_keep]:
            os.remove(ckpt_to_remove)
            print(f"[Checkpoint] Removed old checkpoint {ckpt_to_remove}")

    return best_val_loss

def get_last_checkpoint(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt"))
    if not ckpts:
        return None
    ckpts_sorted = sorted(ckpts, key=parse_step_from_ckpt)
    return ckpts_sorted[-1]

def load_checkpoint(trainer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer = trainer.model.configure_optimizers(trainer.config)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    trainer.iter_num = checkpoint.get('iter_num', 0)
    trainer.step_num = checkpoint.get('step_num', 0)

    train_losses = checkpoint.get('train_losses', [])
    train_steps = checkpoint.get('train_steps', [])

    print(f"[Checkpoint] Loaded from {checkpoint_path}, step {trainer.step_num}, iter {trainer.iter_num}")
    return trainer, train_losses, train_steps
"""

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = model_type
model_config.vocab_size = vocab_size
model_config.block_size = block_size
model = GPT(model_config)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = learning_rate
train_config.max_iters = max_iters
train_config.batch_size = batch_size
train_config.num_workers = num_workers
train_config.gradient_accumulation_steps = gradient_accumulation_steps
train_config.betas = (0.9, 0.95)
train_config.weight_decay = 0.1
train_config.grad_norm_clip = 1.0

trainer = Trainer(train_config, model, train_dataset)

train_losses_mingpt = []
train_steps_mingpt = [] 
# best_val_loss = None

def batch_end_callback_step(trainer):
    # global best_val_loss

    # intervals measured in EFFECTIVE STEPS (updates)
    log_interval_steps = 100
    val_interval_steps = 1000

    # LOG
    if trainer.step_num % log_interval_steps == 0:
        tr_loss = float(trainer.loss.item())
        ppl = math.exp(trainer.loss.item()) if tr_loss < 20 else float('inf')
        print(f"[minGPT] step {trainer.step_num} | micro_iter {trainer.iter_num} | "
              f"train loss {tr_loss:.5f} | ppl {ppl:.2f} | dt {trainer.iter_dt*1000:.1f}ms")
        train_losses_mingpt.append(tr_loss)
        train_steps_mingpt.append(trainer.step_num)

    # VALID + CKPT + (optional) generation
    if trainer.step_num > 0 and trainer.step_num % val_interval_steps == 0:
        vloss, vppl = evaluate(trainer.model, val_loader_small)
        print("-" * 100)
        print(f"[Validation, minGPT] step {trainer.step_num}: loss {vloss:.5f}, ppl {vppl:.2f}")
        # Short sample generation (optional)
        prompt = 'Artificial intelligence in modern age'
        sample = generate_with_model(trainer.model, tokenizer, prompt, steps=50, num_samples=1)[0]
        print(f"[Text generation, minGPT] step {trainer.step_num} | prompt: {prompt}\n{sample}")
        print("-" * 100)
        
trainer.set_callback('on_step_end', batch_end_callback_step)

trainer.run()

# Save minGPT final
final_ckpt_path = os.path.join(save_dir, "minGPT_final.pt")
payload = {
    'config': _to_serializable_config(trainer.config),
    'step_num': trainer.step_num,
    'iter_num': trainer.iter_num,
    'model_state_dict': trainer.model.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'train_losses': train_losses_mingpt,
    'train_steps': train_steps_mingpt,
}
safe_torch_save(payload, final_ckpt_path)
print(f"[Checkpoint] Final minGPT model saved at {final_ckpt_path}")

# free up memory before starting minGPT_tce
del trainer, model
torch.cuda.empty_cache()

# create a GPT instance
model_config_tce = GPT_tce.get_default_config()
model_config_tce.model_type = model_type
model_config_tce.vocab_size = vocab_size
model_config_tce.block_size = block_size
model_tce = GPT_tce(model_config_tce)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = learning_rate
train_config.max_iters = max_iters
train_config.batch_size = batch_size
train_config.num_workers = num_workers
train_config.gradient_accumulation_steps = gradient_accumulation_steps
train_config.betas = (0.9, 0.95)
train_config.weight_decay = 0.1
train_config.grad_norm_clip = 1.0

trainer_tce = Trainer(train_config, model_tce, train_dataset)

train_losses_mingpt_tce = []
train_steps_mingpt_tce = []
# best_val_loss_tce = None

def batch_end_callback_step_tce(trainer):
    # global best_val_loss_tce

    # intervals measured in EFFECTIVE STEPS (updates)
    log_interval_steps = 100
    val_interval_steps = 1000

    # LOG
    if trainer.step_num % log_interval_steps == 0:
        tr_loss = float(trainer.loss.item())
        ppl = math.exp(trainer.loss.item()) if tr_loss < 20 else float('inf')
        print(f"[minGPT_tce] step {trainer.step_num} | micro_iter {trainer.iter_num} | "
              f"train loss {tr_loss:.5f} | ppl {ppl:.2f} | dt {trainer.iter_dt*1000:.2f}ms")
        train_losses_mingpt_tce.append(tr_loss)
        train_steps_mingpt_tce.append(trainer.step_num)

    # VALID + CKPT + (optional) generation
    if trainer.step_num > 0 and trainer.step_num % val_interval_steps == 0:
        vloss, vppl = evaluate(trainer.model, val_loader_small)
        print("-" * 100)
        print(f"[Validation, minGPT_tce] step {trainer.step_num}: loss {vloss:.5f}, ppl {vppl:.2f}")
        # Short sample generation (optional)
        prompt = 'Artificial intelligence in modern age'
        sample = generate_with_model(trainer.model, tokenizer, prompt, steps=50, num_samples=1)[0]
        print(f"[Text generation, minGPT_tce] step {trainer.step_num} | prompt: {prompt}\n{sample}")
        print("-" * 100)

trainer_tce.set_callback('on_step_end', batch_end_callback_step_tce)

trainer_tce.run()

# Save minGPT_tce final
final_ckpt_path_tce = os.path.join(save_dir, "minGPT_tce_final.pt")
payload_tce = {
    'config': _to_serializable_config(trainer_tce.config),
    'step_num': trainer_tce.step_num,
    'iter_num': trainer_tce.iter_num,
    'model_state_dict': trainer_tce.model.state_dict(),
    'optimizer_state_dict': trainer_tce.optimizer.state_dict(),
    'train_losses': train_losses_mingpt_tce,
    'train_steps': train_steps_mingpt_tce,
}
safe_torch_save(payload_tce, final_ckpt_path_tce)
print(f"[Checkpoint] Final minGPT_tce model saved at {final_ckpt_path_tce}")


plt.figure(figsize=(10,6))
plt.plot(train_steps_mingpt,      train_losses_mingpt,      label="minGPT")
plt.plot(train_steps_mingpt_tce,  train_losses_mingpt_tce,  label="minGPT_tce")
plt.xlabel("Optimizer step")
plt.ylabel("Training loss")
plt.title("Training loss vs optimizer step")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_loss.png")) # Save training loss plot
plt.close()

def compare_tce(prompt='', steps=20, b_seed=None, num_samples=1):
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    if prompt == '':
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids'].expand(num_samples, -1)

    # Generation with fixed b (self.conformal_b)
    y_fixed = model_tce.generate(
        x,
        max_new_tokens=steps,
        do_sample=True,
        top_k=40,
        b_seed=None,
        b_mode="default"
    )

    # Generation with a new b generated by seed
    y_new = model_tce.generate(
        x,
        max_new_tokens=steps,
        do_sample=True,
        top_k=40,
        b_seed=b_seed,
        b_mode="seed"
    )

    y_ortho = model_tce.generate(
        x,
        max_new_tokens=steps,
        do_sample=True,
        top_k=40,
        b_seed=None,
        b_mode="orthogonal"
    )

    print("-"*40, "default b", "-"*40)
    for i in range(num_samples):
        print(tokenizer.decode(y_fixed[i].cpu().squeeze()))

    print("\n" + "-"*40, f"changed b_seed={b_seed}", "-"*40)
    for i in range(num_samples):
        print(tokenizer.decode(y_new[i].cpu().squeeze()))
    
    print("\n" + "-"*40, "orthogonal b", "-"*40)
    for i in range(num_samples):
        print(tokenizer.decode(y_ortho[i].cpu().squeeze()))

compare_tce(prompt="Artificial intelligence in modern age", steps=20, b_seed=2100, num_samples=1)