import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader

from datasets import load_dataset
from transformers import GPT2Tokenizer

from mingpt.model import GPT
from transform.model_tce import GPT as GPT_tce, generate_b, special_conformal_transform
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

from itertools import chain

import math
import random

# ------------------------------------------------------------------
# Seeds & Hyperparameters
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Utility: ComposedTCE wrapper
# ------------------------------------------------------------------
class ComposedTCE(nn.Module):
    """
    Composite model: simulates f_{b1+b2}(x0) using two trained TCE models.
    """
    def __init__(self, T1, T2):
        super().__init__()
        self.T1 = T1
        self.T2 = T2

        assert T1.block_size == T2.block_size
        assert T1.transformer.wte.embedding_dim == T2.transformer.wte.embedding_dim
        assert T1.lm_head.out_features == T2.lm_head.out_features

        self.block_size = T1.block_size
        self.vocab_size = T1.lm_head.out_features
        self.n_embd = T1.transformer.wte.embedding_dim

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # embeddings
        tok_emb = self.T1.transformer.wte(idx)
        pos_emb = self.T1.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # forward through f_b1, f_b2
        x = special_conformal_transform(x, self.T1.conformal_b)
        x = special_conformal_transform(x, self.T2.conformal_b)

        # blocks of T1
        for block in self.T1.transformer.h:
            x = block(x)
        # blocks of T2
        for block in self.T2.transformer.h:
            x = block(x)

        # inverse transforms
        x = special_conformal_transform(x, -self.T2.conformal_b)
        x = special_conformal_transform(x, -self.T1.conformal_b)

        # final projection (we use T1 head here)
        x = self.T1.transformer.ln_f(x)
        logits = self.T1.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ------------------------------------------------------------------
# Dataset preparation
# ------------------------------------------------------------------
dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

tokenizer = GPT2Tokenizer.from_pretrained(model_type)
tokenizer.model_max_length = int(1e9)
vocab_size = len(tokenizer)

def tokenize_function(examples):
    return tokenizer([t + tokenizer.eos_token for t in examples["text"]],
                     add_special_tokens=False,
                     return_attention_mask=False)

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

train_dataset = wikiDataset(lm_train)
val_dataset = wikiDataset(lm_val)

rng = random.Random(3407) 
indices = rng.sample(range(len(val_dataset)), 2000)
val_subset = Subset(val_dataset, indices)

val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,  batch_size=batch_size, num_workers=num_workers)
val_loader_small = DataLoader(val_subset, shuffle=False, pin_memory=True,  batch_size=batch_size, num_workers=num_workers)

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
    
# ------------------------------------------------------------------
# Train function
# ------------------------------------------------------------------
def train_tce(b_seed, ckpt_path):
    from copy import deepcopy
    model_config = GPT_tce.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = vocab_size
    model_config.block_size = block_size

    model = GPT_tce(model_config).to(device)
    # override conformal_b with seed
    model.conformal_b = generate_b(b_seed, model_config.n_embd).to(device)

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
    trainer.run()

    payload = {
        "config": deepcopy(trainer.config),
        "model_state_dict": trainer.model.state_dict(),
        "step_num": trainer.step_num
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(payload, ckpt_path)
    return model

# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Train T1 and T2
    ckpt_dir = "/home/jan/gptTransform/minGPTTransform/savedModel"
    ckpt_T1 = os.path.join(ckpt_dir, "T1_b42.pt")
    ckpt_T2 = os.path.join(ckpt_dir, "T2_b2100.pt")

    T1 = train_tce(42, ckpt_T1)
    T2 = train_tce(2100, ckpt_T2)

    # Evaluate T1 and T2
    loss1, ppl1 = evaluate(T1, val_loader)
    loss2, ppl2 = evaluate(T2, val_loader)

    # Build composed model
    combo = ComposedTCE(T1, T2).to(device)
    lossC, pplC = evaluate(combo, val_loader)

    # Generation example
    prompt = "Artificial intelligence in modern age"
    x0 = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    y1 = T1.generate(x0, max_new_tokens=30, do_sample=True, top_k=40)
    y2 = T2.generate(x0, max_new_tokens=30, do_sample=True, top_k=40)
    yc = combo.generate(x0, max_new_tokens=30, do_sample=True, top_k=40)

    print("\n=== Evaluation Results ===")
    print(f"T1 (b=42):     loss={loss1:.4f}, ppl={ppl1:.2f}")
    print(f"T2 (b=2100):   loss={loss2:.4f}, ppl={ppl2:.2f}")
    print(f"Composed:      loss={lossC:.4f}, ppl={pplC:.2f}")

    print("\n=== Sample Generations ===")
    print("T1:", tokenizer.decode(y1[0].cpu().squeeze()))
    print("T2:", tokenizer.decode(y2[0].cpu().squeeze()))
    print("Composed:", tokenizer.decode(yc[0].cpu().squeeze()))

    # Save results to file
    results_path = os.path.join(ckpt_dir, "results_composed.txt")
    with open(results_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write(f"T1 (b=42):     loss={loss1:.4f}, ppl={ppl1:.2f}\n")
        f.write(f"T2 (b=2100):   loss={loss2:.4f}, ppl={ppl2:.2f}\n")
        f.write(f"Composed:      loss={lossC:.4f}, ppl={pplC:.2f}\n\n")
        f.write("=== Sample Generations ===\n")
        f.write("T1:\n" + tokenizer.decode(y1[0].cpu().squeeze()) + "\n\n")
        f.write("T2:\n" + tokenizer.decode(y2[0].cpu().squeeze()) + "\n\n")
        f.write("Composed:\n" + tokenizer.decode(yc[0].cpu().squeeze()) + "\n\n")
