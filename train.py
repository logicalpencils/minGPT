import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import GPT2Tokenizer
import json
import time
import pickle
from mingpt.utils import set_seed
set_seed(3407)

class JSONLDataset(Dataset):
  def __init__(self, file_path):
    self.file_path = file_path
    self.data = []
    with open(file_path, "r") as f:
      for line in f:
        self.data.append(json.loads(line.strip()))

    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    return self.data[idx]

  def get_vocab_size(self):
    return len(self.tokenizer)

  def get_block_size(self):
    return 512

file_path = '/home/bms22386/minGPT/data.jsonl'
print('Making dataset...')
train_dataset = JSONLDataset(file_path)

# create a GPT instance
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# create a Trainer object
from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 2000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)


def batch_end_callback(trainer, model, t):
  if trainer.iter_num % 100 == 0:
    print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    torch.save(model,"checkpoints/take_"+str(t))
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
