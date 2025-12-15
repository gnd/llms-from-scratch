import os
import torch
import tiktoken
from tokenizer import TokenizerV1
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    
    def __init__(self, all_ids, context_length, stride):
        self.input_ids = []
        self.target_ids = []
        self.all_ids = all_ids

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# some variables
source_dir = "sources"
#tokenizer = TokenizerV1()
tokenizer = tiktoken.get_encoding("o200k_base")
all_ids = []

def import_sources(source_dir):
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                print(f"Opening {filepath} ..")
                raw_text = f.read() + " <|endoftext|>"
                all_ids.extend(tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"}))

    # Save the generated vocabulary
    # Only TokenizerV1
    #tokenizer.save_vocab("objects/vocab.json")

    # test
    # Only TokenizerV1
    #print(f"All words: {len(tokenizer.str_to_int)}")


# import source texts from sources dir into proprocessed_texts array
import_sources(source_dir)

# print all_ids size
print(f"All ids: {len(all_ids)}")

# test
# tokenizer.load_vocab("objects/vocab.json")
ids = tokenizer.encode("Hello, do you like tea? <|endoftext|>", allowed_special={"<|endoftext|>"})
print(ids)

# test
print(tokenizer.decode(ids))

# generate input output pairs
context_size = 128
stride = 32
dataset = GPTDatasetV1(all_ids, 128, 32)