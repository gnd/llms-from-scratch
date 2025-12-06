import re
import json

class TokenizerV1:
    def __init__(self):
        self.str_to_int = {}
        self.int_to_str = {}

    def encode(self, text):
        ids = []
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed_text = [item.strip("<>:=@").strip() for item in preprocessed_text if item.strip()]
        for token in preprocessed_text:
            if token not in self.str_to_int:
                self.str_to_int[token] = len(self.str_to_int)+1
            else:
                ids.append(self.str_to_int[token])
        self.int_to_str = {i:s for s,i in self.str_to_int.items()}
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

    def save_vocab(self, file):
        with open(file, "w") as f:
            json.dump(self.str_to_int, f, indent=2)

    def load_vocab(self, file):
        with open(file, "r") as f:
            self.str_to_int = json.load(f)
            self.int_to_str = {i:s for s,i in self.str_to_int.items()}