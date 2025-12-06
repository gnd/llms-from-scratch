import os
import tiktoken
from tokenizer import TokenizerV1

# some variables
source_dir = "sources"
#tokenizer = TokenizerV1()
tokenizer = tiktoken.get_encoding("o200k_base")

def import_sources(source_dir):
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                print(f"Opening {filepath} ..")
                raw_text = f.read() + " <|endoftext|>"
                tokenizer.encode(raw_text)

    # Save the generated vocabulary
    # Only TokenizerV1
    #tokenizer.save_vocab("objects/vocab.json")

    # test
    # Only TokenizerV1
    #print(f"All words: {len(tokenizer.str_to_int)}")


# import source texts from sources dir into proprocessed_texts array
#import_sources(source_dir)

# test
# tokenizer.load_vocab("objects/vocab.json")
ids = tokenizer.encode("Hello, do you like tea? <|endoftext|>", allowed_special={"<|endoftext|>"})
print(ids)

# test
print(tokenizer.decode(ids))