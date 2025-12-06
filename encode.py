import os
from tokenizer import TokenizerV1

# some variables
source_dir = "sources"
tokenizer = TokenizerV1()

def import_sources(source_dir):
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                print(f"Opening {filepath} ..")
                raw_text = f.read()
                tokenizer.encode(raw_text)

    # Save the generated vocabulary
    tokenizer.save_vocab("objects/vocab.json")

    # test
    print(f"All words: {len(tokenizer.str_to_int)}")


# import source texts from sources dir into proprocessed_texts array
#import_sources(source_dir)

# test
tokenizer.load_vocab("objects/vocab.json")
ids = tokenizer.encode("Hello, do you like tea?")
print(ids)

# test
print(tokenizer.decode(ids))