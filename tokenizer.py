import re
import os

# some variables
source_dir = "sources"

# some globals
preprocessed_texts = []

# import source texts from sources dir into proprocessed_texts array
for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)

    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
        	print(f"Opening {filepath} ..")
        	raw_text = f.read()
        	result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        	preprocessed_texts.extend([item.strip("<>:=@").strip() for item in result if item.strip()])

# generate a vocabulary
all_words = sorted(set(preprocessed_texts))
print(f"All words: {len(all_words)}")
vocab = {token:integer for integer,token in enumerate(all_words)}
