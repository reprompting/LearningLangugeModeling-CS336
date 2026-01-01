import os 
import regex as re 
from collections import defaultdict
from typing import BinaryIO
import pickle 
from tqdm import tqdm  

def save_vocab_and_merges(vocab, merges_list, file_name = "trained_model.pkl"):
    with open (file_name, "wb") as f:  # Fixing 'Wb' to 'wb' for binary write mode
        pickle.dump({'vocab': vocab, 'merges': merges_list}, f)
    print(f"Model saved to {file_name}")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

desired_num_chunks = 500
regex_syntax = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
special_token = b"<|endoftext|>"
file_path = "TinyStories-train.txt" 
local_word_counts = defaultdict(int)
pair_counts = defaultdict(int)
new_word_counts = defaultdict(int)
global_word_counts = defaultdict(int)
new_token_int = 257
target_vocab_size = 2048

with open(file_path, "rb") as f:
    """
        -> opens file in binary mode
        -> sends the opened file to function that finds chunk boundaries
        -> returns chunked data with divides, each at special token
    """
    chunked_data = find_chunk_boundaries(f, desired_num_chunks, special_token)

for start, end in zip(chunked_data[:-1], chunked_data[1:]):
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end-start).decode("utf-8")

for part in text.split("<|endoftext|>"):
    for m in re.finditer(regex_syntax, part):
        tok = tuple(m.group().encode("utf-8"))
        global_word_counts[tok] += 1  

def train(global_word_counts, new_token_int, target_vocab_size):
    """
    -> initialize vocab with all single-byte tokens (0-255)
    -> initialize empty list of merges
    -> repeat until target vocab size is reached:
        -> count all adjacent byte pairs in the current tokenized text
        -> find the most frequent pair
        -> if no pairs left, break
        -> add new token to vocab representing the merged pair
        -> store the merge in the merges list
        -> update all words by replacing occurrences of the most frequent pair with the new token
        -> increment new_token_int
    -> return final vocab, merges, updated global word counts, and final token id
    """
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    while new_token_int < target_vocab_size:
        pair_counts = defaultdict(int)

        for word, count in global_word_counts.items():
            for i in range(len(word)-1):
                pair_counts[(word[i], word[i+1])] += count

        if not pair_counts:
            break 

        most_frequent = max(pair_counts, key=pair_counts.get)
        a, b = most_frequent
        merges.append((a, b))
        vocab[new_token_int] = vocab[a] + vocab[b] 

        new_word_counts = defaultdict(int)

        for word, count in global_word_counts.items():
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == most_frequent:
                    new_word.append(new_token_int)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] += count
        global_word_counts = new_word_counts
        new_token_int += 1

        # Show progress of merges using tqdm
        tqdm.write(f"Merged pair: {a}, {b} -> New token ID: {new_token_int-1}")

    return vocab, merges, global_word_counts, new_token_int

def encode(text, merges_list):
    """
    -> split the text into initial tokens (using regex)
    -> convert each token to bytes (tuple of byte values)
    -> for each token:
        - repeatedly apply merges in order:
            -> if a pair in the token matched a merge, replace it with the new token id
    -> output the sequence of token ids

    text: str
    merges: dict mapping tuple (byte pairs) -> new token int
    """
    tokens = []

    for m in re.finditer(regex_syntax, text):
        word = list(m.group().encode("utf-8"))

        # Apply merges in learned order
        for new_token, (a, b) in enumerate(merges_list, start=257):
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        tokens.extend(word)

    return tokens

def decode(token_seq, reverse_merges):
    """
    -> for each token id:
        - look up its byte sequence in vocab
    -> concatenate all byte sequences
    -> decode concatenated bytes into UTF-8 string
    """
    bytes_out = []
    for tok in token_seq:
        if tok in reverse_merges:
            bytes_out.extend(reverse_merges[tok])
        else:
            bytes_out.append(tok)
    return bytes(bytes_out).decode("utf-8", errors="ignore")

def validate(validation_file_path, merges_list, reverse_merges, chunk_size=100_000):
    """
    Validate the model by encoding and decoding the validation file with progress bars.
    """
    print("Starting validation...")

    encoded_all = []
    decoded_chunks = []

    print("Encoding validation text...")
    with open(validation_file_path, "r", encoding="utf-8") as val_file:
        with tqdm(desc="Encoding chunks", unit="chunk") as pbar:
            while True:
                text_chunk = val_file.read(chunk_size)
                if not text_chunk:
                    break

                encoded_chunk = encode(text_chunk, merges_list)
                encoded_all.extend(encoded_chunk)

                pbar.update(1)

    print(f"Total encoded tokens: {len(encoded_all)}")

    print("Decoding validation tokens...")
    with tqdm(total=len(encoded_all), desc="Decoding tokens", unit="tok") as pbar:
        bytes_out = []
        for tok in encoded_all:
            if tok in reverse_merges:
                bytes_out.extend(reverse_merges[tok])
            else:
                bytes_out.append(tok)
            pbar.update(1)

    decoded_text = bytes(bytes_out).decode("utf-8", errors="ignore")

    return decoded_text

if __name__ == "__main__":

    validation_file_path = "TinyStories-valid.txt"  

    vocab, merges_list, global_word_counts, final_token_int = train(global_word_counts, new_token_int, target_vocab_size)
    print(f"Training done! Final token id: {final_token_int}")

    save_vocab_and_merges(vocab, merges_list)

    # converting merges list to dict for faster encoding
    merges = {pair: idx for idx, pair in enumerate(merges_list, start=257)}

    # reverse merges for decoding
    reverse_merges = {v: vocab[v] for v in vocab}

    example_text = "Once upon a time, there was a tiny story.<|endoftext|>  money was no problem"

    encoded = []
    print("Encoding...")
    for token in tqdm([example_text], desc="Encoding tokens"):
        encoded.extend(encode(token, merges))

    print(f"Encoded token ids: {encoded}")

    print("Decoding...")
    decoded_text = decode(encoded, reverse_merges)
    print(f"Decoded text: {decoded_text}")

    print("Validating with validation file...")

    decoded_validation_text = validate(
        validation_file_path, merges_list, reverse_merges
    )

    with open(validation_file_path, "r", encoding="utf-8") as f:
        original_validation_text = f.read()

    print(f"Original Validation Text: {original_validation_text[:500]}...")  # First 500 chars
    print(f"Decoded Validation Text: {decoded_validation_text[:500]}...")  # First 500 chars