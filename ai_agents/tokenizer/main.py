import os
import re
import numpy as np
from collections import Counter

class BPETrainer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        new_corpus = {}
        for word in corpus:
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def fit(self, text):
        words = text.strip().split()
        corpus = Counter([' '.join(list(w) + ['</w>']) for w in words])

        for _ in range(self.vocab_size):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            corpus = self.merge_vocab(best, corpus)

        # Build vocab from final corpus tokens
        idx = 0
        for word in corpus:
            tokens = word.split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1

        # Add unknown token
        self.vocab["<|unk|>"] = idx
        return self.vocab


class TokenizerV2:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def bpe_encode_word(self, word):
        chars = list(word) + ['</w>']
        while True:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            pair_freq = {pair: idx for idx, pair in enumerate(self.merges) if pair in pairs}
            if not pair_freq:
                break
            best = min(pair_freq, key=lambda p: self.merges.index(p))
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best:
                    new_chars.append(''.join(best))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            if len(chars) == 1:
                break
        return chars

    def encode(self, text):
        words = text.strip().split()
        tokens = []
        for word in words:
            subwords = self.bpe_encode_word(word)
            tokens.extend(subwords)
        ids = [self.str_to_int.get(t, self.str_to_int["<|unk|>"]) for t in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.int_to_str.get(i, "<|unk|>") for i in ids]
        words = []
        word = ''
        for token in tokens:
            if token.endswith('</w>'):
                word += token[:-4]
                words.append(word)
                word = ''
            else:
                word += token
        return ' '.join(words)


def one_hot_encode(ids, vocab_size):
    one_hot = np.zeros((len(ids), vocab_size))
    for i, idx in enumerate(ids):
        one_hot[i, idx] = 1
    return one_hot


class LearnedEmbedding:
    def __init__(self, vocab_size, embed_dim=8):
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, embed_dim))

    def get_vector(self, token_id):
        return self.embeddings[token_id]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(embedding, vocab):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embedding.embeddings)
    plt.figure(figsize=(10, 8))
    for token, idx in vocab.items():
        x, y = reduced[idx]
        plt.scatter(x, y, s=25, color='steelblue', alpha=0.7)
        plt.text(x + 0.01, y + 0.01, token, fontsize=8)
    plt.title("Token Embeddings (PCA Projection)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()



with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    text = f.read()

trainer = BPETrainer(vocab_size=150)
vocab = trainer.fit(text)

tokenizer = TokenizerV2(vocab, trainer.merges)
print("Sample learned vocab:", list(vocab.keys())[:20])

ids = tokenizer.encode("Justice delayed is justice denied.")
print("Token IDs:", ids)

decoded = tokenizer.decode(ids)
print("Decoded text:", decoded)

one_hot = one_hot_encode(ids, len(vocab))
embedding = LearnedEmbedding(len(vocab))
vectors = [embedding.get_vector(i) for i in ids]
print("Cosine similarity between first two tokens:", cosine_similarity(vectors[0], vectors[1]))
plot_embeddings(embedding, vocab)