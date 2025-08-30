from collections import Counter
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


def build_vocab(texts, min_freq=2):
    """Build vocabulary from cleaned texts"""
    word_counts = Counter()
    
    for text in texts:
        if pd.isna(text) or not text or str(text) == 'nan':
            continue
            
        text = str(text).lower().strip()
        
        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        tokens = [token for token in tokens if token]
        word_counts.update(tokens)
    
    # Only use basic special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for word, count in word_counts.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab


def collate_fn(batch):
    """Custom collate function for padding sequences"""
    texts, labels = zip(*batch)
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels