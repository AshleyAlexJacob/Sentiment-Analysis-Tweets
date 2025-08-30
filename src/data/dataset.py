from torch.utils.data import Dataset
import torch
import re
import pandas as pd


class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def tokenize(self, text):
        """Simple tokenization for cleaned text"""
        if pd.isna(text) or not text or str(text) == 'nan':
            return []
        
        text = str(text).lower().strip()
        
        # Remove extra whitespace and non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        return [token for token in tokens if token]  # Remove empty tokens

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # tokenize the text
        tokens = self.tokenize(text) # [token1, token2, ...]
        # ['unk'] ['pad']
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
        
    