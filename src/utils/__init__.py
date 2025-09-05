from collections import Counter
import os
import pickle
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

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


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch, (batch_texts, batch_labels) in enumerate(train_loader):
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
        print(f"[Train]: Batch {batch+1}/{len(train_loader)}")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch,(batch_texts, batch_labels) in enumerate(test_loader):
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            print(f"[Validation]: Batch {batch+1}/{len(test_loader)}")
        
            
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def save_model_and_vocab(model, vocab, model_path='sentiment_model.pth', vocab_path='vocab.pkl'):
    """Save trained model and vocabulary"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Save vocabulary
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vocabulary saved to {vocab_path}")

def load_vocab(vocab_size, model_path='sentiment_model.pth', vocab_path='vocab.pkl'):
    """Load trained model and vocabulary"""
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    
    return vocab

def predict_sentiment(model, vocab, text, device):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Tokenize and convert to indices
    tokens = text.lower().split()
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if token]
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Convert to tensor
    text_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.softmax(output, dim=1)
    
    sentiment = "Positive" if predicted.item() == 1 else "Negative"
    confidence = probabilities[0][predicted].item()
    
    return sentiment, confidence
