import torch.nn as nn

class RnnSentimentClassifier(nn.Module):
    def __init__(self,vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super(RnnSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                  batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                  bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)

        # RNN Forward Pass
        rnn_out, hidden = self.rnn(embedded)

        # Use the last hidden state
        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        out = self.dropout(last_hidden)
        # prediction
        out = self.fc(out)

        return out


