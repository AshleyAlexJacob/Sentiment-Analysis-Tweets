import torch
import torch.nn as nn
from src.data.load_data import load_csv_data
from src.data.dataset import TwitterSentimentDataset
from src.utils import build_vocab, collate_fn, train_model, evaluate_model, save_model_and_vocab
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from src.models.rnn import RnnSentimentClassifier
import mlflow
from datetime import datetime
import dagshub

def main():

    experiment_name = "Twitter-Sentiment-Analysis"
    _ = mlflow.set_experiment(experiment_name)


    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = load_csv_data('data/processed/processed.csv')

    # For quick testing
    # data = data[:100]

    texts = data['cleaned_text'].tolist()
    labels = data['labels'].tolist()

    # Build or load
    vocab = build_vocab(texts, min_freq=2)

    # split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )


    # Create dataset
    train_dataset = TwitterSentimentDataset(X_train, y_train, vocab, max_length=50)

    val_dataset = TwitterSentimentDataset(X_val, y_val, vocab, max_length=50)

    # Create dataloaders

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Further processing and training logic would go here
    print(f"Loaded Training Samples {len(train_loader)} Batches.")
    print(f"Loaded Validation Samples {len(val_loader)} Batches.")
    
    # model declaration
    model = RnnSentimentClassifier(
        vocab_size= len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_classes= 2,
        num_layers=2,
    )
    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(device)

    # start training and validation loop

    with mlflow.start_run(run_name=f"{datetime.isoformat(datetime.now())}") as run:
        mlflow.log_params({
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "vocab_size": len(vocab),
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "criterion": "CrossEntropyLoss",
            "optimizer": "Adam"
        })
        for epoch in range(NUM_EPOCHS):
            best_val_acc = 0.0
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            # Save the model and vocab after training
            if val_acc>best_val_acc:
                best_val_acc = val_acc
                save_model_and_vocab(model, vocab, 'models/rnn_sentiment_model.pth', 'models/vocab.pkl')
                print("Model and vocabulary saved.")
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                print("Validation accuracy did not improve.")


if __name__ == "__main__":
    dagshub.init(repo_owner='alexjacob260', repo_name='Sentiment-Analysis-Tweets', mlflow=True)

    main()