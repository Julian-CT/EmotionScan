import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from models.bertimbau_mlp import BERTimbauMLP
from data.preprocess import load_and_prepare_data, EMOTIONS
from data.dataset import EmotionDataset

def print_label_distribution(labels, split_name):
    arr = np.array(labels)
    print(f"\nLabel distribution for {split_name}:")
    for idx, emo in enumerate(EMOTIONS):
        print(f"{emo}: {arr[:, idx].sum()}")

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Print outputs and labels for the first batch of the first epoch
        if i == 0:
            print("\nSample outputs (first batch):", outputs[:2].detach().cpu().numpy())
            print("Sample labels (first batch):", labels[:2].detach().cpu().numpy())
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device, threshold=0.5):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            preds.extend(outputs.cpu().numpy())
            true.extend(labels.cpu().numpy())
    preds = (np.array(preds) > threshold).astype(int)
    print(classification_report(true, preds, target_names=EMOTIONS, zero_division=0))
    acc = accuracy_score(true, preds)
    return true, preds, acc

def main(file_path, epochs=3, batch_size=16, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts, labels = load_and_prepare_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    print_label_distribution(y_train, "train")
    print_label_distribution(y_test, "test")
    train_data = EmotionDataset(X_train, y_train, tokenizer)
    test_data = EmotionDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    model = BERTimbauMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
        print("Evaluation on test set:")
        true, preds, acc = eval_model(model, test_loader, device, threshold=0.5)
        print(f"Accuracy: {acc:.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), "bertimbau_mlp.pt")

if __name__ == "__main__":
    main("Base_Categorizada.xlsx")