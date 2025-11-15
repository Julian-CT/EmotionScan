import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from models.bertimbau_mlp import BERTimbauMLP
from data.preprocess import load_and_prepare_data, SENTIMENTS
from data.dataset import EmotionDataset

def print_label_distribution(labels, split_name):
    arr = np.array(labels)
    unique, counts = np.unique(arr.argmax(axis=1), return_counts=True)
    print(f"\nDistribuição de rótulos ({split_name}):")
    for idx, count in zip(unique, counts):
        print(f"{SENTIMENTS[idx]}: {count}")

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        # Convert one-hot to class indices for CrossEntropyLoss
        target_classes = torch.argmax(labels, dim=1)
        loss = loss_fn(logits, target_classes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i == 0:
            print("\nPrimeiro batch:")
            print("Logits:", logits[:2].detach().cpu().numpy())
            print("Labels:", target_classes[:2].detach().cpu().numpy())
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            # Get predicted class (max logit)
            pred_classes = torch.argmax(logits, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            preds.extend(pred_classes.cpu().numpy())
            true.extend(true_classes.cpu().numpy())
    acc = accuracy_score(true, preds)
    print(classification_report(true, preds, target_names=SENTIMENTS, zero_division=0))
    return true, preds, acc

def main(file_path, epochs=5, batch_size=16, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts, labels = load_and_prepare_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.2, 
        random_state=42,
        stratify=[np.argmax(label) for label in labels]  # stratify by sentiment
    )
    
    print_label_distribution(y_train, "treino")
    print_label_distribution(y_test, "teste")
    
    train_data = EmotionDataset(X_train, y_train, tokenizer)
    test_data = EmotionDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    model = BERTimbauMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()  # Changed from BCELoss to CrossEntropyLoss
    
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Loss treino: {train_loss:.4f}")
        print("\nAvaliação conjunto de teste:")
        true, preds, acc = eval_model(model, test_loader, device)
        print(f"Acurácia: {acc:.4f}")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "bertimbau_sentiment_best.pt")
            print(f"Novo melhor modelo salvo (acc: {acc:.4f})")

    print(f"\nTreinamento finalizado. Melhor acurácia: {best_acc:.4f}")

if __name__ == "__main__":
    main("Base_Categorizada.xlsx")