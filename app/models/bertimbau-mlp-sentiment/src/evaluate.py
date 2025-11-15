import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
from models.bertimbau_mlp import BERTimbauMLP
from data.preprocess import load_and_prepare_data, SENTIMENTS
from data.dataset import EmotionDataset

def evaluate_model(model, dataloader, device):
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
    return true, preds, acc

def print_classification_report(true, preds, target_names):
    print(classification_report(true, preds, target_names=target_names, zero_division=0))

def main(model_path, data_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts, labels = load_and_prepare_data(data_path)
    dataset = EmotionDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model = BERTimbauMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("\nAvaliando modelo...")
    true, preds, acc = evaluate_model(model, dataloader, device)
    print(f"\nAcurácia: {acc:.4f}")
    print("\nRelatório de classificação:")
    print_classification_report(true, preds, target_names=SENTIMENTS)
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, preds)
    print("\nMatriz de confusão:")
    print("Real \\ Predito")
    for i, row in enumerate(cm):
        print(f"{SENTIMENTS[i]:8s}: {row}")

if __name__ == "__main__":
    main("bertimbau_sentiment_best.pt", "Base_Categorizada.xlsx")