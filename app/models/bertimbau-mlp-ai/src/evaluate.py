import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report, accuracy_score
from models.bertimbau_mlp import BERTimbauMLP
from data.preprocess import load_and_prepare_data, EMOTIONS
from data.dataset import EmotionDataset

def evaluate_model(model, dataloader, device):
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
    preds = (torch.tensor(preds) > 0.5).int().numpy()
    acc = accuracy_score(true, preds)
    return true, preds, acc

def print_classification_report(true, preds, target_names):
    print(classification_report(true, preds, target_names=target_names, zero_division=0))

def main(model_path, data_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    texts, labels = load_and_prepare_data(data_path)
    dataset = EmotionDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = BERTimbauMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    true, preds, acc = evaluate_model(model, dataloader, device)
    print(f"Accuracy: {acc:.4f}")
    print_classification_report(true, preds, target_names=EMOTIONS)

if __name__ == "__main__":
    main("bertimbau_mlp.pt", "Base_Categorizada.xlsx")