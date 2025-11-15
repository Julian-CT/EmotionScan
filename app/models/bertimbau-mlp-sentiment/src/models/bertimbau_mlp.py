import torch
import torch.nn as nn
from transformers import BertModel

class BERTimbauMLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BERTimbauMLP, self).__init__()
        self.bert = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3),  # 3 classes: NEGATIVO, NEUTRO, POSITIVO
        )
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return self.classifier(pooled)