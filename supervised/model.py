#Add a MLP to the SaProt's model class
from transformers import EsmTokenizer, EsmForMaskedLM
from torch import nn
import torch
from torch.nn import functional as F

class SaProtMEP(nn.Module):
    # SaProt model with a Multi-Layer Perceptron for regression task
    def __init__(self, model_path, num_classes=1):
        super(SaProtMEP, self).__init__()
        self.esm = EsmForMaskedLM.from_pretrained(model_path)
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        x = esm_output['last_hidden_state'][:,0]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(EmbeddingMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
	
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x.squeeze(-1)
        
