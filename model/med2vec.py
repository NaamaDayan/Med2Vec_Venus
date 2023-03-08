import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

__all__ = ['Med2Vec']

class Med2Vec(BaseModel):
    def __init__(self, icd9_size, demographics_size=0, embedding_size=500, hidden_size=100, device='cpu'):
        super(Med2Vec, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.demographics_size = demographics_size
        self.hidden_size = hidden_size
        self.vocabulary_size = icd9_size
        self.embedding_demo_size = self.embedding_size + self.demographics_size
        self.embedding_w = torch.nn.Parameter(torch.FloatTensor(self.embedding_size, self.vocabulary_size).to(device), requires_grad=True)
        torch.nn.init.uniform_(self.embedding_w, a=-0.1, b=0.1)
        self.embedding_b = torch.nn.Parameter(torch.FloatTensor(1, self.embedding_size).to(device), requires_grad=True)
        self.embedding_b.data.fill_(0)
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size).to(device)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(self.embedding_demo_size, self.hidden_size).to(device)
        self.probits = nn.Linear(self.hidden_size, self.vocabulary_size).to(device)
        self.probits.requires_grad_()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, d=torch.FloatTensor([])):
        x = x.float()
        emb = F.relu(self.embedding_w)
        x = self.embedding_layer(x.long())
        x = self.relu1(x)
        if self.demographics_size:
            x = torch.cat((x, d), dim=1)
        x = self.linear(x)
        x = self.relu2(x)
        probits = self.probits(x)
        probits = probits[:, 0, :]
        return probits, emb