import torch
import torch.nn as nn


from transformers import DistilBertModel, DistilBertConfig
from sentence_transformers import SentenceTransformer
from transformers import RobertaConfig, RobertaModel


class Encoder(nn.Module):
    def __init__(self,device):
        super(Encoder, self).__init__()
        self.ln_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.hidden_dim = 384
        self.lin_dim = 1024
        self.lin1 = nn.Linear(self.hidden_dim, self.lin_dim)
        self.device = device

    def forward(self, x):
        ln_repr = self.ln_model.encode(x)
        output = self.lin1(torch.tensor(ln_repr).to(self.device))
        return output

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(self.device)


class Decoder(nn.Module):
    def __init__(self, args ,rel2idx):
        super(Decoder, self).__init__()
        self.units = 1024
        self.hidden_dim = 384
        self.vocab_rel_size = len(rel2idx)
        self.embedding = nn.Embedding(self.vocab_rel_size, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim + self.units,
                          self.units,
                          batch_first=True)

        self.fc = nn.Linear(args.dim, self.vocab_rel_size)
        self.last_fc = nn.Linear(self.units, args.dim)

    def forward(self, x, context_vector):
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.gru(x)
        output = output.view(-1, output.size(2))
        x2 = self.last_fc(output)
        x = self.fc(x2)
        return x, x2

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))