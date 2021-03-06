import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_

class LSTM_PruningModel(nn.Module):
    def __init__(self, args, rel2idx, vocab_size, pretrained_language_model):
        super(LSTM_PruningModel, self).__init__()
        self.rank = args.dim
        self.ls = args.labels_smoothing

        self.hidden_dim = 256
        self.bidirectional = True
        self.n_layers = 1
        self.pretrained_language_model = pretrained_language_model
        self.transformer_layer_1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_layer_2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_layer_3 = nn.TransformerEncoderLayer(d_model=512, nhead=8)

        self.GRU = nn.LSTM(self.rank, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_dim = self.hidden_dim * 2        
        self.hidden2rel = nn.Linear(self.hidden_dim, len(rel2idx))
        self.word_embeddings = nn.Embedding(vocab_size, self.rank)
        self.loss = torch.nn.BCELoss(reduction='sum')

    def apply_nonLinear(self, input):
        input = torch.unsqueeze(input, 0)
        output = self.transformer_layer_1(input)
        output = self.transformer_layer_2(output)
        output = self.transformer_layer_3(output)
        output = torch.squeeze(output, 0)
        return self.hidden2rel(output)

    def get_question_embedding(self, question, question_len):
        question_len = question_len.cpu()
        embeds = self.word_embeddings(question)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.pretrained_language_model(packed_output)
        question_embedding = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        return question_embedding        

    def get_score_ranked(self, question, question_len):
        question_embedding = self.get_question_embedding(question, question_len)
        question_embedding = self.apply_nonLinear(question_embedding)
        prediction = torch.sigmoid(question_embedding).squeeze()
        return prediction
        


