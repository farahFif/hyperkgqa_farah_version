import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.init import xavier_normal_
from abc import ABC, abstractmethod
import random
from qamodels.base_qamodel import Base_QAmodel
from transformers import DistilBertModel, DistilBertConfig
from sentence_transformers import SentenceTransformer
from transformers import RobertaConfig, RobertaModel

class ENC_DEC_QAmodel(Base_QAmodel):

    def __init__(self, args, model, vocab_size, rel2idx,entity2idx,device):

        super(ENC_DEC_QAmodel, self).__init__(args, model, vocab_size)
        self.rel2idx = rel2idx
        self.device = device
        self.rel2idx['<start>'] = len(self.rel2idx)
        self.rel2idx['<end>'] = len(self.rel2idx)
        self.idx2rel = {v:k for k,v in rel2idx.items()}
        self.entity2idx = entity2idx
        self.idx2entity = {v:k for k,v in entity2idx.items()}

        # Encoder
        self.ln_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.hidden_dim = 384
        self.lin_dim = 1024
        self.lin1 = nn.Linear(self.hidden_dim, self.lin_dim)
        # Decoder
        self.units = 1024
        self.vocab_rel_size = len(self.rel2idx)
        self.embedding = nn.Embedding(self.vocab_rel_size, self.hidden_dim)
        self.gru = nn.LSTM(self.hidden_dim + self.units,
                          self.units,
                          batch_first=True)
        self.loss_ = torch.nn.KLDivLoss(reduction='sum')
        self.loss_crp = nn.CrossEntropyLoss()
        self.fc = nn.Linear(args.dim, self.vocab_rel_size)
        self.last_fc = nn.Linear( self.units,args.dim)
        self.gru2 = nn.LSTM(args.dim, self.units)
        self.enc_fc = nn.Linear(self.units,args.dim)
        #self.hidden = self.initialize_hidden_state(self.device,)
        self.batch_norm = nn.BatchNorm1d(args.dim)


    def apply_nonLinear(self, input):
        pass

    def get_question_embedding(self, question, question_len):
        pass

    def loss_rel(self,real, pred):
        mask = real.ge(1).type(torch.cuda.FloatTensor)
        loss = self.loss_crp(pred, real) * mask
        return torch.mean(loss)

    def loss(self, scores, targets):
        k = self.loss_(F.log_softmax(scores, dim=1),F.normalize(targets, p=1, dim=1))
        return k

    def decoder_forward(self, input ,context_vector):
        x = self.embedding(input)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.gru(x)
        output = output.view(-1, output.size(2))
        x2 =  self.last_fc(output)
        x2 = self.batch_norm(x2)
        x = self.fc(x2)
        #return x , x2
        return x ,x2

    def encode_again(self,x):
        #print('xxxxx',x.shape)
        x = x.view(x.shape[0],1, x.shape[1])
        #print('xxxxx2222', x.shape)
        # print('emmmmb',x.shape)
        output, _ = self.gru2(x)
        output = output.view(-1, output.size(2))
        output = self.enc_fc(output)
        return output

    def initialize_hidden_state(self, batch_sz, device):
        return torch.zeros((1, batch_sz, self.units)).to(device)

    def get_encoder_embedding(self, question):
        ln_repr = self.ln_model.encode(question)
        output = self.lin1(torch.tensor(ln_repr).to(self.device))
        #return ln_repr, output
        return output

    def get_predictions(self, question, chains, head, attention_mask):
        pred  = super().get_score_ranked(head, question, chains, attention_mask)
        return pred

    def get_predictions2(self, question, chains, head, attention_mask):
        pred  = super().get_score_ranked2(head, question, chains, attention_mask)
        return pred
    def get_predictions3(self, question, chains, head, attention_mask):
        pred  = super().get_score_ranked3(head, question, chains, attention_mask)
        return pred


    def get_entity_emb(self,idx):
        return super().get_entity_embedding(idx)
    #def train_encoder_decoder(self,loader):


