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
import pickle
from utils.utils import *
from scipy.spatial import distance
from numba import jit
import asyncio
import time
from joblib import Parallel, delayed


class Base_QAmodel(nn.Module):

    def __init__(self, args, model, vocab_size):
        super(Base_QAmodel, self).__init__()
        self.emb_model = model
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.ls = args.labels_smoothing
        self.freeze = args.freeze
        self.relation_dim = args.dim
        self.rank = args.dim
        self.word_embeddings = nn.Embedding(vocab_size, self.rank)
        self.emb_model_name = model.__class__.__name__
        self.hyperbolic_layers = False
        self.context_layer = False
        self.hyperbolic_models = ['RefH', 'RotH', 'AttH']
        if self.emb_model_name in self.hyperbolic_models:
            self.hyperbolic_layers = True

        if self.emb_model_name == 'AttH':
            self.context_layer = True

    @abstractmethod
    def apply_nonLinear(self, input):
        pass

    @abstractmethod
    def get_question_embedding(self, question, question_len):
        pass

    @abstractmethod
    def calculate_valid_loss(self, samples):
        pass

    @abstractmethod
    def calculate_loss(self, question, head, tail, question_len):
        pass

    def get_score_ranked2(self, head, question,chains, question_param):
        #question_embedding = self.get_question_embedding(question, question_param)
        #question_embedding, hyperbolic_layers = self.apply_nonLinear(question_embedding)
        relation_embedding = []
        for chain in chains :
            print('chaiiiiin ',chain , chains)
            if len(chain) == 1:
                print('hnaa')
                s = self.emb_model.embeddings[1](chain)
                s = s.reshape([s.shape[-1]])
                relation_embedding.append(s)
            if len(chain) > 1 :
                t = [self.emb_model.embeddings[1](rel) for rel in chain]
                relation_embedding.append(torch.stack(t, dim=0).sum(dim=0))
            if len(chain) == 0 :
                s = self.emb_model.embeddings[1](torch.tensor([11],device='cuda:0'))
                s = s.reshape([s.shape[-1]])
                relation_embedding.append(s)

        relation_embedding = torch.stack(relation_embedding,dim=0)
        #print(head.shape , 'RELLLLLL',relation_embedding.shape)
        lhs_e = self.emb_model.get_queries(head, relation_embedding)
        rhs_e = self.emb_model.get_rhs()
        scores = self.emb_model.similarity_score(lhs_e, rhs_e)
        #print('LHSEEEEEE ',lhs_e.shape,rhs_e.shape)
        ### return other vectors for contrastive loss
        return scores

    def get_score_ranked3(self, head,chain):
        #question_embedding = self.get_question_embedding(question, question_param)
        #question_embedding, hyperbolic_layers = self.apply_nonLinear(question_embedding)
        s = self.emb_model.embeddings[1](chain)
        #s = s.reshape([s.shape[-1]])
        #print(head.shape , 'RELLLLLL',relation_embedding.shape)
        lhs_e = self.emb_model.get_queries(head, s)
        rhs_e = self.emb_model.get_rhs()
        scores = self.emb_model.similarity_score(lhs_e, rhs_e)
        #print('LHSEEEEEE ',lhs_e.shape,rhs_e.shape)
        ### return other vectors for contrastive loss
        return scores


    def get_score_ranked(self, head, question,chains, question_param):
        #question_embedding = self.get_question_embedding(question, question_param)
        #question_embedding, hyperbolic_layers = self.apply_nonLinear(question_embedding)
        #relation_embedding =  torch.stack(chains,dim=0)
        #print('NWEEEEEE',relation_embedding.shape)
        lhs_e = self.emb_model.get_queries(head, chains)
        rhs_e = self.emb_model.get_rhs()
        scores = self.emb_model.similarity_score(lhs_e, rhs_e)
        return scores

    def get_entity_embedding(self,idx):
        res = [self.emb_model.embeddings[0](i) for i in idx]
        return torch.stack(res,dim=0)


    def get_score_ranked_complex(self,head, path, entity2idx):

        head = head.unsqueeze(0)
        beg = head.item()
        candidates = [i for i in range(len(entity2idx))]
        candidates = np.array(candidates)
        scores = torch.zeros(1).to(self.device)
        all_entities = torch.from_numpy(candidates).to(self.device)
        for p in range(1,len(path)-1):
            s1 = self.get_score_ranked3(head,torch.tensor([path[p]]).to(self.device))
            s1[:,head] = - float('inf')
            scores = scores.unsqueeze(1) + s1
            scores = scores.max(0)[0]
            selection = scores > 0 if scores.max(0)[0] > 0 else scores.argmax(0)
            #print('****',all_entities.shape,selection.shape,scores.shape)

            head = all_entities[selection]
            scores = scores[selection]


            # remove duplicates
            # combined = np.array([[c, all_scores[i]] for i, c in enumerate(all_candidates)])
            # combined = combined[combined[:, 1].argsort()[::-1]]
            # unique_keys, indices = np.unique(combined[:, 0], return_index=True)
            # combined = combined[indices]
            # all_scores = combined[:, 1]
            # all_candidates = np.array([int(c) for c in combined[:, 0]])

                # k = 10 if len(all_scores) >= 10 else len(all_scores)
                # k = len(all_scores)
                # all_scores = torch.tensor(all_scores)
                # all_candidates = np.array(all_candidates)
            if len(scores) >= 50 : k = 50
            else : k= len(scores)
            scores, idx = torch.topk(scores, k=k, largest=True)
            head = head[idx]
            # for i, h in enumerate(head):
            #    print(h.item(), ' (', idx2entity[h.item()], ')  ', "{:.2f}".format(final_scores[i].item()))
            # print('********************************************')
        return scores, head

    def get_score_ranked_complex_withfaiss(self,head, path, entity2idx):

        rhs_e = self.emb_model.get_rhs()
        head = head.unsqueeze(0)
        beg = head.item()
        candidates = [i for i in range(len(entity2idx))]
        final_scores = [0]
        candidates.remove(beg)
        candidates = np.array(candidates)

        for p in range(1,len(path)-1):
            relation_embedding = self.emb_model.embeddings[1](torch.tensor([path[p]]).to(device='cuda:0'))

            all_scores = []
            all_candidates = []

            # fetch per head
            for i, h in enumerate(head):
                lhs_e = self.emb_model.get_queries(h, relation_embedding)
                scores = self.emb_model.similarity_score(lhs_e, rhs_e) + final_scores[i]
                print('score shape ',scores.shape)
                scores = scores.squeeze(0).cpu().detach().tolist()
                scores.pop(beg)  # exclude head
                scores = np.array(scores)
                if (scores > 0).any():
                    tp_candidates = candidates[scores > 0]
                    scores = scores[scores > 0]
                else:  # select top 1
                    idx = np.argmax(scores)
                    scores = np.array([scores[idx]])
                    tp_candidates = np.array([candidates[idx]])

                k = 50 if len(scores) >= 50 else len(scores)
                idx = np.argpartition(scores, -k)[-k:]
                scores = scores[idx]
                tp_candidates = tp_candidates[idx]
                # print(scores, tp_candidates)
                all_scores.extend(scores)
                all_candidates.extend(tp_candidates)
            # print('############################################################################')

            all_scores = np.array(all_scores)
            all_candidates = np.array(all_candidates)

            # remove duplicates
            combined = np.array([[c, all_scores[i]] for i, c in enumerate(all_candidates)])
            combined = combined[combined[:, 1].argsort()[::-1]]
            unique_keys, indices = np.unique(combined[:, 0], return_index=True)
            combined = combined[indices]
            all_scores = combined[:, 1]
            all_candidates = np.array([int(c) for c in combined[:, 0]])

            # k = 10 if len(all_scores) >= 10 else len(all_scores)
            # k = len(all_scores)
            # all_scores = torch.tensor(all_scores)
            # all_candidates = np.array(all_candidates)
            final_scores = torch.tensor(all_scores)
            final_scores, idx = torch.topk(final_scores, k=final_scores.shape[0], largest=True)
            idx = idx.cpu().detach().numpy()
            head = torch.tensor(all_candidates[idx]).to(self.device)

            # for i, h in enumerate(head):
            #    print(h.item(), ' (', idx2entity[h.item()], ')  ', "{:.2f}".format(final_scores[i].item()))
            # print('********************************************')
        return final_scores, head


    # def get_score_ranked2(self, head, question,chains, question_param):
    #     #question_embedding = self.get_question_embedding(question, question_param)
    #     #question_embedding, hyperbolic_layers = self.apply_nonLinear(question_embedding)
    #     relation_embedding = []
    #
    #     s = self.emb_model.embeddings[1](chains)
    #     s = s.reshape([s.shape[-1]])
    #     print('sssssssss ',s.shape)
    #     lhs_e = self.emb_model.get_queries(head, s)
    #     rhs_e = self.emb_model.get_rhs()
    #     scores = self.emb_model.similarity_score(lhs_e, rhs_e)
    #     #print('LHSEEEEEE ',lhs_e.shape,rhs_e.shape)
    #     ### return other vectors for contrastive loss
    #     return scores
