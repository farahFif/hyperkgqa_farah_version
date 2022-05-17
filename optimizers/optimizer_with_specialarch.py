"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import pickle
from scipy import spatial
import networkx as nx
from collections import Counter
from utils.utils import get_relations_in_path
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_metric_learning import losses

class QAOptimizer(object):
    """Knowledge Graph embedding model optimizer.
    KGOptimizers performs loss computations for one phase
    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
    """

    def __init__(self, args, model, optimizer, regularizer, dataset, device):
        self.model = model
        self.args = args
        self.hops = args.hops
        self.thresh = 0.5
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_weight = args.reg
        self.dataset = dataset
        self.device = device
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.gamma = args.rel_gamma
        self.use_relation_matching = args.use_relation_matching

    def train(self, loader, epoch):
        running_loss = 0
        for i_batch, a in enumerate(loader):
            #if i_batch == 100 : break
            question = a[0]
            question_param = a[1].to(self.device)
            head = a[2].to(self.device)
            tail = a[3].to(self.device)
            path = a[4].to(self.device)
            #loss_rel , chains = self.calculate_rel_loss(question, path,self.device)
            #loss_qa  = self.calculate_qa_loss(question, chains, head, tail, question_param)
            loss =  self.calculate_new_joint_loss(question, path,head,tail,question_param,self.device)
            #if self.reg_weight > 0:
            #    loss = loss + self.regularizer.forward(self.model.parameters())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, self.max_epochs))
            loader.update()

        return running_loss

    def calculate_new_joint_loss(self,question, path,head,tail,question_param,device):
        loss_rel = 0
        loss_qa = 0
        chains = []
        hidden = self.model.get_encoder_embedding(question)
        dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
        counter = 0
        for t in range(1, path.shape[1]):
            predictions, emb = self.model.decoder_forward(dec_input, hidden)
            loss_rel += self.model.loss_rel(path[:, t].long(), predictions)
            pred = self.model.get_predictions(question, emb, head, question_param)
            ## method 2 use idx instead of emb
            p_tail = tail
            if self.model.ls:
                p_tail = ((1.0 - self.model.ls) * p_tail) + (1.0 / p_tail.size(1))
            loss_qa += self.model.loss(pred, p_tail)

            top_2 = torch.topk(pred, k=2, largest=True, sorted=True)
            pred_ans = top_2[1][:,1]
            #print('preeeed ',pred_ans.shape, pred_ans)
            dec_input = path[:, t].unsqueeze(1)
            pred_entity_vecs = self.model.get_entity_emb(pred_ans)
            #hidden = torch.sum(torch.stack([pred_entity_vecs,hidden],dim=0),dim=0)
            hidden = pred_entity_vecs
            ### try to only use pred ent vecs

        return loss_rel + loss_qa

    def calculate_rel_loss(self, question, path,device):
        """ This one with embedding from model not idx chains
        """
        loss = 0
        chains = []
        #emb_sum = torch.zeros((len(question),self.args.dim),requires_grad =False).to(device)
        hidden = self.model.get_encoder_embedding(question)
        dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
        counter = 0
        for t in range(1, path.shape[1]):
            predictions , emb = self.model.decoder_forward(dec_input, hidden)
            loss += self.model.loss_rel(path[:, t].long(), predictions)
            dec_input = path[:, t].unsqueeze(1)
            if t != 1 and t != path.shape[1] :
                counter += 1
                #emb_sum += emb
                chains.append(emb)
        chains_ = torch.hstack(chains)
        #print('chaaaaains',chains_.shape)
        #chains_ = emb_sum / counter
        return loss , chains_

    def calculate_qa_loss(self, question,chains, head, tail, question_param):
        pred, lhse,rhse = self.model.get_predictions(question,chains, head, question_param)
        p_tail = tail

        if self.model.ls:
            p_tail = ((1.0-self.model.ls)*p_tail) + (1.0/p_tail.size(1))
        #loss = self.model.loss(pred, p_tail)
        loss = 0
        ys = torch.eq(pred,tail).long()
        loss_cont = 0
        return loss , loss_cont

    def predict_sentence(self,quest,tail):
        score = 0
        with torch.no_grad():
            hidden = self.model.get_encoder_embedding(quest).unsqueeze(0)
            next = '<start>'
            res = []
            dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * 1).to(self.device)
            while next != '<end>':
                predictions = self.model.decoder_forward(dec_input, hidden)
                dec_input = predictions.argmax(1).unsqueeze(1)
                next = self.model.idx2rel[predictions.squeeze().argmax().item()]
                if next == '<end>': break
                res.append(dec_input.item())
        if res == tail[1:-1] : score = 1
        return score

    def calculate_valid_loss(self, samples):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0
        predicted_answers = []
        failed_answers = []
        scores = 0
        total_preds = 0
        for i in tqdm(range(len(samples))):
            #if i == 100: break
            d = next(data_gen)
            head = d[0].to(self.device)
            question = d[1]
            ans = d[2]
            chain = d[4]
            attention_mask = d[3].unsqueeze(0).to(self.device)
            hidden = self.model.get_encoder_embedding(question).unsqueeze(0)
            dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * 1).to(self.device)
            next_word = '<start>'
            max_iter = 0
            while next_word != '<end>':
                max_iter +=1
                if max_iter == 4 : break
                predictions, emb = self.model.decoder_forward(dec_input, hidden)
                dec_input = predictions.argmax(1).unsqueeze(1)
                next_word = self.model.idx2rel[predictions.squeeze().argmax().item()]
                if next_word =='<end>': break
                #scores += self.predict_sentence(question,chain)
                scores = self.model.get_score_ranked(head, question, emb, attention_mask)
                top_2 = torch.topk(scores, k=2, largest=True, sorted=True)
                top_2_idx = top_2[1].tolist()[0]

                head_idx = head.tolist()
                if top_2_idx[0] == head_idx:
                    pred_ans = top_2_idx[1]
                else:
                    pred_ans = top_2_idx[0]
                if type(ans) is int:
                    ans = [ans]
                if pred_ans in ans:
                    total_correct += 1

                pred_entity_vecs = self.model.get_entity_emb(torch.tensor([ans[0]]).to(self.device))
                hidden = torch.sum(torch.stack([pred_entity_vecs, hidden], dim=0), dim=0)
                predicted_answers.append(pred_ans)
                total_preds +=1
        #print('CHAIN SCORE ', scores/len(samples))
        accuracy1 = total_correct/len(samples)
        accuracy2 = total_correct/total_preds
        print(accuracy1 , 'acc2 ',accuracy2)
        return accuracy2, predicted_answers



    def compute_score(self, candidate_rels, pruning_rels):
        return len(candidate_rels & pruning_rels) / len(pruning_rels.union(candidate_rels))

    def compute_score_with_relation_matching(self, samples, G, model, dataset, idx2rel):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0

        for i in tqdm(range(len(samples))):
            d = next(data_gen)

            head = d[0].to(self.device)
            question_tokenized = d[1].unsqueeze(0).to(self.device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(self.device)

            scores = self.model.get_score_ranked(head, question_tokenized, attention_mask)
            scores, candidates =  torch.topk(scores, k=50, largest=True)
            scores = scores.squeeze(0).cpu().detach().numpy()
            candidates = candidates.squeeze(0).cpu().detach().numpy()

            if candidates[0] == head:
                candidates = candidates[1:]
                scores = scores[1:]

            question_tokenized, attention_mask = dataset.tokenize_question(samples[i][1].replace('NE', ''))
            question_tokenized = question_tokenized.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            rel_scores = model.get_score_ranked(question_tokenized, attention_mask)

            pruning_rels_scores, pruning_rels_torch = torch.topk(rel_scores, 5)
            pruning_rels = [p.item() for s, p in zip(pruning_rels_scores, pruning_rels_torch) if s > self.thresh]

            pred_ans_initial = candidates[np.argmax(scores)]
            if len(pruning_rels) > 0:
                for j, tail in enumerate(candidates):
                    candidate_rels, _ = get_relations_in_path(G, head.item(), tail)
                    if len(candidate_rels) > 0:
                        candidate_rels_names = [idx2rel[r] for r in candidate_rels]
                        score = self.compute_score(set(candidate_rels), set(pruning_rels))
                        scores[j] = scores[j] + self.gamma*score

            pred_ans = candidates[np.argmax(scores)]

            if type(ans) is int:
                ans = [ans]

            if pred_ans in ans:
                total_correct += 1

        accuracy = total_correct/len(samples)
        print(accuracy)
        return accuracy

    def generate_tail_embed(self, question, path,device):
        with torch.no_grad():
            emb_sum = torch.zeros((len(question),self.args.dim),requires_grad =False).to(device)
            hidden = self.model.get_encoder_embedding(question)
            dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
            counter = 0
            for t in range(1, path.shape[1]):
                predictions , emb = self.model.decoder_forward(dec_input, hidden)
                dec_input = path[:, t].unsqueeze(1)
                if t != 1 and t != path.shape[1] :
                    counter += 1
                    emb_sum += emb
            chains_ = emb_sum / counter
            return chains_

    def get_chains(self,question, path, encoder, decoder, device):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            emb_sum = torch.zeros((len(question), self.args.dim), requires_grad=False).to(device)
            hidden = encoder(question)
            dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
            counter = 0
            for t in range(1, path.shape[1]):
                predictions, emb = decoder(dec_input, hidden)
                if t != 1 and t != path.shape[1]:
                    counter += 1
                    emb_sum += emb
            chains_ = emb_sum / counter
            return chains_



#
# model1_param = []
#         if  epoch == 0 :
#             for poc in range(preheat):
#                 print('Start preheating epoch ' ,poc )
#                 running_loss = 0
#                 for i_batch, a in enumerate(loader):
#                     question = a[0]
#                     question_param = a[1].to(self.device)
#                     head = a[2].to(self.device)
#                     tail = a[3].to(self.device)
#                     path = a[4].to(self.device)
#                     loss_rel , chains = self.calculate_rel_loss(question, path,self.device)
#
#                     self.optimizer.zero_grad()
#                     loss_rel.backward()
#                     self.optimizer.step()
#
#                     running_loss += loss_rel.item()
#                     loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
#                     loader.set_description('{}/{}'.format(epoch, self.max_epochs))
#                     loader.update()
#             model1_param = list(self.model.decoder_forward.parameters())[0]
#             print("End preheating")


    # def train(self, loader,encoder,decoder, epoch):
    #
    #     running_loss = 0
    #     for i_batch, a in enumerate(loader):
    #         question = a[0]
    #         question_param = a[1].to(self.device)
    #         head = a[2].to(self.device)
    #         tail = a[3].to(self.device)
    #         path = a[4].to(self.device)
    #         chains = self.get_chains(question, path,encoder,decoder,self.device)
    #         loss = self.calculate_qa_loss(question, chains, head, tail, question_param)
    #
    #         if self.reg_weight > 0:
    #             loss = loss + self.regularizer.forward(self.model.parameters())
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         running_loss += loss.item()
    #         loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
    #         loader.set_description('{}/{}'.format(epoch, self.max_epochs))
    #         loader.update()
    #
    #     return running_loss

    #
    #
    #
    # def calculate_valid_loss(self, samples,encoder,decoder):
    #     data_gen = self.dataset.data_generator(samples)
    #     total_correct = 0
    #     predicted_answers = []
    #     scores = 0
    #     scores_rel = 0
    #     for i in tqdm(range(len(samples))):
    #         d = next(data_gen)
    #         head = d[0].to(self.device)
    #         question = d[1]
    #         ans = d[2]
    #         chain = d[4]
    #         attention_mask = d[3].unsqueeze(0).to(self.device)
    #         hidden = encoder(question).unsqueeze(0)
    #         dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * 1).to(self.device)
    #         next_word = '<start>'
    #         chains = []
    #         emb_sum = None
    #         count = 0
    #         i = 0 # to ignore start tag embedding
    #         while next_word != '<end>':
    #             predictions,r = decoder(dec_input, hidden)
    #             dec_input = predictions.argmax(1).unsqueeze(1)
    #             next_word = self.model.idx2rel[predictions.squeeze().argmax().item()]
    #             if i != 0 :
    #                 count += 1
    #                 if emb_sum is None:
    #                     emb_sum = r
    #                 else:
    #                     emb_sum += r
    #             i+=1
    #             if next_word =='<end>': break
    #
    #         emb_ = emb_sum / count
    #         #scores_rel += self.predict_sentence(question,chain)
    #         scores = self.model.get_score_ranked(head, question, emb_, attention_mask)
    #         top_2 = torch.topk(scores, k=2, largest=True, sorted=True)
    #         top_2_idx = top_2[1].tolist()[0]
    #         head_idx = head.tolist()
    #         if top_2_idx[0] == head_idx:
    #             pred_ans = top_2_idx[1]
    #         else:
    #             pred_ans = top_2_idx[0]
    #
    #         if type(ans) is int:
    #             ans = [ans]
    #
    #         if pred_ans in ans:
    #             total_correct += 1
    #         if i% 100 == 0:
    #             print('CHAIN SCORE ', scores_rel / i)
    #         predicted_answers.append(pred_ans)
    #     print('CHAIN SCORE ', scores_rel/len(samples))
    #     accuracy = total_correct/len(samples)
    #     print(accuracy)
    #     return accuracy, predicted_answers

    # def calculate_rel_loss(self, question, path,device):
    #    ''' This function returns relation loss and a list of inx of relations
    #    '''
    #     loss = 0
    #     chains = []
    #     hidden = self.model.get_encoder_embedding(question)
    #     dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
    #     for t in range(1, path.shape[1]):
    #         predictions = self.model.decoder_forward(dec_input, hidden)
    #         loss += self.model.loss_rel(path[:, t].long(), predictions)
    #         dec_input = path[:, t].unsqueeze(1)
    #         predictions = predictions.argmax(1).detach().cpu().numpy()
    #         chains.append(predictions)
    #     chains = np.vstack(chains).T
    #     chains_ = [chain[:np.where(chain == self.model.rel2idx['<end>'])[0][0]] if self.model.rel2idx['<end>'] in chain else chain
    #                for chain in chains ]
    #     chains_ = [torch.tensor(c).to(self.device) for c in chains_]
    #     return loss , chains_

    # def calculate_rel_loss(self, question, path,device):
    #     """ This one with embedding from model not idx chains
    #     """
    #     loss = 0
    #     chains = []
    #     #emb_sum = torch.zeros((len(question),self.args.dim),requires_grad =False).to(device)
    #     hidden = self.model.get_encoder_embedding(question)
    #     dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * hidden.shape[0]).to(device)
    #     counter = 0
    #     for t in range(1, path.shape[1]):
    #         predictions , emb = self.model.decoder_forward(dec_input, hidden)
    #         loss += self.model.loss_rel(path[:, t].long(), predictions)
    #         dec_input = path[:, t].unsqueeze(1)
    #         if t != 1 and t != path.shape[1] :
    #             counter += 1
    #             #emb_sum += emb
    #             chains.append(emb)
    #     chains_ = torch.hstack(chains)
    #     #print('chaaaaains',chains_.shape)
    #     #chains_ = emb_sum / counter
    #     return loss , chains_

    # def calculate_valid_loss(self, samples):
    #     data_gen = self.dataset.data_generator(samples)
    #     total_correct = 0
    #     predicted_answers = []
    #     failed_answers = []
    #     scores = 0
    #     for i in tqdm(range(len(samples))):
    #         #if i == 100: break
    #         d = next(data_gen)
    #         head = d[0].to(self.device)
    #         question = d[1]
    #         ans = d[2]
    #         chain = d[4]
    #         attention_mask = d[3].unsqueeze(0).to(self.device)
    #         hidden = self.model.get_encoder_embedding(question).unsqueeze(0)
    #         dec_input = torch.tensor([[self.model.rel2idx['<start>']]] * 1).to(self.device)
    #         next_word = '<start>'
    #         chains = []
    #         while next_word != '<end>':
    #             predictions = self.model.decoder_forward(dec_input, hidden)
    #             dec_input = predictions.argmax(1).unsqueeze(1)
    #             next_word = self.model.idx2rel[predictions.squeeze().argmax().item()]
    #             if next_word =='<end>': break
    #             chains.append(dec_input)
    #
    #         chains = [torch.tensor(c).to(self.device) for c in chains]
    #         #scores += self.predict_sentence(question,chain)
    #         scores = self.model.get_score_ranked(head, question, chains, attention_mask)
    #         top_2 = torch.topk(scores, k=2, largest=True, sorted=True)
    #         top_2_idx = top_2[1].tolist()[0]
    #
    #         head_idx = head.tolist()
    #         if top_2_idx[0] == head_idx:
    #             pred_ans = top_2_idx[1]
    #         else:
    #             pred_ans = top_2_idx[0]
    #         if type(ans) is int:
    #             ans = [ans]
    #         if pred_ans in ans:
    #             total_correct += 1
    #
    #         predicted_answers.append(pred_ans)
    #     #print('CHAIN SCORE ', scores/len(samples))
    #     accuracy = total_correct/len(samples)
    #     print(accuracy)
    #     return accuracy, predicted_answers

