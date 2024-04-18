#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from transformers import AutoModel, AutoConfig
from src.common import Triaffine, init_esim_weights, FusionGate, NewFusionGate
from openhgnn.models import HAN

import torch
import torch.nn as nn
from itertools import accumulate


class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)

        self.dense_layers = nn.ModuleDict({
            'ent': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 2 * 4),
            'rel': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 5),
        })

        cfg.inner_dim_sub = cfg['inner_dim_sub_{}'.format(cfg.lang)]
        self.dense_layers_1 = nn.ModuleDict({
            'rel': nn.Linear(bert_config.hidden_size, cfg.inner_dim_sub * 3 * 5),
        })

        init_esim_weights(self.dense_layers)
        init_esim_weights(self.dense_layers_1)

        self.triaffine = Triaffine(cfg.inner_dim_sub, 1, bias_x=True, bias_y=False)

        h_graph = []
        
        cfg.category = 'tk'

        cfg.meta_paths_dict = {
            'sp-rep': [('tk', 'spk', 'tk'), ('tk', 'rep', 'tk')],
            'rep-sp': [('tk', 'rep', 'tk'), ('tk', 'spk', 'tk')],
            'sp': [('tk', 'spk', 'tk')],
            'rep': [('tk', 'rep', 'tk')],
            'self': [('tk', 'self', 'tk')],
        }
        cfg.hidden_dim = bert_config.hidden_size
        cfg.out_dim = bert_config.hidden_size
        cfg.num_heads = [cfg.num_head0]
        self.han = HAN.build_model_from_args(cfg, h_graph)
        init_esim_weights(self.han)

        if cfg.fusion_type == 'gate0':
            self.fusion = FusionGate(bert_config.hidden_size)
        else:
            self.fusion = NewFusionGate(bert_config.hidden_size)
        init_esim_weights(self.fusion)

        self.cfg = cfg 
    
    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.cfg.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_ro_embedding(self, qw, kw, token_index, token2sents, pos_type):
        if pos_type == 1:
            pos_emb = []
            for i in range(token2sents.shape[0]):
                p = self.custom_sinusoidal_position_embedding(token2sents[i], pos_type)
                pos_emb.append(p)
            pos_emb = torch.stack(pos_emb)
        else:
            position = torch.arange(0, len(token_index[0]), dtype=torch.long, device=self.cfg.device)
            pos_emb = self.custom_sinusoidal_position_embedding(position, pos_type).unsqueeze(0)

        x_cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        x_sin_pos = pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        cur_qw2 = cur_qw2.reshape(qw.shape)
        cur_qw = qw * x_cos_pos + cur_qw2 * x_sin_pos

        y_cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        y_sin_pos = pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        cur_kw2 = cur_kw2.reshape(kw.shape)
        cur_kw = kw * y_cos_pos + cur_kw2 * y_sin_pos

        bsize, qlen, num, dim = cur_qw.shape

        new_kw = cur_kw.permute(0, 2, 1, 3).contiguous().view(bsize * num, qlen, dim)
        new_qw = cur_qw.permute(0, 2, 1, 3).contiguous().view(bsize * num, qlen, dim)

        return new_kw, new_qw

    def classify_matrix(self, kwargs, sequence_outputs, mat_name='ent'):
        utterance_index, token_index = kwargs['utterance_index'], kwargs['token_index']

        token2sents = kwargs['token2sents']

        outputs = self.dense_layers[mat_name](sequence_outputs)

        if mat_name  == 'rel':
            outputs1 = self.dense_layers_1[mat_name](sequence_outputs)

        q_token, k_token, q_utterance, k_utterance = 0, 0, 0, 0
        q1, q2, q3 = 0, 0, 0
        if mat_name == 'ent':
            num = 2
            outputs = torch.split(outputs, self.cfg.inner_dim * num, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            q_token, k_token = torch.split(outputs, self.cfg.inner_dim, dim=-1)
        elif mat_name == 'rel':
            num = 4
            outputs = torch.split(outputs, self.cfg.inner_dim * num, dim=-1)
            outputs = torch.stack(outputs, dim=-2)
            q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.cfg.inner_dim, dim=-1)

            num = 3
            outputs1 = torch.split(outputs1, self.cfg.inner_dim_sub * num, dim=-1)
            outputs1 = torch.stack(outputs1, dim=-2)
            q1, q2, q3 = torch.split(outputs1, self.cfg.inner_dim_sub, dim=-1)
            sp0, sp1, sp2, sp3 = q1.shape
            q1 = q1.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
            q2 = q2.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
            q3 = q3.permute(0, 2, 1, 3).contiguous().view(-1, sp1, sp3)
        tk_qw, tk_kw = self.get_ro_embedding(q_token, k_token, token_index, token2sents, pos_type=0) # pos_type=0 for token-level relative distance encoding

        ut_qw, ut_kw = 0, 0
        if mat_name != 'ent':
            ut_qw, ut_kw = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, token2sents, pos_type=1) # pos_type=1 for utterance-level relative distance encoding

        return tk_qw, tk_kw, ut_qw, ut_kw, q1, q2, q3

    def get_loss(self, kwargs, logits, input_labels, mat_name):

        nums = logits.shape[-1]
        masks = kwargs['sentence_masks'] if mat_name == 'ent' else kwargs['full_masks']
        criterion = nn.CrossEntropyLoss(logits.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss
    
    def conduct_triffine(self, qw, kw, q1, q2, q3):
        tri_scores = self.triaffine(q1, q2, q3)
        bi_scores = torch.einsum('bmd,bnd->bmn', qw, kw).contiguous()

        rate = bi_scores
        if self.cfg.soft == 'soft':
            K1 = torch.einsum('bij,bijk->bik', rate, tri_scores.softmax(2))
            K2 = torch.einsum('bjk,bijk->bik', rate, tri_scores.softmax(2))
        else:
            K1 = torch.einsum('bij,bijk->bik', rate, tri_scores)
            K2 = torch.einsum('bjk,bijk->bik', rate, tri_scores)

        K_score = K1 + K2 + bi_scores

        return K_score
    
    def merge_sentence(self, sequence_outputs, dialogue_length):
        res = []
        for i, w in enumerate(dialogue_length):
            res.append(sequence_outputs[i, :w])
        res = torch.cat(res, 0)
        return res
    
    def split_sentence(self, sequence_outputs, dialogue_length):
        bsize = len(dialogue_length)
        res = sequence_outputs.new_zeros([bsize, max(dialogue_length), sequence_outputs.shape[-1]])
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            res[i, :e-s] = sequence_outputs[s:e]
        return res

    def forward(self, **kwargs):
        input_ids, input_masks, input_segments, hgraphs = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments', 'hgraphs']]

        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]

        graph_output = self.merge_sentence(sequence_outputs, kwargs['dialogue_length'])
        h_dict = {'tk': graph_output}
        graph_output = self.han(hgraphs, h_dict)['tk']

        graph_output = self.split_sentence(graph_output, kwargs['dialogue_length'])

        sequence_outputs = self.fusion(sequence_outputs, graph_output)

        mat_names = ['ent', 'rel']
        losses, tags = [], []
        bsize, qlen = sequence_outputs.shape[:2]

        for i in range(len(mat_names)):
            mat_name = mat_names[i]
            tkp, tkq, utp, utq, q1, q2, q3 = self.classify_matrix(kwargs, sequence_outputs, mat_names[i])
            input_labels = kwargs[f"{mat_names[i]}_matrix"]
            if mat_name == 'ent':
                logits = torch.einsum('bmd,bnd->bmn', tkp, tkq).contiguous()
            else:
                logits0 = self.conduct_triffine(tkp, tkq, q1, q2, q3)
                logits1 = torch.einsum('bmd,bnd->bmn', utp, utq).contiguous()
                logits = logits1 + logits0
            
            num = logits.shape[0] // bsize
            logits = logits.view(bsize, num, qlen, qlen).permute(0, 2, 3, 1).contiguous()
            
            loss = self.get_loss(kwargs, logits, input_labels, mat_names[i])
            losses.append(loss)
            tags.append(logits)

        return losses, tags 
