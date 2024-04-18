
import numpy as np
from collections import defaultdict

import os
import random
import torch
import numpy as np
from typing import Callable, Optional
import dgl


import torch
import torch.nn as nn

class WordPair:
    def __init__(self, max_sequence_len=512):
        self.max_sequence_len = max_sequence_len

        self.entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
        self.rel_dic = {"O": 0, "h2h": 1, 'pos': 2, 'neg': 3, 'other': 4}

    def encode_entity(self, elements, entity_type='ENT-T'):
        '''
        Convert the elements in the dataLoader to a list of entities rel_list.
        The format is [(starting position, ending position, entity type in the dictionary)].
        '''
        entity_list = []
        for line in elements:
            start, end = line[:2]
            entity_list.append((start, end, self.entity_dic[entity_type]))
        return entity_list
    
    def encode_relation(self, triplets):
        '''
        Convert the triplets in the dataLoader to a list of relations `rel_list`.
        Each relation is represented as a tuple with three elements: the starting position, the ending position, and the relation type in the dictionary.
        '''
        rel_list = []
        for triplet in triplets:
            s_en, e_en, s_as, e_as, s_op, e_op, polar = triplet

            if s_en != -1 and s_as != -1:
                rel_list.append((s_en, s_as, self.rel_dic['h2h']))

            # Add relation from aspect to opinion
            if s_as != -1 and s_op != -1:
                rel_list.append((s_as, s_op, self.rel_dic['h2h']))

            # Add relation from entity to opinion
            if s_en != -1 and s_op != -1:
                assert polar != -1 and polar != 0
                rel_list.append((s_en, s_op, polar + 1)) 

        return rel_list

    def encode_polarity(self, triplets):
        '''
        Convert triplets in the dataLoader to polarity.
        Each polarity is represented as a tuple with three elements: the starting position, the ending position, and the polarity category.
        '''
        rel_list = []
        for triplet in triplets:
            s_en, e_en, s_as, e_as, s_op, e_op, polar = triplet
            # Add head-to-head relations for the quadruples->head_rel_list
            # Add relation entity->opinion
            rel_list.append((s_en, s_op, polar))
            rel_list.append((e_en, e_op, polar))

        return rel_list

    def list2rel_matrix4batch(self, batch_rel_list, seq_len=512):
        '''
        Convert a sentence's relation list to a matrix.
        batch_rel_matrix:[batch_size, seq_len, seq_len]
        '''
        rel_matrix = np.zeros([len(batch_rel_list), seq_len, seq_len], dtype=int)
        for batch_id, rel_list in enumerate(batch_rel_list):
            for rel in rel_list:
                rel_matrix[batch_id, rel[0], rel[1]] = rel[2]
        return rel_matrix.tolist()

    # Decoding section
    def rel_matrix2list(self, rel_matrix):
        '''
        Convert a (512*512) matrix to a list of relations.
        '''
        rel_list = []
        nonzero = rel_matrix.nonzero()
        for x_index, y_index in zip(*nonzero):
            dic_key = int(rel_matrix[x_index][y_index].item())
            rel_elem = (x_index, y_index, dic_key)
            rel_list.append(rel_elem)
        return rel_list
    
    def get_triplets(self, ent_matrix, rel_matrix, token2sents, pol_score):
        ent_list = self.rel_matrix2list(ent_matrix)
        rel_list = self.rel_matrix2list(rel_matrix)
        res, pair = self.decode_triplet(ent_list, rel_list, token2sents, pol_score)
        return res, pair
    
    def decode_triplet(self, ent_list, rel_list, token2sents, pol_score):
        # Entity dictionary, with structure (head: [(tail, relation type)])
        entity_elem_dic = defaultdict(list)
        entity2type = {}
        for entity in ent_list:
            if token2sents[entity[0]] != token2sents[entity[1]]: continue
            entity_elem_dic[entity[0]].append((entity[1], entity[2]))
            entity2type[entity[:2]] = entity[2]
        
        # (boundary,boundary -> polarity) set
        b2b_relation_set = {}
        for rel in rel_list:
            start, end, idx = rel 
            if idx == 1: continue
            score = pol_score[start, end, idx]
            b2b_relation_set[rel[:2]] = (idx, score)
        
        # head2head dictionary, with structure (head1: [(head2, relation type)])
        h2h_entity_elem = defaultdict(list)
        for h2h_rel in rel_list:
            # for each head-to-head relationship, mark its entity as 0
            if h2h_rel[2] == self.rel_dic['h2h']:
                h2h_entity_elem[h2h_rel[0]].append((h2h_rel[1], h2h_rel[2]))
            elif h2h_rel[2] > 0:
                h2h_entity_elem[h2h_rel[0]].append((h2h_rel[1], h2h_rel[2]))
        
        # for all head-to-head relations
        triplets = []
        for h1, values in h2h_entity_elem.items():
            if h1 not in entity_elem_dic: continue
            for h2, rel_tp in values:
                if h2 not in entity_elem_dic: continue
                for t1, ent1_tp in entity_elem_dic[h1]:
                    for t2, ent2_tp in entity_elem_dic[h2]:
                        # if (t1, t2) not in t2t_relation_set: continue
                        triplets.append((h1, t1, h2, t2))

        # if there is a (0,0,0,0) in triplets, remove it
        if (0, 0, 0, 0) in triplets:
            triplets.remove((0, 0, 0, 0))
        
        triplet_set = set(triplets)
        ele2list = defaultdict(list)
        for line in triplets:
            e0, e1 = line[:2], line[2:]
            ele2list[e0].append(e1)
        
        tetrad = []
        for subj, obj_list in ele2list.items():
            for obj in obj_list:
                if obj not in ele2list: continue
                for third in ele2list[obj]:
                    if (*subj, *third) not in triplet_set: continue
                    tp0, score0 = b2b_relation_set.get((subj[0], third[0]), (-1, 0))
                    tp1, score1 = b2b_relation_set.get((subj[1], third[1]), (-1, 0))
                    if tp1 == -1 and tp0 == -1: 
                        tetrad.append((*subj, *obj, *third, 1))
                    elif tp0 == tp1 or tp0 == -1:
                        tetrad.append((*subj, *obj, *third, tp1))
                    elif tp1 == -1:
                        tetrad.append((*subj, *obj, *third, tp0))
                    elif score0 > score1:
                        tetrad.append((*subj, *obj, *third, tp0))
                    else:
                        tetrad.append((*subj, *obj, *third, tp1))
                    
        pairs = {'ta': [], 'to': [], 'ao': []}
        for line in triplets:
            h1, t1, h2, t2 = line
            tp1 = entity2type[(h1, t1)]
            tp2 = entity2type[(h2, t2)]
            if tp1 == 1 and tp2 == 2:
                pairs['ta'].append(line)
            elif tp1 == 2 and tp2 == 3:
                pairs['ao'].append(line)
            elif tp1 == 1 and tp2 == 3:
                pairs['to'].append(line)
        return set(tetrad), pairs

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []
    
    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)
    
    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return self.score[best_id], res

def update_config(config):
    lang = config.lang
    keys = ['json_path']
    for k in keys:
        config[k] = config[k] + '_' + lang
    keys = ['cls', 'sep', 'pad', 'unk', 'bert_path']
    for k in keys:
        config[k] = config['bert-' + config.lang][k]
    return config



class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring :cite:`zhang-etal-2020-efficient,wang-etal-2019-second`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        n_proj (Optional[int]):
            If specified, applies MLP layers to reduce vector dimensions. Default: ``None``.
        dropout (Optional[float]):
            If specified, applies a :class:`SharedDropout` layer with the ratio on MLP outputs. Default: 0.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.
        decompose (bool):
            If ``True``, represents the weight as the product of 3 independent matrices. Default: ``False``.
        init (Callable):
            Callable initialization method. Default: `nn.init.zeros_`.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_proj: Optional[int] = None,
        dropout: Optional[float] = 0,
        scale: int = 0,
        bias_x: bool = False,
        bias_y: bool = False,
        decompose: bool = False,
        init: Callable = nn.init.zeros_
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_proj = n_proj
        self.dropout = dropout
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.decompose = decompose
        self.init = init

        if n_proj is not None:
            self.mlp_x = MLP(n_in, n_proj, dropout)
            self.mlp_y = MLP(n_in, n_proj, dropout)
            self.mlp_z = MLP(n_in, n_proj, dropout)
        self.n_model = n_proj or n_in
        if not decompose:
            self.weight = nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x, self.n_model, self.n_model + bias_y))
        else:
            self.weight = nn.ParameterList((nn.Parameter(torch.Tensor(n_out, self.n_model + bias_x)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model)),
                                            nn.Parameter(torch.Tensor(n_out, self.n_model + bias_y))))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.n_proj is not None:
            s += f", n_proj={self.n_proj}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"
        if self.decompose:
            s += f", decompose={self.decompose}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        if self.decompose:
            for i in self.weight:
                self.init(i)
        else:
            self.init(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if hasattr(self, 'mlp_x'):
            x, y, z = self.mlp_x(x), self.mlp_y(y), self.mlp_z(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len, seq_len]
        if self.decompose:
            wx = torch.einsum('bxi,oi->box', x, self.weight[0])
            wz = torch.einsum('bzk,ok->boz', z, self.weight[1])
            wy = torch.einsum('byj,oj->boy', y, self.weight[2])
            s = torch.einsum('box,boz,boy->bozxy', wx, wz, wy)
        else:
            w = torch.einsum('bzk,oikj->bozij', z, self.weight)
            p = torch.einsum('bozij,byj->boziy', w, y)
            s = torch.einsum('bxi,boziy->bozxy', x, p)
            # s = torch.einsum('bxi,bozij,byj->bozxy', x, w, y)
        return s.squeeze(1) / self.n_in ** self.scale


class MLP(nn.Module):
    r"""
    Applies a linear transformation together with a non-linear activation to the incoming tensor:
    :math:`y = \mathrm{Activation}(x A^T + b)`

    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
        activation (bool):
            Whether to use activations. Default: True.
    """

    def __init__(self, n_in: int, n_out: int, dropout: float = .0, activation: bool = True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.

        Returns:
            A tensor with the size of each output feature `n_out`.
        """
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SharedDropout(nn.Module):
    r"""
    :class:`SharedDropout` differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.

    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.

    Examples:
        >>> batch_size, seq_len, hidden_size = 1, 3, 5
        >>> x = torch.ones(batch_size, seq_len, hidden_size)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p: float = 0.5, batch_first: bool = True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            A tensor with the same shape as `x`.
        """

        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.FloatTensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)

def init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight) # random 
        nn.init.constant_(module.bias.data, 0.0) # 0.0-> 85.45, 0.1-> 85.28
    elif isinstance(module, nn.Conv2d):
        nn.init.uniform_(module.weight.data, -0.1, 0.1) # 81.71
        nn.init.constant_(module.bias.data, 0.0) # 无所谓
    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1) # 81.71

def build_hgraph(token2sents=None, speakers=None, replies=None, dialogue_length=None):
    if token2sents is None:
        token2sents = [0, 1, 2, 3]
        speakers = [0, 1, 0, 1]
        replies = [-1, 0, 1, 2]
        dialogue_length = 4


    token2sents = token2sents[:dialogue_length]
    spk_arr = np.array(speakers)
    ts_arr = np.array(token2sents)

    token2speaker = spk_arr[ts_arr]
    spk_matrix = (token2speaker[:, None] == token2speaker).astype(int)

    rep_arr = np.array(replies)
    token2reply = rep_arr[ts_arr]
    rep_matrix = (token2reply[:, None] == ts_arr).astype(int)

    spk_start, spk_end = spk_matrix.nonzero()
    rep_start, rep_end = rep_matrix.nonzero()

    self_start = np.arange(dialogue_length)
    self_end = np.arange(dialogue_length)

    graph_data = {
        ('tk', 'spk', 'tk'): (spk_start, spk_end),
        ('tk', 'rep', 'tk'): (rep_start, rep_end),
        ('tk', 'self', 'tk'): (self_start, self_end)
    }
    g = dgl.heterograph(graph_data)
    return g


class NewFusionGate(nn.Module):
    def __init__(self, hid_size):
        super(NewFusionGate, self).__init__()
        self.fuse = nn.Linear(hid_size * 2, hid_size)

    def forward(self, a, b):
        # Concatenate a and b along the last dimension
        concat_ab = torch.cat([a, b], dim=-1)
        # Apply the linear layer
        fusion_coef = torch.sigmoid(self.fuse(concat_ab))
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor


class FusionGate(nn.Module):
    def __init__(self, hid_size):
        super(FusionGate, self).__init__()
        self.fuse_weight = nn.Parameter(torch.Tensor(hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fuse_weight)

    def forward(self, a, b):
        fusion_coef = torch.sigmoid(self.fuse_weight)
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor