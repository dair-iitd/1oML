"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from IPython.core.debugger import Pdb
import copy

#from ste import reg_cardinality, reg_att_sudoku_c1

logger = logging.getLogger(__name__)


def get_sudoku_attention_mask():
    col_indices = torch.arange(9).repeat(9, 1).view(-1)
    row_indices = torch.arange(9).repeat(9, 1).transpose(0, 1).reshape(-1)
    block_row_indices = torch.div(torch.arange(9), 3, rounding_mode='floor').repeat(9, 1).view(-1)
    block_col_indices = torch.div(torch.arange(9), 3, rounding_mode='floor').repeat(9, 1).permute(1,0).reshape(-1)
    block_indices = (3 * block_row_indices + block_col_indices)

    att = torch.zeros(81,81).fill_(float('-inf'))
    for i in range(81):
        for j in range(81):
            if (
                    (col_indices[i] == col_indices[j]) or 
                    (row_indices[i] == row_indices[j]) or 
                    (block_indices[i] == block_indices[j])
                ): 
                att[i,j] = 0  
    return att



class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.C = self.f = self.create_v = self.tok_emb = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.causal_mask = config.causal_mask if hasattr(config, 'causal_mask') else True


        self.apply_sudoku_attention_mask = config.sudoku_attention_mask
        if self.apply_sudoku_attention_mask:
            self.sudoku_attention_mask = get_sudoku_attention_mask()

            


    def forward(self, x, layer_past=None):
        if isinstance(x, tuple):
            x = x[0]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal_mask:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
       
        #if (x.shape[0] == 32) and (T != 81): 
            #Pdb().set_trace()
        if self.apply_sudoku_attention_mask:
            att = att + self.sudoku_attention_mask[:T,:T].to(att.device)
            #att = att + self.sudoku_attention_mask.to(att.device)

        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_to_check


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        
        #TODO check if we need this for cross attention or not
        #self.causal_mask = config.causal_mask if hasattr(config, 'causal_mask') else True


        self.apply_sudoku_attention_mask = config.sudoku_attention_mask
        if self.apply_sudoku_attention_mask:
            self.sudoku_attention_mask = get_sudoku_attention_mask()

            


    def forward(self, x, key_value_states,  layer_past=None):
        if isinstance(x, tuple):
            x = x[0]
        B, T, C = x.size()
        Be,Te,Ce = key_value_states.size()
        assert B == Be and C == Ce

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # from key_value_states 
        k = self.key(key_value_states).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, Te, hs)
        v = self.value(key_value_states).view(B, Te, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, Te, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, Te) -> (B, nh, T, Te)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #if self.causal_mask:
        #    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
       
        #Pdb().set_trace()
        if self.apply_sudoku_attention_mask:
            att = att + self.sudoku_attention_mask[:T,:].to(att.device)

        #Pdb().set_trace()
        att_to_check = att.clone()
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, Te) x (B, nh, Te, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att_to_check



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        # x = x + self.attn(self.ln1(x))
        att, att_to_check = self.attn(self.ln1(x))
        x = x + att
        x = x + self.mlp(self.ln2(x))
        return x, att_to_check

class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    """ has cross attention as well"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.cross_attn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, encoder_hidden_states):
        if isinstance(x, tuple):
            x = x[0]
        # x = x + self.attn(self.ln1(x))

        #x_ = self.ln1(x)
        
        #print("In decoder block: ", x_.shape)

        #if x.shape[1] < 80:
            #Pdb().set_trace()

        att, att_to_check = self.attn(self.ln1(x))

        x = x + att

        #now apply cross attention using x and encoder_hidden_states
        cross_attention_outputs, cross_attn_to_check = self.cross_attn(
                                    self.ln3(x),
                                    key_value_states = encoder_hidden_states 
        )     

        x = x + cross_attention_outputs 

        x = x + self.mlp(self.ln2(x))
        return x, att_to_check


class GPTEncoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__()

        # input embedding stem
        #if config.tok_emb:
        #    self.tok_emb = config.tok_emb(config=config)
        #else:
        #    self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.tok_emb = token_embeddings
        self.pos_emb = position_embeddings
        
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=False)
        self.losses = config.losses
        self.all_layers = config.all_layers
        self.n_recur = config.n_recur
        self.hyper = config.hyper
        self.C = config.C
        self.f = config.f
        self.create_v = config.create_v

        self.block_size = config.block_size
        self.test = {
            'n_recur[cross,uec]': False,
            'n_layer[uec_last]': False
            }
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        logger.info("number of trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.wt_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None, for_reward=False):
        """
        Returns:
            the loss as a scalar
            the logits in the final prediction; (batch_size, 81, 9)
            the attention for the 1st data in a batch; (n_layer * n_recur, num_heads, 81, 81)
        """
        b, t = idx.shape[0], idx.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # collect the attention matrices and predicted logits
        atts = []
        logits = []
        hidden_state = None

        for _ in range(self.n_recur):
            for block in self.blocks:
                x, att_to_check = block(x) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                atts.append(att_to_check)
                if self.all_layers and ((targets is not None) or for_reward) :
                    hidden_state = self.ln_f(x)
                    #logits.append(self.head(self.ln_f(x)))
                    logits.append(self.head(hidden_state))
        if not self.all_layers or ((targets is None) and (not for_reward)):
            hidden_state = self.ln_f(x)
            logits.append(self.head(hidden_state))
            #logits.append(self.head(self.ln_f(x)))

        #logits are of shape: batch size x 81 x 10
        
        logits = torch.stack(logits, dim=0)
        
        #logits.shape = (num_layers-1) x batch_size x 81 x 10
        
        #Pdb().set_trace()
        logits = logits.permute(1,3,2,0)


        return hidden_state, logits

 
        
        # compute losses
#        loss = 0
#        if targets is not None:
#            # 1. compute losses on predictions
#            for logit in logits: # (batch_size, 81, 9)
#                loss += F.cross_entropy(logit.reshape(-1, logit.size(-1)), targets.view(-1))
#                # compute the constraint losses
#                if 'c1' in self.losses:
#                    probs = torch.nn.functional.softmax(logit, dim=-1) # (batch_size, 81, 9)
#                    probs = probs.view(-1,9,9,9) # (batch_size, 9, 9, 9)
#                    L_c1 = reg_cardinality(probs.permute(0,3,2,1).reshape(-1,9), num=1) + \
#                        reg_cardinality(probs.permute(0,3,1,2).reshape(-1,9), num=1) + \
#                        reg_cardinality(probs.reshape((-1,3,3,3,3,9)).permute(0,5,1,3,2,4).reshape(-1,9), num=1)
#                    loss += L_c1 * self.hyper[0]
#
#            # 2. compute losses on attentions
#            for att in atts: # (batch_size, num_heads, 81, 81) for Sudoku
#                if 'att_c1' in self.losses:
#                    att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81) # (batch_size * num_heads, 81, 81)
#                    loss += reg_att_sudoku_c1(att_p) * self.hyper[1]
#
#        atts = torch.stack(atts) # (n_layer * n_recur, batch_size, num_heads, 81, 81)
#        atts = F.softmax(atts, dim=-1)
#
#        # compute loss for unlabeled data
#        if idx_ulb is not None:
#            # forward the GPT model
#            token_embeddings = self.tok_emb(idx_ulb) # each index maps to a (learnable) vector
#            x = self.drop(token_embeddings + position_embeddings)
#            # collect the attention matrices and predicted logits
#            atts_ulb = []
#            logits_ulb = []
#            for _ in range(self.n_recur):
#                for block in self.blocks:
#                    x, att_to_check = block(x) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
#                    atts_ulb.append(att_to_check)
#                    if self.all_layers:
#                        logits_ulb.append(self.head(self.ln_f(x)))
#            if not self.all_layers:
#                logits_ulb.append(self.head(self.ln_f(x)))
#
#            # 1. compute losses on predictions
#            for logit in logits_ulb: # (batch_size, 81, 9)
#                if 'c1' in self.losses:
#                    probs = torch.nn.functional.softmax(logit, dim=-1) # (batch_size, 81, 9)
#                    probs = probs.view(-1,9,9,9) # (batch_size, 9, 9, 9)
#                    L_c1 = reg_cardinality(probs.permute(0,3,2,1).reshape(-1,9), num=1) + \
#                        reg_cardinality(probs.permute(0,3,1,2).reshape(-1,9), num=1) + \
#                        reg_cardinality(probs.reshape((-1,3,3,3,3,9)).permute(0,5,1,3,2,4).reshape(-1,9), num=1)
#                    loss += L_c1 * self.hyper[0]
#
#            # 2. compute losses on attentions
#            for att in atts_ulb: # (batch_size, num_heads, 81, 81)
#                if 'att_c1' in self.losses:
#                    att_p = F.softmax(att, dim=-1).reshape(-1, 81, 81) # (batch_size * num_heads, 81, 81)
#                    loss += reg_att_sudoku_c1(att_p) * self.hyper[1]
#        
#        return logits[-1], loss, atts[:,0,...].detach().cpu()



class GPTDecoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, token_embeddings, position_embeddings
            ):
        super().__init__()
        assert config.causal_mask

        # input embedding stem
        self.tok_emb = token_embeddings 
        self.pos_emb = position_embeddings 
        # nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.num_classes, bias=False)
        self.losses = config.losses
        self.all_layers = config.all_layers
        self.n_recur = config.n_recur
        self.hyper = config.hyper
        self.C = config.C
        self.f = config.f
        self.create_v = config.create_v

        self.block_size = config.block_size
        self.test = {
            'n_recur[cross,uec]': False,
            'n_layer[uec_last]': False
            }
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        logger.info("number of trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.wt_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, encoder_state,  targets=None, for_reward=False):
        """
        Returns:
            the loss as a scalar
            the logits in the final prediction; (batch_size, 81, 9)
            the attention for the 1st data in a batch; (n_layer * n_recur, num_heads, 81, 81)
        """
        b, t = idx.shape[0], idx.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # collect the attention matrices and predicted logits
        atts = []
        logits = []
        #Pdb().set_trace()
        return_all = (targets is not None) or (for_reward)
        for _ in range(self.n_recur):
            for block in self.blocks:
                x, att_to_check = block(x, encoder_state) # (batch_size, 81, 128) (batch_size, num_heads, 81, 81)
                atts.append(att_to_check)
                if self.all_layers and return_all:
                    logits.append(self.head(self.ln_f(x)))
        if not self.all_layers or (not return_all):
            logits.append(self.head(self.ln_f(x)))

        #logits are of shape: batch size x 81 x 10
        
        logits = torch.stack(logits, dim=0)
        #logits.shape = (num_layers-1) x batch_size x 81 x 10
        
        #Pdb().set_trace()
        logits = logits.permute(1,3,2,0)
        #logits.shape = batch_size x 10 x 81 x (num_layers)
        return logits



class MyGPTEncoderDecoderSudokuSolver(nn.Module):
    def __init__(self, args):
        super(MyGPTEncoderDecoderSudokuSolver, self).__init__()
        #python main.py --all_layers --n_layer 1 --n_recur 32 --n_head 4 --epochs 200 --eval_interval 1 --lr 0.001 --dataset satnet --gpu 0

        mconf = GPTConfig(vocab_size=12, block_size=81, n_layer=1, n_head=4, n_embd=128, 
                         num_classes=10, causal_mask=False, losses=[], n_recur=args.sudoku_num_steps, all_layers=True,
                                 hyper=[1, 0.1], decoder_start_token_id = 10)
        
        mconf.sudoku_attention_mask = args.sudoku_attention_mask 
        self.encoder_model_config = mconf 
        
        
        
        self.tok_emb = nn.Embedding(self.encoder_model_config.vocab_size, self.encoder_model_config.n_embd)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.encoder_model_config.block_size, self.encoder_model_config.n_embd))
        #self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.encoder_stack = GPTEncoder(mconf, self.tok_emb, self.position_embeddings)
       
        self.decoder_config = copy.deepcopy(mconf)
        self.decoder_config.causal_mask = True
        #self.decoder_start_token_id = 10
        self.decoder_stack  = GPTDecoder(self.decoder_config, self.tok_emb, self.position_embeddings)
        
        self.args= args
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_config.decoder_start_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        #shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


    def get_optimizer(self, args):
        return self.model.configure_optimizers(args)

    @torch.no_grad()
    def generate(self, x, for_reward=False):
        batch_size = x.shape[0]
        B = batch_size

        encoder_hidden_states, encoder_logits = self.encoder_stack(x, for_reward=for_reward)
        labels = torch.zeros(B, 1, dtype=torch.long, device=x.device) 
        labels = torch.zeros(B, 1, dtype=torch.long, device=x.device) + self.decoder_config.decoder_start_token_id 


        
        for _ in range(self.decoder_config.block_size):
            decoder_logits = self.decoder_stack(
                labels,
                encoder_hidden_states,
                for_reward=for_reward
                )
            #[:,:,-1] #batch_size x seq length x num layers
             
            #Pdb().set_trace()
            #run inference on only the last seq
            top_labels = decoder_logits[:,:, -1, -1].argmax(-1).unsqueeze(-1)
            labels = torch.cat([labels, top_labels], dim=-1)
    
        #Pdb().set_trace()
        if for_reward:
            if self.args.loss_on_encoder:
                return torch.cat([encoder_logits, decoder_logits], dim = -1) 
            else:
                return decoder_loigts

        return decoder_logits



    def forward(self, x, target=None, is_training=False, for_reward=False):
        x = x.view(-1,81)
        batch_size = x.shape[0]

        if (not for_reward) and (is_training) and (target is not None):
            target = target.view(-1, 81)
            encoder_hidden_states, encoder_logits = self.encoder_stack(x,target)
           
            #Pdb().set_trace()
            shifted_target = self._shift_right(target)
            decoder_logits = self.decoder_stack(shifted_target, encoder_hidden_states, target) 
                       
            #logits.shape = batch_size x 10 x 81 x (num_layers)
            if self.args.loss_on_encoder:
                logits = torch.cat([encoder_logits, decoder_logits], dim=-1)
            else:
                logits = decoder_logits
            
        else:
            with torch.no_grad():
                self.eval()
                logits = self.generate(x, for_reward)
                self.train() 
        return logits

