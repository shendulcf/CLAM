import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_normal_
# import utils.utils
# from topk import SmoothTop1SVM
from nystrom_attention import NystromAttention
from performer_pytorch import SelfAttention as PerformerAttention
from memory_efficient_attention_pytorch import Attention as MemoryEfficientAttention
from FlashAttention2.attention import FlashAttention
from flash_pytorch import FLASH
from utils.utils_myself import initialize_weights

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# def attention(self,q,k,v): # requires q,k,v to have same dim
#         B, N, C = q.shape
#         attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).reshape(B, N, C)
#         return x

class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., project_out = False):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim,inner_dim, bias = False)
        self.to_k = nn.Linear(dim,inner_dim, bias = False)
        self.to_v = nn.Linear(dim,inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k,v):
        print(q.shape)
        b, n, _, h = *q.shape, self.heads
        q, k, v = self.to_q(q),self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v]) 

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # Q*K/d^-0.5
        # mask_value = -torch.finfo(dots.dtype).max

        # attn = dots.softmax(dim=-1)
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn
    
class SelfAttention(MultiheadAttention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def forward(self, x):
        return super().forward(x,x,x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, project_out = True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            out,attn=self_attn(x)
            x = out + x
            x = ff(x) + x
            attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x
            
class TransformerEncoder_Nystorm(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads=heads, dropout=dropout, num_landmarks=dim//2, pinv_iterations = 6, residual=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            # out,attn=self_attn(x) # 改了注意力，所以不能直接输出attn
            out = self_attn(x)
            x = out + x
            x = ff(x) + x
            # attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x
        
class TransformerEncoder_PerformerAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PerformerAttention(dim = dim, heads=heads, dim_head=dim_head, causal = False)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            # out,attn=self_attn(x) # 改了注意力，所以不能直接输出attn
            out = self_attn(x)
            x = out + x
            x = ff(x) + x
            # attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x

class TransformerEncoder_MemoryEfficientAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MemoryEfficientAttention(dim = dim, heads=heads, dim_head=dim_head, causal = False, k_bucket_size=1024)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            # out,attn=self_attn(x) # 改了注意力，所以不能直接输出attn
            out = self_attn(x)
            x = out + x
            x = ff(x) + x
            # attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x    

class TransformerEncoder_FLASH(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FLASH(dim = dim, group_size=256,causal = True, query_key_dim=128, expansion_factor = 2, laplace_attn_fn=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            # out,attn=self_attn(x) # 改了注意力，所以不能直接输出attn
            out = self_attn(x)
            x = out + x
            x = ff(x) + x
            # attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x     
        

class TransformerEncoder_FLASHAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FlashAttention(dim=dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attn=False):
        
        attn_list=list()
        for self_attn, ff in self.layers:
            # out,attn=self_attn(x) # 改了注意力，所以不能直接输出attn
            out = self_attn(x)
            x = out + x
            x = ff(x) + x
            # attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x  
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    depth: number of transformer_encoder layers
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

class TransformerMIL_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", depth = 2,  dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(TransformerMIL_SB, self).__init__()

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        # self.instance_loss_fn = SmoothTop1SVM(num_class, tau = tau).cuda() tau = 0.7
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        ## size[0]: input_dim size[1]: hidden_dim
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        initialize_weights(self)
        ################################################################
        ## self_add
        self.cls_token = nn.Parameter(torch.rand(1,1,size[1]))
        self.transformer = TransformerEncoder_FLASHAttention(size[1], depth, 8, 64, 2048, 0.1) # mlp_dim = 2048 一般取4*dim增强模型的表达能力
        self.projector = nn.Linear(1024, size[1])
        self.dropout = nn.Dropout(0.1)
        self.attention = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        # 这是用于计算注意力权重的两个线性层，输入是隐藏层表示H，输出经过激活函数处理后得到注意力的两个部分
        self.attention_V2 = nn.Sequential(
            nn.Linear(size[1], size[1]),
            nn.Tanh()
        )
        self.attention_U2 = nn.Sequential(
            nn.Linear(size[1], size[1]),
            nn.Sigmoid()
        )
        self.attention_weights2 = nn.Linear(size[1], 1) # 该线性层用于将注意力的两个部分相乘并降维为一个值，用于后续的Softmax操作
        ################################################################
        # initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        
        self.projector = self.projector.to(device)
        # self.cls_token = self.cls_token.to(device)
        self.transformer =self.transformer.to(device)
        self.dropout = self.dropout.to(device)
        self.attention = self.attention.to(device)
        self.attention_V2 = self.attention_V2.to(device)
        self.attention_U2 = self.attention_U2.to(device)
        self.attention_weights2 =  self.attention_weights2.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, instance_feature, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[1] < self.k_sample:
            self.k_sample = A.shape[1]
        ## top k
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        # top_p = [instance_feature[i] for i in top_p_ids]
        # top_p = torch.cat(top_p, dim = 1).squeeze(0)
        ## top -k
        top_n_ids = torch.topk(-A, self.k_sample)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        # top_n = [instance_feature[i] for i in top_n_ids]
        # top_n = torch.cat(top_n, dim = 1).squeeze(0)
        ## 伪标签
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h,instance_feature, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[1] < self.k_sample:
            self.k_sample = A.shape[1]
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        # top_p = [instance_feature[i] for i in top_p_ids]
        # top_p = torch.cat(top_p, dim = 1).squeeze(0)
        ################################################################
        # ## self add top-k 
        # top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        # # top_n = torch.index_select(h, dim=0, index=top_n_ids)
        # top_n = [instance_feature[i] for i in top_n_ids]
        # top_n = torch.cat(top_n, dim = 1).squeeze(0)
        # n_targets = self.create_negative_targets(self.k_sample, device)
        ################################################################
        p_targets = self.create_negative_targets(self.k_sample, device)
        
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    def forward(self, xs, label=None, instance_eval=False, return_features=False, attention_only=False, use_cluster=False):
        if use_cluster:
        ################################################################
        # ----> add cluster 
            device = xs.device          
            H = []
            instance_feature = []
            # print('----------------')
            # print(xs.size())
            
            
            for x in xs:
                mask = x != 0
                mask = mask.any(dim=1)
                x = x[mask]
                x = self.projector(x) # 2048 -> 512 delete,clam 1024 -> 512
                # x = torch.cat((self.cls_token, x), dim=1)
                x = x.unsqueeze(0)
                x = torch.cat((self.cls_token, x.cpu()), dim=1)
                x = x.to(device)
                x = self.dropout(x)
                rep = self.transformer(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim
                # rep = rep.squeeze(0)
                H.append(rep[:, 0]) # class_token
                instance_feature.append(rep[:, 1:])
            H = torch.cat(H) # B,10,512
            ## MOMA cls_token attention 
            A_V = self.attention_V2(H)  # NxD
            A_U = self.attention_U2(H)  # NxD
            A = self.attention_weights2(A_V * A_U) # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            if attention_only:
                return A
            A_raw = A
            A = F.softmax(A, dim=1)  # softmax over N

            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets =[]
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()# binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item() # 用于将张量或张量中的元素转换为Python标量
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, H, instance_feature, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, H, instance_feature, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss
                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)
            ################################################################
            # return villare
            # print(A.shape)
            # print(H.shape)
            M = torch.mm(A,H) # KxL
            # print(M.shape)
            logits_t = self.classifiers(M)
            Y_hat_t = torch.topk(logits_t, 1, dim=1)[1]
            Y_prob_t = F.softmax(logits_t, dim = 1)
            # return logit, total_inst_loss, A_raw
            ################################################################
            # # ----> 使用cls_token 进行分类
            # print(f'HHHHHHHHH = {H.size()}')
            # logits_t = self.classifiers(H)
            # print(logits_t.size())
            # Y_hat_t = torch.topk(logits_t, 1, dim=1)[1]
            # Y_prob_t = F.softmax(logits_t, dim = 1)
            ################################################################
            ## 使用原来的AttentionMIL进行分类
            # M = torch.mm(A, h) 
            # logits = self.classifiers(M)
            # Y_hat = torch.topk(logits, 1, dim = 1)[1]
            # Y_prob = F.softmax(logits, dim = 1)
            if instance_eval:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
                'inst_preds': np.array(all_preds)}
            else:
                results_dict = {}
            if return_features:
                results_dict.update({'features': M})
            return logits_t, Y_prob_t, Y_hat_t, A_raw, results_dict
        else:
            # device = xs.device
            # A, h = self.attention_net(xs)  # NxK        
            # A = torch.transpose(A, 1, 0)  # KxN
            # if attention_only:
            #     return A
            # A_raw = A
            # A = F.softmax(A, dim=1)  # softmax over N
            device = xs.device
            xs = xs.unsqueeze(0)

            instance_feature = []
            # print(xs.device)
            x = self.projector(xs) # 2048 -> 512 delete,clam 1024 -> 512
            # print(x.shape)
            # print(self.cls_token.shape)
            x = torch.cat((self.cls_token, x.cpu()), dim=1)
            x = x.to(device)
            x = self.dropout(x)
            rep = self.transformer(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim --> b,n,dim
            rep = rep.squeeze(0)
            H = rep[0]
            # instance_feature.append(rep[1:])
            instance_feature = rep[1:]
            # Instance_features = torch.cat(instance_feature)

            ## clam attention
            A, h = self.attention(instance_feature)
            A = torch.transpose(A, 1, 0)
            A_raw = A
            A = F.softmax(A, dim=1)
            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets =[]
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()# binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item() # 用于将张量或张量中的元素转换为Python标量
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, h, instance_feature, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, instance_feature, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss
                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

            # ################################################################
            # # ----> 使用cls_token 进行分类
            # print(H.size())
            # H = H.unsqueeze(0)
            # logits_t = self.classifiers(H)
            # print(logits_t.size())
            # Y_hat_t = torch.topk(logits_t, 1, dim=1)[1]
            # Y_prob_t = F.softmax(logits_t, dim = 1)
            ################################################################
            ## 使用原来的AttentionMIL进行分类
            # M = torch.mm(A, h) 
            # logits = self.classifiers(M)
            # Y_hat = torch.topk(logits, 1, dim = 1)[1]
            # Y_prob = F.softmax(logits, dim = 1)
            if instance_eval:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
                'inst_preds': np.array(all_preds)}
            else:
                results_dict = {}
            if return_features:
                results_dict.update({'features': M})
            return logits_t, Y_prob_t, Y_hat_t, A_raw, results_dict

