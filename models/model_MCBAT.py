from models.model_TransformerMIL import *

# ---->> fusion 
class FusionEncoder(nn.Module):
    def __init__(self, num_latents):
        super(FusionEncoder, self).__init__()

        # self.trans_low = trans_low
        # self.trans_high = trans_high
        # self.cls_low = cls_token_low
        # self.cls_high = cls_token_high # shape = (1, 1, 512)
        # concat_cls = torch.cat([self.cls_low, self.cls_high], dim=2)
        # self.projector = nn.Linear(concat_cls, 512)
        # self.concat_cls = self.projector(concat_cls)
        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 512).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1)) # 将 self.scale_a 初始化为 0 是为了在初始状态下，
        self.scale_v = nn.Parameter(torch.zeros(1)) # 让模型默认不采用全局信息的交叉注意力，而是在训练过程中学习适当的融合程度

    # single head self-attention
    def attention(self,q,k,v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    # 定义fusion方法,将低分辨率特征和高分辨率特征拼接,与latent vector做交叉Attention,得到融合的低分辨率和高分辨率表示
    def fusion(self, low_tokens, high_tokens):
        # shapes
        BS = low_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((low_tokens,high_tokens),dim=1) # B, N+M, C
        # cross attention (local -->> global latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_) # B, num_latents, C
        # cross attention (global latents -->> local)
        low_tokens = low_tokens + self.scale_a * self.attention(q=low_tokens, k=fused_latents, v=fused_latents)
        high_tokens = high_tokens + self.scale_v * self.attention(q=high_tokens, k=fused_latents, v=fused_latents)
        return low_tokens, high_tokens, fused_latents
    
    def forward(self, x, y, return_bottoken = False):
        # Bottleneck Fusion
        x,y,z = self.fusion(x,y)
        # x = self.trans_low(x)
        # y = self.trans_high(y)
        if return_bottoken:
            return x,y,z
            
        else:
            return x,y

class DRFF(nn.Module):
    def __init__(self, num_latents):
        super(DRFF, self).__init__()

        # self.trans_low = trans_low
        # self.trans_high = trans_high
        # self.cls_low = cls_token_low
        # self.cls_high = cls_token_high # shape = (1, 1, 512)
        # concat_cls = torch.cat([self.cls_low, self.cls_high], dim=2)
        # self.projector = nn.Linear(concat_cls, 512)
        # self.concat_cls = self.projector(concat_cls)
        # Latents
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 512).normal_(std=0.02))

    # single head self-attention
    def attention(self,q,k,v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    # 定义fusion方法,将低分辨率特征和高分辨率特征拼接,与latent vector做交叉Attention,得到融合的低分辨率和高分辨率表示
    def initial_latents(self, low_tokens, high_tokens):
        # shapes
        BS = low_tokens.shape[0]
        
        # concat all the tokens
        concat_ = torch.cat((low_tokens,high_tokens),dim=1) # B, N+M, C
        # cross attention (local -->> global latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_) # B, num_latents, C
        return fused_latents
    
    def DRFF(self, low_tokens, high_tokens,fused_latents):
        fused_latents = self.attention(q=fused_latents, k=low_tokens, v=low_tokens)
        fused_latents = self.attention(q=fused_latents, k=high_tokens, v=high_tokens)
        return fused_latents

    
    def forward(self, x, y, l=1):
        z = self.initial_latents(x,y)
        for _ in range(l):
            z = self.DRFF(x,y,z)
        return z

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

class MCBAT_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", depth = 1,  dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(MCBAT_SB, self).__init__()

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
        self.catclassifiers = nn.Linear(512*6, n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # initialize_weights(self)
        ################################################################
        ## self_add
        self.cls_token_low = nn.Parameter(torch.rand(1,1,size[1]))
        self.cls_token_high = nn.Parameter(torch.rand(1,1,size[1]))
        self.transformer_low  = TransformerEncoder_FLASH(size[1], depth, 8, 64, 2048, 0.1) # mlp_dim = 2048 一般取4*dim增强模型的表达能力
        self.transformer_high = TransformerEncoder_FLASH(size[1], depth, 8, 64, 2048, 0.1) # mlp_dim = 2048 一般取4*dim增强模型的表达能力
        self.projector = nn.Linear(1024, size[1])
        self.dropout = nn.Dropout(0.1)
        self.attention = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        encoder_layers = []
        # for i in range(depth):
        #     encoder_layers.append(FusionEncoder(2))
        # self.fusion_encoder = nn.Sequential(*encoder_layers)
        # self.fusion_encoder = FusionEncoder(2, self.transformer_low, self.transformer_high)
        # self.fusion_encoder = FusionEncoder(4)
        # self.fusion_encoder = FusionEncoder(4)
        self.fusion_encoder = DRFF(4)

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
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.catclassifiers = self.catclassifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        
        self.projector = self.projector.to(device)
        # self.cls_token_low = self.cls_token_low.to(device)
        self.transformer_low =self.transformer_low.to(device)
        self.transformer_high =self.transformer_high.to(device)
        self.dropout = self.dropout.to(device)
        self.attention = self.attention.to(device)
        self.attention_V2 = self.attention_V2.to(device)
        self.attention_U2 = self.attention_U2.to(device)
        self.attention_weights2 =  self.attention_weights2.to(device)
        self.fusion_encoder = self.fusion_encoder.to(device)
    
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
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
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
        ## self add top-k 
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        # top_n = [instance_feature[i] for i in top_n_ids]
        # top_n = torch.cat(top_n, dim = 1).squeeze(0)
        n_targets = self.create_negative_targets(self.k_sample, device)
        ################################################################
        p_targets = self.create_negative_targets(self.k_sample, device)
        
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    def forward_low_scale(self, x):
        device = x.device
        x = x.unsqueeze(0)
        x = self.projector(x)
        x = torch.cat((self.cls_token_low, x.cpu()), dim=1)
        x = x.to(device)
        x = self.dropout(x)

        rep = self.transformer_low(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim --> b,n,dim
        rep = rep.squeeze(0)
        H= rep[0]
        instance_feature = rep[1:]

        return x,H, instance_feature
    
    def forward_high_scale(self, x):
        device = x.device
        x = x.unsqueeze(0)
        x = self.projector(x)
        x = torch.cat((self.cls_token_high, x.cpu()), dim=1)
        x = x.to(device)
        x = self.dropout(x)

        rep = self.transformer_high(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim --> b,n,dim
        rep = rep.squeeze(0)
        H= rep[0]
        instance_feature = rep[1:]
        # Instance_features = torch.cat(instance_feature)

        return x,H, instance_feature


    def forward(self, xs, ys, label=None, instance_eval=False, return_features=False, attention_only=False, cluster_feature=False):
        if cluster_feature:
        ################################################################
        # ----> add cluster            
            H = []
            instance_feature = []
            for x in xs:
                x = self.projector(x) # 2048 -> 512 delete,clam 1024 -> 512
                x = torch.cat((self.cls_token_low, x), dim=1)
                x = self.dropout(x)
                rep = self.transformer(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim
                H.append(rep[:, 0]) # class_token
                instance_feature.append(rep[:, 1:])
            H = torch.cat(H) # B,10,512
            ## MOMA cls_token_low attention 
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
                M = torch.mm(A,H) # KxL
                logit = self.classifiers(M)
                return logit, total_inst_loss, A_raw
        else:
            # device = xs.device
            # A, h = self.attention_net(xs)  # NxK        
            # A = torch.transpose(A, 1, 0)  # KxN
            # if attention_only:
            #     return A
            # A_raw = A
            # A = F.softmax(A, dim=1)  # softmax over N

            ## forward respectively
            x, H_low, instance_feature_low = self.forward_low_scale(xs)
            y, H_high, instance_feature_high = self.forward_high_scale(ys)
            # x,y,bottle_token = self.fusion_encoder(x,y,return_bottoken = True)
            H_fuse = self.fusion_encoder(x,y,l=3)

            ## clam attention
            A, h = self.attention(instance_feature_high)
            A = torch.transpose(A, 1, 0)
            if attention_only:
                return A
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
                        instance_loss, preds, targets = self.inst_eval(A, h, instance_feature_high, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, instance_feature_high, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss
                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

            ################################################################
            # ----> 之前实验用的
            # x = x[:, 0]
            # y = y[:, 0]
            # H = (x+y)*0.5
            ################################################################
            # ----> cat low&high
            # H = torch.cat((H_high, H_low))
            # H = H.unsqueeze(0)
            ################################################################
            # ---->> 使用cls_token 和 bot_token 一起进行分类
            H_fuse = H_fuse.squeeze(0)
            H_low = H_low.unsqueeze(0)
            H_high = H_high.unsqueeze(0)
            H = torch.cat((H_high, H_low))
            H = torch.cat((H_fuse, H))
            H = torch.flatten(H).unsqueeze(0)
            logits_t = self.catclassifiers(H)
            Y_hat_t = torch.topk(logits_t, 1, dim=1)[1]
            Y_prob_t = F.softmax(logits_t, dim = 1)
            
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















