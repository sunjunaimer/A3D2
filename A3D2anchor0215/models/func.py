
import numpy as np
import torch
#from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################
#https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

class ClassEmbedding():
    def __init__(self, dim, n):
        super(ClassEmbedding, self).__init__()
        self.dim = dim 
        self.n = n 
    
    def get_emd(self, class_label):
        num_class = class_label.shape[0]
        emd = np.zeros((num_class, self.dim))
        for k, label in enumerate(class_label):
            for i in np.arange(int(self.dim/2)):
                denominator = np.power(self.n, 2*i/self.dim)
                emd[k, 2*i] = np.sin(label/denominator)
                emd[k, 2*i+1] = np.cos(label/denominator)
        emd = torch.tensor(emd)
        emd = emd.to(torch.float32)
        emd = F.normalize(emd, p=2, dim=1)
        return emd

############################################
#https://github.com/ckarouzos/slp_daptmlm/blob/master/slp/util/embeddings.py

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding
    PE(pos,2i)=sin(pos/10000^(2i/dmodel))
    PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
    """
    def __init__(self, max_length, embedding_dim=512, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_length, embedding_dim,
                         dtype=torch.float, device=device)
        embedding_indices = torch.arange(0, embedding_dim,
                                         dtype=torch.float, device=device)
        position_indices = (torch
                            .arange(0, max_length,
                                    dtype=torch.float, device=device)
                            .unsqueeze(-1))
        # freq => (E,)
        freq_term = 10000 ** (2 * embedding_indices / embedding_dim)
        pe[:, 0::2] = torch.sin(position_indices / freq_term[0::2])
        pe[:, 1::2] = torch.cos(position_indices / freq_term[1::2])
        # pe => (1, max_length, E)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x => (B, L, E) sequence of embedded tokens
        """
        # (B, L, E)
        F.normalize(x, p=2, dim=2) + F.normalize(self.pe[:, :x.size(1)], p=2, dim=2)
        return x + 0.1*self.pe[:, :x.size(1)]

###############################################
"""
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #print('LLLLLLLLLLL:', labels)
        #print('MMMMMMMMMMMMMMMM:', mask, mask.shape, sum(mask))
        mask = mask * logits_mask
        #print('AAAAAAAAAAA:', mask, sum(mask))
        #print('BBBBBBBBBBBBBB:', logits_mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class SingleFullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=False):
        ''' Fully connected layer
            Parameters:
            --------------------------
            input_dim: input feature dim
            dropout: dropout rate
            use_bn: use batchnorm or not
        '''
        super().__init__()
        self.all_layers = []

        self.all_layers.append(nn.Linear(input_dim, output_dim))
        self.all_layers.append(nn.BatchNorm1d(output_dim))
        self.all_layers.append(nn.LeakyReLU(negative_slope=0.01))
        


    
        self.module = nn.Sequential(*self.all_layers).to('cuda')
    
    def forward(self, x):
        feat = self.module(x)
        return feat
    

# --------------------Prototype based Reliability Measurement ---------------------

def cal_similarity(a, b, gamma=1):  # similarity formula(2)  
    a = a.to(device)
    b = b.to(device)
    distance = torch.norm(a - b)
    # print(distance)
    # print('a:', a)
    # print('b:', b)
    # print(torch.exp(-distance**2 / gamma))
    return torch.exp(-distance**2 / gamma)



class PrototypeLoss(nn.Module): 
    def __init__(self):
        super(PrototypeLoss, self).__init__()

    def forward(self, features, label, prototype):  
        # features:[batch, prototype_dim], label:[batch, cls_dim], prototypes:[cls_dim, prototype_dim]
        """ Compute the multi-label classification loss by measuring the similarity 
        between each input sample with the prototype of each class 
        to learn the prototypes

        Args:
            features: Feature Extractor → Prototype Mapper 👉 features
            label: one-hot label.
            prototype: All prototypes for classes in this modality.
        Returns:
            A loss scalar.
        """

        features = features.to(device)
        prototype = prototype.to(device)
        label = label.to(device)
    
        epsilon = 0.001

        a_expanded = features.unsqueeze(1)  # 维度变为 (64, 1, 256)
        b_expanded = prototype.unsqueeze(0)  # 维度变为 (1, 4, 256)


        euclidean_distances = torch.cdist(a_expanded, b_expanded, p=2.0) # 计算欧几里得距离[batch, 1, 4]
        euclidean_distances_compressed = euclidean_distances.squeeze(dim=1)
        prototye_similarity = torch.exp(-1 * euclidean_distances_compressed ** 2 / 1)  # 和所有类别的prototype的similarity[batch, cls_dim]
                                                                                        # γ-欧氏距离中的缩放因子

        is_lable = torch.log(prototye_similarity + epsilon) * label * -1
        sum1 = torch.sum(is_lable)

        unlabel = -1 * (1 - label)
        not_label = unlabel * torch.log(1 - prototye_similarity + epsilon)
        sum2 = torch.sum(not_label)

        loss = (sum1 + sum2) / features.shape[0]
        loss = loss.to(device)

        return loss


def label_smoothing(labels, max_realiability):  # function(7) 
    """"
    Generate the smoothing label.

     Args:
        labels: [batch, cls_dim](one-hot)
        max_realiability: [batch]  

    Return:
        labels after smoothing: [batch, cls_dim] (not one-hot)       
    """
    max_realiability = max_realiability.to(device)
    labels = labels.to(device)
    
    num_sample = labels.shape[0]
    cls_dim = labels.shape[1]


    addition = (torch.ones([num_sample]).to(device) - max_realiability) / cls_dim  # [batch]
    smoothing = labels * max_realiability.view(-1, 1) + addition.view(-1, 1)
    # print('***',smoothing.shape)
    return smoothing



def cal_reliability(features, prediction, prototype):  # function(5)
        # features:[batch, prototype_dim], prediction:[batch], prototypes:[cls_dim, prototype_dim]
        """
        Measure the similarity between each input sample with the prototype of the class computed through G.
        (Single-modality)

        Args:
            features: Feature Extractor → Prototype Mapper 👉 features.
            prediction: One-hot prediction outputed by classifier G. [batch, cls_dim]
            prototype: All prototypes for classes in this modality.  # 只有一个模态的prototype
        Returns:
            Reliabilities: [batch]  (numpy)
        """

        epsilon = 0.001

        prediction = prediction.to(torch.float32).to(device)
        features = features.to(torch.float32).to(device)
        prototype = prototype.to(torch.float32).to(device)

        a_expanded = features.unsqueeze(1)  # 维度变为 (batch, 1, prototype_dim)
        b_expanded = prototype.unsqueeze(0)  # 维度变为 (1, cls_dim, prototype_dim)


        euclidean_distances = torch.cdist(a_expanded, b_expanded, p=2.0)  # 计算欧几里得距离[batch, 1, 4]
        euclidean_distances_compressed = euclidean_distances.squeeze(dim=1)  
        prototye_similarity = torch.exp(-1 * euclidean_distances_compressed ** 2 / 1.5)  # 和所有类别的prototype的similarity[batch, cls_dim] γ缩放因子
        
        matrix_reliability = torch.matmul(prediction, prototye_similarity.t())
        diagonal_elements = matrix_reliability.diagonal() + epsilon
        return diagonal_elements

            


# --------------------Asynchronous Learning ---------------------
def target_loss(features, label):   # function(10)
        # features:[modality_num, batch, cls_dim], label:[batch, cls_dim]
        """
        Calculate the cross-entropy loss of the sub-model to predict the label.(All sub-models)
       
        Args:
            features: Feature Extractor → Classifier 👉 features.
            label: Labels after smoothing. (choose the highest-reliability modality)
        Returns:
            loss: [modality, batch]  
        """  
        
        all_loss = torch.empty(features.shape[0], features.shape[1])
        features = features.to(device)
        label = label.to(device)

        for modality in range(features.shape[0]):   # modality_num

            loss = F.cross_entropy(features[modality], label, reduction='none')
            all_loss[modality] = loss
        
        return all_loss



def Asynchronous_target_loss(features, label):   # function(10)
        # features:[modality_num, batch, cls_dim], label:[batch, cls_dim]
        """
        Calculate the cross-entropy loss of the !!⭐multi-modal to predict the label.
       
        Args:
            features: Feature Extractor → Classifier 👉 features.
            label: Labels after smoothing. (choose the highest-reliability modality)
        Returns:
            loss: [modality, batch]  
        """  
        all_features = torch.mean(features, dim=0)  # [batch, cls_dim]


        all_features = all_features.to(device)
        entropy_loss = F.cross_entropy(all_features, label, reduction='none')  # 使用没有softmax的
        all_loss = torch.unsqueeze(entropy_loss, dim=0)  # 模仿多模态，返回三个一样的loss_t
        result_loss = all_loss.repeat(3, 1)

        return result_loss# 最后返回的是torch
        



def sample_selection(loss, threshold):   # function(11)
        # loss:[modality, batch], threshold:float
        """
        Choose the easier samples with the loss in targert domain less than τ.
       
        Args:
            loss: The cross-entropy loss of the sub-model to precdict the label.(All sub-models)
            threshold: Control the learning difficulty for each modality. 
        Returns:
            V_binary: Determine whether the sample in the target domain is chosen to learn. 
            num: The number of selected samples.
            [modality, batch]
        """    
        boundary = torch.empty((loss.shape[0],))
        for modality in range(loss.shape[0]):
            sorted_loss, _ = torch.sort(loss[modality]) # 排列每个模态的loss(10) 默认是从小到大
            batch = loss.shape[1]
            boundary[modality] = sorted_loss[min(int(threshold*batch)+1, batch-1)]  # 最后boundary:[modality]  修改✔
            # boundary[modality] = sorted_loss[int(threshold * batch)]  # 最后boundary:[modality]

        result = []
        for modality in range(loss.shape[0]):
            result.append(torch.where(loss.to(device) < boundary[modality].to(device), torch.tensor(1).to(device), torch.tensor(0).to(device)))
            # 最后result中有三个tensor, 都是[modality, batch], 下一步使用掩码拼接一下

        tensor_new0 = torch.zeros_like(loss)
        tensor_new0[0, :] = 1
        tensor_new1 = torch.zeros_like(loss)
        tensor_new1[1, :] = 1
        tensor_new2 = torch.zeros_like(loss)
        tensor_new2[2, :] = 1

        v_binary = result[0].to(device) * tensor_new0.to(device) + result[1].to(device) * tensor_new1.to(device) + result[2].to(device) * tensor_new2.to(device)

                 
        return v_binary, min(int(threshold*batch)+1, batch-1) - 1 




# -------------------Reliabiliy-aware Fusion ---------------------
def reliability_late_fusion(features, reliability):   # function(12)
        # features:[modality, batch, cls_dim], reliability:[modality, batch]
        """
        Fuse the prediction results from all sub-models 
       
        Args:
            features: Feature Extractor → Classifier 👉 features.
            reliability: The reliability of the class corresponding to pseudo-labels. Control the learning difficulty for each modality. 
        Returns:
            reuslt: Final prediction vector [batch, cls_dim]
            
        """   
        features = features.to(device)
        reliability = reliability.to(device)
        prediction = []
        for sample in range(features.shape[1]): 
            sum_re = torch.sum(reliability[:, sample])
            feature1 = features[:, sample, :]  # [modality, cls_dim]
            weight = reliability[:, sample] # [moadlity]
            # print('ffffff', feature1.shape)
            # print('wwwwww', weight.shape)
            # print((feature1 * weight.view(-1, 1)).shape)
            s_prediction = torch.sum(feature1 * weight.view(-1, 1), dim=0) / sum_re
            prediction.append(s_prediction)

        result = torch.stack(prediction)
        return result



def prototype(cls_dim, prototype_dim):
        """
        Prototype for single modality 
       
        Returns:
            reuslt: [cls_dim, prototype_dim]
            
        """  
        prototype = torch.nn.Parameter(torch.randn(cls_dim, prototype_dim))

        return prototype



def late_fusion(features):
        # features:[modality, batch, cls_dim]
        """
        Late fusion in MM-SADA.fuction(1)
        
        Args:
            features: The prediction of all modalities 
        Returns:
            result: [batch, cls_dim]
        """       
# 直接进入CrossEntropy,不需要softmax
        result = torch.sum(features, dim=0, keepdim=False)
        return result


def cal_cxy(x, y, dim):  # x: n
    # x/y : [n, a_length, v_length, l_length]
    """
        n个是样本数
        第一个样本的维度是ijk
        第二个样本不一样的地方是l
    """
    if dim == 0:
        c = torch.einsum('nijk,nljk->il', x, y)    
    elif dim == 1:
        c = torch.einsum('nijk,nilk->jl', x, y)   
    elif dim == 2:
        c = torch.einsum('nijk,nijl->kl', x, y)   

    return c # [n, a_length, a_length]    [n, v_length, v_length]    [n, l_length, l_length]


def outer_3vector(a, v, l):
    batch = a.shape[0]
    a_length = a.shape[1]
    v_length = v.shape[1]
    l_length = l.shape[1]

    outer_abc = torch.zeros((batch, a_length, v_length, l_length), device=a.device)
    
    # 计算每一组向量的外积
    for i in range(batch):
        outer_ab = torch.ger(a[i], v[i])  # 计算第 i 组的 a 和 b 的外积
        outer_abc[i] = torch.einsum('ij,k->ijk', outer_ab, l[i])  # 使用第 i 组的 c 计算最终的外积

    return outer_abc  # [batch, a_length, v_length, l_length]





class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads

        self.linear_qs = torch.nn.Linear(input_dim, input_dim, bias=False).to('cuda')
        self.linear_kt = torch.nn.Linear(input_dim, input_dim, bias=False).to('cuda')

    

        torch.nn.init.xavier_normal_(self.linear_qs.weight)
        torch.nn.init.xavier_normal_(self.linear_kt.weight)


    def forward(self, s, t):
    
        Qs = self.linear_qs(s)
        Kt = self.linear_kt(t)
        Qs = Qs.unsqueeze(2)
        Kt = Kt.unsqueeze(1)
        Mst = torch.bmm(Qs, Kt)

        return Mst











if __name__ == '__main__':      
    cls_emb = ClassEmbedding(dim=4, n=100)
    label = torch.tensor([1,2,2,3,0, 0])

    print(cls_emb.n)
    a1 = cls_emb.get_emd(label)
    print(a1)




