import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder, LSTMEncoder2, TextSubNet
from models.networks.textcnn import TextCNN, TextCNN2, TextCNN3, RCNN
from models.networks.classifier import FcClassifier, OnelayerClassifier, SimpleFC
from models.networks.discriminator import DomainDiscriminator, DomainDiscriminatorS, DomainDiscriminatorSig
from models.networks.msa import Wav2vec, Bert, APViT, APViT_video, ViT, Wavlm
from models.coralutils.coral import log_var
from models.myfunc import entropy_re, dsm, ots
from models.func import ClassEmbedding, SupConLoss, PositionalEncoding
from models.opt import pte
import numpy as np
from scipy.stats import entropy
import math
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


class Config():
    def __init__(self):
        self.num_labels = 10  # CIFAR-10 classes
        self.percent = 1 # 0.2
        self.batch_size = 64
        self.path = "./data"

        self.is_lora = 1
        self.lora_alpha = 1
        self.lora_r = 16
        self.is_bert = 0
        self.is_quantization = 1

        self.learning_rate = 2e-4
        self.epochs = 2
        self.amp = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EmotionLightAnchorOT2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        parser.add_argument('--anchor', type=str, default='', help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.cos_theta1, self.cos_theta2, self.cos_theta3 = torch.tensor([]), torch.tensor([]), torch.tensor([])
        self.cos_theta4, self.cos_theta5, self.cos_theta6 = torch.tensor([]), torch.tensor([]), torch.tensor([])
        self.Logits_A, self.Logits_V, self.Logits_L = torch.tensor([]), torch.tensor([]), torch.tensor([])
        self.Logits_A_t, self.Logits_V_t, self.Logits_L_t = torch.tensor([]), torch.tensor([]), torch.tensor([])

        self.source = opt.source
        self.weight = opt.weight
        self.anchor = opt.anchor
        self.bs = opt.batch_size
        self.lora_r = 32
        self.iter = 0
        self.is_amp = 1
        self.x = [1/opt.embd_size_a * torch.eye(opt.embd_size_a).to(self.device)] * 2
        self.x2 = [1/opt.output_dim * torch.eye(opt.output_dim).to(self.device)] * 2
        self.x3 = [1/(opt.embd_size_a+opt.output_dim) * torch.eye(opt.embd_size_a+opt.output_dim).to(self.device)] * 2
        self.error_pte = []
        self.error_pte2 = []
        self.disc_test = opt.disc_test
        self.loss_names = ['CE', 'CE_A', 'CE_V', 'CE_L', 'domdisc_a', 'domdisc_v', 'domdisc_l', 'ot', 'ot2']
        self.modality = opt.modality
        self.model_names = ['C']

        self.embd_size_a = opt.embd_size_a
        self.embd_size_v = opt.embd_size_v
        self.embd_size_l = opt.embd_size_l

        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        self.output_dim = opt.output_dim 
        self.netC = SimpleFC(input_dim=cls_input_size, hidden=[128], output_dim=self.output_dim)

        self.netdiscCom = DomainDiscriminator(opt.embd_size_a + opt.embd_size_v + opt.embd_size_l)
        self.model_names.append('discCom')

        ini = torch.ones((self.bs, self.bs)).to(self.device)
        ini.diagonal().fill_(1)
        self.Pot1 = nn.Parameter(ini)
        self.Pot2 = nn.Parameter(ini)
        
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.model_names.append('CA')
            self.netCA = SimpleFC(input_dim=opt.embd_size_a, hidden=[], output_dim=opt.output_dim)
            self.netA = Wavlm(self.device, self.lora_r, opt.embd_size_a)
            self.netdiscA = DomainDiscriminator(self.embd_size_a * self.output_dim)
            self.model_names.append('discA')

        if 'L' in self.modality:
            self.model_names.append('L')
            self.model_names.append('CL')
            self.netCL = SimpleFC(input_dim=opt.embd_size_l, hidden=[], output_dim=opt.output_dim)
            self.netL = Bert(self.device, self.lora_r, opt.embd_size_l)
            self.netdiscL = DomainDiscriminator(self.embd_size_l * self.output_dim)
            self.model_names.append('discL')
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.model_names.append('CV')
            self.netCV = SimpleFC(input_dim=opt.embd_size_v, hidden=[], output_dim=opt.output_dim)
            self.netV = APViT_video(self.device, self.lora_r, opt.embd_size_v)
            self.netdiscV = DomainDiscriminator(self.embd_size_v * self.output_dim)
            self.model_names.append('discV')
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_domdisc = torch.nn.CrossEntropyLoss() #torch.nn.BCELoss()   #
            self.criterion_supcon = SupConLoss()
            self.criterion_nll = torch.nn.NLLLoss()
            self.criterion_mse = torch.nn.MSELoss()

            ft_lr = 5e-5 #1e-4
            parameters = [
                *[
                    {'params': [param for param in getattr(self, 'net' + net).parameters() if net not in ['A', 'V', 'L']], # 修改 
                    'weight_decay': opt.weight_decay, 'lr':opt.lr} 
                  
                    for net in self.model_names
                ]]
            parameters.append({'params': self.Pot1, 'lr': opt.lr*10})
            parameters.append({'params': self.Pot2, 'lr': opt.lr*10})

            if 'A' in self.modality:
                acoustic_params = []
                other_params_A = []
                for name, param in self.netA.named_parameters():
                    if 'model_A' in name:  
                        acoustic_params.append(param)
                    else:
                        other_params_A.append(param)
                
                # Create parameter groups for optimizer
                parameters.append({'params': acoustic_params, 'lr': ft_lr})
                parameters.append({'params': other_params_A, 'lr': opt.lr})

            if 'V' in self.modality:
                visual_params = []
                other_params_V = []
                for name, param in self.netV.named_parameters():
                    if 'model_V' in name:  
                        visual_params.append(param)
                    else:
                        other_params_V.append(param)

                parameters.append({'params': visual_params, 'lr': ft_lr})
                parameters.append({'params': other_params_V, 'lr': opt.lr})

            if 'L' in self.modality:
                lexical_params = []
                other_params_L = []
                for name, param in self.netL.named_parameters():
                    if 'model_L' in name:  
                        lexical_params.append(param)
                    else:
                        other_params_L.append(param)
                
                # Create parameter groups for optimizer
                parameters.append({'params': lexical_params, 'lr': ft_lr})
                parameters.append({'params': other_params_L, 'lr': opt.lr})

            self.optimizer = torch.optim.Adam(parameters, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input, input_t=None, alpha=None):
        """F
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        if self.isTrain or (self.disc_test and input_t != None):
            if 'A' in self.modality:
                self.acoustic = input['audio']
                self.acoustic_t = input_t['audio']

            if 'L' in self.modality:
                self.lexical = input['text']
                self.lexical_t = input_t['text']
   
            if 'V' in self.modality:
                self.visual = input['vision']
                self.visual_t = input_t['vision']

            self.label = input['label'].to(self.device)
            self.label_t = input_t['label'].to(self.device)
            self.alpha = alpha


            domain_source_labels = torch.zeros(self.label.shape[0]).type(torch.LongTensor) # 源域是0
            domain_target_labels = torch.ones(self.label_t.shape[0]).type(torch.LongTensor)  # 目标域是1
            self.domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()

        else:
            if 'A' in self.modality:
                self.acoustic = input['audio']
            if 'L' in self.modality:
                self.lexical = input['text']

            if 'V' in self.modality:
                self.visual = input['vision']
            
            self.label = input['label']
            


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.final_embd = []
        self.final_embd_target = []
        if self.isTrain:
            if 'A' in self.modality:
                self.feat_A = self.netA(**self.acoustic)
                self.final_embd.append(self.feat_A)

                self.feat_A_t = self.netA(**self.acoustic_t)
                self.final_embd_target.append(self.feat_A_t)
                
                self.logits_A = self.netCA(self.feat_A)
                self.pred_A = F.softmax(self.logits_A, dim=-1)

                self.logits_A_t = self.netCA(self.feat_A_t)
                self.pred_A_t = F.softmax(self.logits_A_t, dim=-1)
                self.predlabel_A_t = self.pred_A_t.argmax(dim=1).detach()#.cpu().numpy()

            
            if 'V' in self.modality:
                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)

                self.feat_V_t = self.netV(self.visual_t)
                self.final_embd_target.append(self.feat_V_t)

                self.logits_V = self.netCV(self.feat_V)
                self.pred_V = F.softmax(self.logits_V, dim=-1)

                self.logits_V_t = self.netCV(self.feat_V_t)
                self.pred_V_t = F.softmax(self.logits_V_t, dim=-1)
                self.predlabel_V_t = self.pred_V_t.argmax(dim=1)#.detach().cpu().numpy()

            if 'L' in self.modality:
                self.feat_L = self.netL(**self.lexical)              
                self.final_embd.append(self.feat_L)

                self.feat_L_t = self.netL(**self.lexical_t)
                self.final_embd_target.append(self.feat_L_t)

                self.logits_L = self.netCL(self.feat_L)
                self.pred_L = F.softmax(self.logits_L, dim=-1)

                self.logits_L_t = self.netCL(self.feat_L_t)
                self.pred_L_t = F.softmax(self.logits_L_t, dim=-1)
                self.predlabel_L_t = self.pred_L_t.argmax(dim=1)#.detach().cpu().numpy()
            
            if self.anchor == 'A':
                self.anchorlabel = self.predlabel_A_t
            if self.anchor == 'V':
                self.anchorlabel = self.predlabel_V_t
            if self.anchor == 'L':
                self.anchorlabel = self.predlabel_L_t

        else:
            if 'A' in self.modality:

                self.feat_A = self.netA(**self.acoustic)
                
                #self.feat_A = self.netAL(self.feat_A)
                self.final_embd.append(self.feat_A)

                self.logits_A = self.netCA(self.feat_A)
                self.pred_A = F.softmax(self.logits_A, dim=-1)
                
            
            if 'V' in self.modality:

                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)

                self.logits_V = self.netCV(self.feat_V)
                self.pred_V = F.softmax(self.logits_V, dim=-1)

            if 'L' in self.modality:
                self.feat_L = self.netL(**self.lexical)
                self.final_embd.append(self.feat_L)
                
                self.logits_L = self.netCL(self.feat_L)
                self.pred_L = F.softmax(self.logits_L, dim=-1)
        
        # get model outputs
        self.feat = torch.cat(self.final_embd, dim=-1)

        self.logits = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)

        if self.isTrain:
            self.feat_t = torch.cat(self.final_embd_target, dim=-1)
            self.logits_t = self.netC(self.feat_t)
            self.pred_t = F.softmax(self.logits_t, dim=-1)
            softmax_c = torch.cat((self.pred, self.pred_t), dim=0)

            if 'A' in self.modality:
                final_A_c = torch.cat((self.feat_A, self.feat_A_t), dim=0)
                op_out_A = torch.bmm(softmax_c.unsqueeze(2), final_A_c.unsqueeze(1))
                self.domainA = self.netdiscA(op_out_A.view(-1, self.embd_size_a * self.output_dim), self.alpha)
              
            if 'V' in self.modality:
                final_V_c = torch.cat((self.feat_V, self.feat_V_t), dim=0)
                op_out_V = torch.bmm(softmax_c.unsqueeze(2), final_V_c.unsqueeze(1))
                self.domainV = self.netdiscV(op_out_V.view(-1, self.embd_size_v * self.output_dim), self.alpha)

            if 'L' in self.modality:
                final_L_c = torch.cat((self.feat_L, self.feat_L_t), dim=0)
                op_out_L = torch.bmm(softmax_c.unsqueeze(2), final_L_c.unsqueeze(1))
                self.domainL = self.netdiscL(op_out_L.view(-1, self.embd_size_l * self.output_dim), self.alpha)
     
    def wd(self):
        D = []
        dd = torch.ones((self.bs, self.bs)).to(self.device)
        modalities = [i for i in self.modality]
        for m in ['A', 'V', 'L']:
            if m==self.anchor:
                anchor = f"feat_{m}"
                anchor_feat = getattr(self, anchor, None)
                modalities.remove(m)
                for n in modalities:
                    drift = f"feat_{n}_t"
                    drift_feat = getattr(self, drift, None)
                    d = dd - torch.matmul(anchor_feat, drift_feat.T)
                    D.append(d)
        return D
    
    def wd_p(self):
        D = []
        dd = torch.ones((self.embd_size_a, self.embd_size_a)).to(self.device)
        modalities = [i for i in self.modality]
        for m in ['A', 'V', 'L']:
            if m==self.anchor:
                anchor = f"feat_{m}_t"
                anchor_feat = getattr(self, anchor, None)
                anchor_feat = anchor_feat.detach()
                modalities.remove(m)
                for n in modalities:
                    drift = f"feat_{n}_t"
                    drift_feat = getattr(self, drift, None)
                    d = dd - torch.matmul(anchor_feat.T, drift_feat)/self.bs
                    D.append(d)
        return D

    def wd_p2(self):
        D = []
        #dd = torch.ones((self.bs, self.bs)).to(self.device)
        dd = torch.ones((self.output_dim, self.output_dim)).to(self.device)
        modalities = [i for i in self.modality]
        for m in ['A', 'V', 'L']:
            if m==self.anchor:
                anchor = f"pred_{m}_t"
                anchor_feat = getattr(self, anchor, None)
                anchor_feat = anchor_feat.detach()
                modalities.remove(m)
                for n in modalities:
                    drift = f"pred_{n}_t"
                    drift_feat = getattr(self, drift, None)
                    d = dd - torch.matmul(anchor_feat.T, drift_feat)/self.bs
                    D.append(d)
        return D

            
    def wd_p3(self):
        D = []
        dd = 2 * torch.ones((self.output_dim+self.embd_size_a, self.output_dim+self.embd_size_a)).to(self.device)
        modalities = [i for i in self.modality]
        for m in ['A', 'V', 'L']:
            if m==self.anchor:
                anchor_p = f"pred_{m}_t"
                anchor_f = f"feat_{m}_t"
                anchor_feat_f = getattr(self, anchor_f, None)
                anchor_feat_f = anchor_feat_f.detach()
                anchor_feat_p = getattr(self, anchor_p, None)
                anchor_feat_p = anchor_feat_p.detach()
                anchor_feat = torch.cat((anchor_feat_f, anchor_feat_p), dim=1)
                modalities.remove(m)
                for n in modalities:
                    drift_p = f"pred_{m}_t"
                    drift_f = f"feat_{m}_t"
                    drift_feat_f = getattr(self, drift_f, None)
                    drift_feat_p = getattr(self, drift_p, None)
                    drift_feat = torch.cat((drift_feat_f, drift_feat_p), dim=1)
                    d = dd - torch.matmul(anchor_feat.T, drift_feat)/self.bs
                    D.append(d)
        return D
    def lossot(self):
        D = self.wd_p()
        loss = 0
        beta = 0.05
        
        for i in range(0, len(D)):
            n_c = D[i].shape[0]
            xx, obj = ots(D[i].detach().cpu().numpy(), n_c)
            xx = torch.tensor(xx).to(self.device)
            self.x[i] = ( 1- beta) * self.x[i] + beta * xx
            loss +=  torch.sum(D[i]*self.x[i])
        return loss

        
    def lossot2(self):
        D = self.wd_p2()
        loss = 0
        beta = 0.05
        
        for i in range(0, len(D)):
            n_c = D[i].shape[0]
            xx, obj = ots(D[i].detach().cpu().numpy(), n_c)
            xx = torch.tensor(xx).to(self.device)
            self.x2[i] = ( 1- beta) * self.x2[i] + beta * xx
            loss +=  torch.sum(D[i]*self.x2[i])
        return loss

    def lossot3(self):
        D = self.wd_p3()
        loss = 0
        beta = 0.05
        for i in range(0, len(D)):
            n_c = D[i].shape[0]
            xx, obj = ots(D[i].detach().cpu().numpy(), n_c)
            xx = torch.tensor(xx).to(self.device)
            self.x3[i] = (1- beta) * self.x3[i] + beta * xx
            loss +=  torch.sum(D[i]*self.x3[i])
        return loss


    def loss_cal(self):  
        """Calculate the loss (for amp)"""
        self.iter += 1

        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_domdisc_a = 0
        self.loss_domdisc_v = 0
        self.loss_domdisc_l = 0 
        self.loss_CE_A = self.loss_CE_V = self.loss_CE_L = 0
        self.loss_lv_a = self.loss_lv_v = self.loss_lv_l = 0
        self.loss_drift_CE_A_t = self.loss_drift_CE_V_t = self.loss_drift_CE_L_t = 0
        if 'A' in self.modality:
            self.loss_domdisc_a = self.criterion_domdisc(self.domainA, self.domain_combined_label)
            comparison = (self.domainA[:, 0] < self.domainA[:, 1]).float()
            acc = (comparison == self.domain_combined_label).float()
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
            self.loss_drift_CE_A_t = self.criterion_ce(self.logits_A_t, self.anchorlabel)
            self.loss_lv_a = log_var(self.feat_A)
        if 'V' in self.modality:
            self.loss_domdisc_v = self.criterion_domdisc(self.domainV, self.domain_combined_label)
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label)
            self.loss_drift_CE_V_t = self.criterion_ce(self.logits_V_t, self.anchorlabel)
            self.loss_lv_v = log_var(self.feat_V)
        if 'L' in self.modality:
            self.loss_domdisc_l = self.criterion_domdisc(self.domainL, self.domain_combined_label)
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)
            self.loss_drift_CE_L_t = self.criterion_ce(self.logits_L_t, self.anchorlabel)
            self.loss_lv_l = log_var(self.feat_L)
        
        loss_lv = (self.loss_lv_a + self.loss_lv_v + self.loss_lv_l)

        self.loss_drift_CE_t = (1- int(self.anchor == 'A')) * self.loss_drift_CE_A_t \
                             + (1- int(self.anchor == 'V')) * self.loss_drift_CE_V_t \
                             + (1- int(self.anchor == 'L')) * self.loss_drift_CE_L_t
        
        self.loss_CEm = self.loss_CE_A + self.loss_CE_V + self.loss_CE_L

        weight_cls = 1


        loss_ada = self.loss_domdisc_a * 0.1 + self.loss_domdisc_v * 0.7 + self.loss_domdisc_l * 0.1  
        self.loss_ot = 0
        self.loss_ot2 = 0
        flag_ot = 0
        if self.source == 'IEMOCAP':
            ot_s = 200
        if self.source == 'MINTREC':
            ot_s = 120  
        if self.iter >= ot_s:  
            flag_ot = 1
            self.loss_ot = self.lossot()
            self.loss_ot2 = self.lossot2()

        self.loss = self.loss_CE + self.weight * loss_ada + 0.1 * self.loss_CEm + 0.001 * loss_lv + flag_ot * (self.loss_ot2 + self.loss_ot)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        if self.is_amp:
            with autocast():
                self.forward() 
                self.loss_cal()  
            # backward
            self.optimizer.zero_grad() 
            scaler.scale(self.loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # forward
            self.forward()   
            # backward
            self.optimizer.zero_grad() 
            self.backward()  
            self.optimizer.step()


