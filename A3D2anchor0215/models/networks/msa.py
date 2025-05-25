import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Processor, WavLMModel
from transformers import BertTokenizerFast, BertModel, ViTModel, RobertaModel
import torchvision.transforms as transforms

import numpy as np
import soundfile as sf
import librosa
from PIL import Image
import os
import sys
import time
import torch
from models.myfunc import lora_bert, lora_wav2vec, lora_wavlm, lora_apvit2, lora_vit #, lora_apvit
from models.networks.textcnn import TextCNN, TextCNN2, TextCNN3, RCNN
from models.networks.lstm import LSTMEncoder, LSTMEncoder2, TextSubNet
from models.networks.classifier import FcClassifier, OnelayerClassifier, SimpleFC

with open('config_dir.json', 'r') as file:
    dir = json.load(file)


wav_dir= dir['wav2vec']
wavlm_dir = dir['wavlm']
bert_dir = dir['bert-uncased']
roberta_dir = dir['roberta-base']
roberta_emo_dir = dir['roberta-base-emo']
apvit_dir = dir['apvit']
vit_dir = dir['vit']

sys.path.append(dir['apvit_folder'])


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params
    return trainable_params, total_params, ratio


def ft_layers(model, ft_layers):
    fine_tune_layers = [f"encoder.layer.{i}" for i in ft_layers] + [f"encoder.layers.{i}" for i in ft_layers] + [f"vit.blocks.{i}" for i in ft_layers]
    for name, param in model.named_parameters():
        if not any(layer in name for layer in fine_tune_layers):
            param.requires_grad = False


def pad_tensor(tensor, length=16):
    current_length = tensor.size(0)
    if current_length < length:
        padding = (0, 0, 0, length - current_length)  # Pad with zeros along the length dimension
        tensor = torch.nn.functional.pad(tensor, padding, "constant", 0)
    return tensor


def load_audio(file):
    """
    Load audio file using librosa.
    """  # librosa返回音频的时间序列和采样率
    y, sr = librosa.load(file, sr=16000)  # 将音频数据重新采样为指定的采样率--每秒采样的次数
    return y, sr


        # encoded_audio = self.audio_premodel(audios, sampling_rate=16000, padding=True, max_length=160320, truncation=True, return_tensors="pt")



class Wavlm(nn.Module):
    """
    Audio feature extractor using Wav2Vec2.
    Ref: https://huggingface.co/transformers/model_doc/wav2vec2.html
    """
    def __init__(self, device, lora_r, output_d):
        super(Wavlm, self).__init__()
        self.device = device
        
        self.audio_premodel = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_dir)
        self.model_A = WavLMModel.from_pretrained(wavlm_dir).to(device)
        ftl = [9, 10, 11]
        ft_layers(self.model_A, ftl)
        #self.model_A = lora_wavlm(self.model_A, lora_r)

        tr, to, r = count_parameters(self.model_A)
        print('AAAAAAAAA model_A', tr, to, r)
        self.textcnn_A = TextCNN3(256, 768).to(self.device)
        self.feature_re_A = SimpleFC(768, output_d, []).to(self.device)

        # for name, param in self.model_A.named_parameters():
        #     param.requires_grad = False

        


    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        
        audio_token = self.audio_premodel(kwargs['audio'], sampling_rate=16000, padding=True, max_length=80160 , truncation=True, return_tensors="pt")   #80160: 5s， 160320： 10s 
        audio_token = {key: value.to(self.device) for key, value in audio_token.items()}
     
        # audio_token = audio_token.to(self.device)
        output = self.model_A(**audio_token)
        hidden = output.last_hidden_state
        # hidden = torch.sum(hidden, dim=1)
        #print(f'wav2vec提取特征..{hidden.shape}')

        hidden = self.textcnn_A(hidden)
        hidden = self.feature_re_A(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)

        return hidden 





class Wav2vec(nn.Module):
    """
    Audio feature extractor using Wav2Vec2.
    Ref: https://huggingface.co/transformers/model_doc/wav2vec2.html
    """
    def __init__(self, device, lora_r, output_d):
        super(Wav2vec, self).__init__()
        self.device = device
        
        
        self.audio_premodel = Wav2Vec2FeatureExtractor.from_pretrained(wav_dir)
        self.model_A = Wav2Vec2Model.from_pretrained(wav_dir).to(device)
        self.model_A = lora_wav2vec(self.model_A, lora_r)
        self.textcnn_A = TextCNN3(256, 768).to(self.device)
        self.feature_re_A = SimpleFC(768, output_d).to(self.device)

        # for name, param in self.model_A.named_parameters():
        #     param.requires_grad = False

        # conditions = ["layers.11.final_layer_norm"]

        # for name, param in self.model.named_parameters():
        #     if any(condition in name for condition in conditions):
        #         param.requires_grad = True



    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):

        
        audio_token = self.audio_premodel(kwargs['audio'], sampling_rate=16000, padding=True, max_length=80160 , truncation=True, return_tensors="pt")   #80160: 5s， 160320： 10s 
        audio_token = {key: value.to(self.device) for key, value in audio_token.items()}
        # #print('AAAAAAAAAAAAAA:', audio_token['input_values'].shape)

        # audio_token = self.audio_premodel(kwargs['audio'], sampling_rate=16000, padding=True, max_length=160320, truncation=True, return_tensors="pt")   #80160: 5s， 160320： 10s 
        # audio_token = {key: value.to(self.device) for key, value in audio_token.items()}
        #audio_token['input_values'] = audio_token['input_values'][:, 0::2]
        # #print('AAAAAAAAAAAAAA:', audio_token['input_values'].shape)

        # audio_token = audio_token.to(self.device)
        output = self.model_A(**audio_token)
        hidden = output.last_hidden_state
        # hidden = torch.sum(hidden, dim=1)
        #print(f'wav2vec提取特征..{hidden.shape}')

        hidden = self.textcnn_A(hidden)
        hidden = self.feature_re_A(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)

        return hidden 



class Bert(nn.Module):
    """
    Text feature extractor using BERT
    Ref: https://huggingface.co/docs/transformers/model_doc/bert
    Pretrained models: https://huggingface.co/models
    """
    def __init__(self, device, lora_r, output_d):
        super(Bert, self).__init__()
        self.device = device
        self.model_L = BertModel.from_pretrained(bert_dir).to(self.device)
        ftl = [9, 10, 11]
        ft_layers(self.model_L, ftl)
        #self.model_L = lora_bert(self.model_L, lora_r)
        tr, to, r = count_parameters(self.model_L)
        print('LLLLLLLLLLL model_L', tr, to, r)

        self.seq2fea_L = TextCNN3(256, 768).to(self.device)
        self.feature_re_L = SimpleFC(768, output_d, []).to(self.device)

        # for param in self.model_L.parameters():
        #     param.requires_grad = False

        

        # # 只训练最后的几层
        # for name, param in self.model.named_parameters():
        #     # if "head.fc" in name or "vit.blocks.7" in name:
        #     # if "pooler.dense" in name:
        #     if "encoder.layer.11" in name:
        #     # if "encoder.layer.11" or "encoder.layer.10" in name:
        #         param.requires_grad = True        


    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # 将所有张量移动到指定的设备
        kwargs = {k: v.to(self.device) for k, v in kwargs.items()}
        # with torch.no_grad():
        output = self.model_L(**kwargs)
        hidden = output.last_hidden_state
        #print('AAAAAAAAAAAAAA:', hidden.shape)
        #hidden = hidden[:, 0, :]
        hidden = self.seq2fea_L(hidden)
        hidden = self.feature_re_L(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)
        return hidden



#####

class RobertaEmo(nn.Module):
    """
    Text feature extractor using BERT
    Ref: https://huggingface.co/docs/transformers/model_doc/bert
    Pretrained models: https://huggingface.co/models
    """
    def __init__(self, device, lora_r, output_d):
        super(RobertaEmo, self).__init__()
        self.device = device
        self.model_L = RobertaModel.from_pretrained(roberta_emo_dir).to(self.device)
        self.model_L = lora_bert(self.model_L, lora_r)

        self.seq2fea_L = TextCNN3(256, 768).to(self.device)
        self.feature_re_L = SimpleFC(768, output_d).to(self.device)

        # for param in self.model_L.parameters():
        #     param.requires_grad = False

        # # 只训练最后的几层
        # for name, param in self.model.named_parameters():
        #     # if "head.fc" in name or "vit.blocks.7" in name:
        #     # if "pooler.dense" in name:
        #     if "encoder.layer.11" in name:
        #     # if "encoder.layer.11" or "encoder.layer.10" in name:
        #         param.requires_grad = True        


    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # 将所有张量移动到指定的设备
        kwargs = {k: v.to(self.device) for k, v in kwargs.items()}
        # with torch.no_grad():
        output = self.model_L(**kwargs)
        hidden = output.last_hidden_state
        #print('AAAAAAAAAAAAAA:', hidden.shape)
        #hidden = hidden[:, 0, :]
        hidden = self.seq2fea_L(hidden)
        hidden = self.feature_re_L(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)
        return hidden



##


#####

class Roberta(nn.Module):
    """
    Text feature extractor using BERT
    Ref: https://huggingface.co/docs/transformers/model_doc/bert
    Pretrained models: https://huggingface.co/models
    """
    def __init__(self, device, lora_r, output_d):
        super(Roberta, self).__init__()
        self.device = device
        self.model_L = RobertaModel.from_pretrained(roberta_dir).to(self.device)
        self.model_L = lora_bert(self.model_L, lora_r)

        self.seq2fea_L = TextCNN3(256, 768).to(self.device)
        self.feature_re_L = SimpleFC(768, output_d).to(self.device)

        # for param in self.model_L.parameters():
        #     param.requires_grad = False

        # # 只训练最后的几层
        # for name, param in self.model.named_parameters():
        #     # if "head.fc" in name or "vit.blocks.7" in name:
        #     # if "pooler.dense" in name:
        #     if "encoder.layer.11" in name:
        #     # if "encoder.layer.11" or "encoder.layer.10" in name:
        #         param.requires_grad = True        


    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # 将所有张量移动到指定的设备
        kwargs = {k: v.to(self.device) for k, v in kwargs.items()}
        # with torch.no_grad():
        output = self.model_L(**kwargs)
        hidden = output.last_hidden_state
        #print('AAAAAAAAAAAAAA:', hidden.shape)
        #hidden = hidden[:, 0, :]
        hidden = self.seq2fea_L(hidden)
        hidden = self.feature_re_L(hidden)
        hidden = F.normalize(hidden, p=2, dim=1)
        return hidden



##



class APViT(nn.Module):  # 分开处理每一个图片

    def __init__(self, device):
        super(APViT, self).__init__()
        self.device = device
        self.model = torch.load(apvit_dir).to(self.device)

    def __call__(self, videos):
        return self.forward(videos)

    def forward(self, videos):
        features = []
        #t1 = time.time()

        for video in videos:
            video_feature = []
            for img in video:
                img = img.to(self.device)
                img = img.unsqueeze(0) 
                with torch.no_grad():                
                    feat = self.model.extract_feat(img)
                    output = feat[0]

                video_feature.append(output.flatten())  


            # print('提取一个视频的视觉特征..')
            video_feature = torch.stack(video_feature, dim=0)
            video_pool = torch.mean(video_feature, dim=0)
            features.append(video_pool) 

          

        features = torch.stack(features, dim=0)  
        #t2 = time.time()
        #print(f'APViT一个batch的特征维度：{features.shape}, 用时{t2-t1}')
        return features


class APViT_video(nn.Module):  # 一整个视频一起处理

    def __init__(self, device, lora_r, output_d):
        super(APViT_video, self).__init__()
        
        self.device = device
        self.model_V = torch.load(apvit_dir).to(self.device)
        #ftl = [5, 6, 7]
        ftl = [7]  #
        ft_layers(self.model_V, ftl)
        
        #self.model_V = lora_apvit2(self.model_V, lora_r)
        tr, to, r = count_parameters(self.model_V)
        print('VVVVVVVVVV model_V', tr, to, r)
        self.seq2fea_V = LSTMEncoder(768, 768, 'maxpool')#LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.feature_re_V = SimpleFC(768, output_d, []).to(self.device)
        #self.model = lora_apvit(self.model)

        # for param in self.model_V.parameters():
        #     param.requires_grad = False
        

        # # 只训练最后的几层
        # for name, param in self.model.named_parameters():
        #     # if "head.fc" in name or "vit.blocks.7" in name:
        #     if "head.fc" in name:
        #         param.requires_grad = True        

    # def __call__(self, videos):
    #     return self.forward(videos)

    # def forward(self, videos):
    #     features = []
    #     #t1 = time.time()
    #     for video in videos:
    #         video = video.to(self.device)
    #         # with torch.no_grad():     
    #         feat = self.model.extract_feat(video)
    #         output = feat[0]
    #         video_pool = torch.mean(output, dim=0)

    #         features.append(video_pool)  # 直接使用 PyTorch 的 flatten 方法


    #     features = torch.stack(features)  # 使用 torch.stack 将列表中的 tensor 堆叠为一个新的 tensor
    #     #t2 = time.time()
    #     #print(f'APViT一个batch的特征维度：{features.shape}, 用时{t2-t1}')
    #     return features
    

    def forward(self, videos):
        features = []
        lengths = [tensor.shape[0] for tensor in videos]

        videos = torch.cat(videos, dim=0)
        videos = videos.to(self.device)

        # with torch.no_grad():     
        feat = self.model_V.extract_feat(videos)
        output = feat[0]
        

        samples = torch.split(output, lengths)

        features = torch.stack(samples)
        features = self.seq2fea_V(features)

        features = self.feature_re_V(features)
        features = F.normalize(features, p=2, dim=1)

        return features



class ViT(nn.Module):  

    def __init__(self, device, lora_r, output_d):
        super(ViT, self).__init__()
        
        self.device = device
        self.model_V = ViTModel.from_pretrained(vit_dir).to(self.device)
        self.model_V = lora_vit(self.model_V, lora_r)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.seq2fea_V = LSTMEncoder(768, 768, 'maxpool')#LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.feature_re_V = SimpleFC(768, output_d).to(self.device)
        
        # for param in self.model_V.parameters():
        #     param.requires_grad = False

        # 只训练最后的几层
        # for name, param in self.model_V.named_parameters():
        #     # if "head.fc" in name or "vit.blocks.7" in name:
        #     if "head.fc" in name:
        #         param.requires_grad = True        

    def __call__(self, videos):
        return self.forward(videos)

    def forward(self, videos):
        features = []
        #t1 = time.time()
        lengths = [tensor.shape[0] for tensor in videos]

        videos = torch.cat(videos, dim=0)
        videos = videos.to(self.device)
        # print(f'batch大小{len(lengths)}')

            
        # with torch.no_grad():     
        output = self.model_V(videos).last_hidden_state[:, 0, :]    # [batch*5, 768]

        # print(f'所有音频last_hidden_state[:, 0, :]  {output.shape}')

        samples = torch.split(output, lengths)

        #features = [sample.mean(dim=0) for sample in samples]
        #print('CCCCCCCCCCCCfeature:', features.shape)
        #samples_p = [pad_tensor(sample, 16) for sample in samples]
        features = torch.stack(samples)
        features = self.seq2fea_V(features)

        features = self.feature_re_V(features)
        features = F.normalize(features, p=2, dim=1)

        #t2 = time.time()
        # print(f'ViT一个batch的特征维度：{features.shape}, 用时{t2-t1}')
        return features







# if __name__ == '__main__':
#     # file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangxinxin33/DA/d2d_data/meld/28_7_train.wav'
    
#     # y, sr = load_audio(file)
#     # preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(dir)
#     # audio_result = preprocessor(y, sampling_rate=sr, return_tensors="pt").input_values

#     # model = Wav2vec('cuda:6')
#     # result = model(audio_result)  # 使用 forward 方法而不是直接调用 extractor
#     # print(result)



#     text = 'I am so tired'
#     tokenizer = BertTokenizerFast.from_pretrained(bert_dir) 
#     text_result = tokenizer.encode(text, return_tensors='pt')

#     model = Bert('cuda:6')
#     result = model(text_result) 
#     print(result)
#     print(result.shape)

#     # image_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangxinxin33/DA/d2d_data/meld/154_9_test'
#     # image_files = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.jpg')])
#     # image = []
#     # # 逐个处理图像
#     # for image_file in image_files:
#     #     img = Image.open(image_file)

#     #     # 图像预处理
#     #     preprocess = transforms.Compose([
#     #         transforms.Resize((112, 112)),
#     #         transforms.ToTensor(),
#     #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     #     ])

#     #     img = preprocess(img)
#     #     image.append(img)

#     # model = APViT('cuda:5')
#     # reuslt = model(image)
#     # print(reuslt)
#     # print(reuslt.shape)




if __name__ == '__main__':
    # model = APViT_video('cuda:0')
    # # model = APViT('cuda:0')
    # model.eval()
    # params = list(model.named_parameters())
    # image = '/data3/sunjun/work/code/TBD/DA22/d2d_data/meld/0_1_test/0000.jpg'
    # image2 = '/data3/sunjun/work/code/TBD/DA22/d2d_data/meld/342_15_train/0001.jpg'
    # x = preprocess_image(image)
    # z = preprocess_image(image)
    # # x2 = x
    # print(x.shape)
    

    # y = [x, x, z]
    # y = torch.stack(y, dim=0)
    # print(y.shape)

    # z = torch.unsqueeze(z, dim=0)

    # model.eval()
    # rz = model(z)
    # model.eval()
    # r2 = model(y)


    # print(r2[-1])
    
    # print(rz-r2[-1])
    samples = tuple([
    torch.randn(2, 16, 768),
    torch.randn(2, 15, 768),
    torch.randn(2, 14, 768)
    ])
    samples_p = [pad_tensor(sample, 16) for sample in samples]
    a = torch.cat(samples_p, dim=0)
    print('AAAAA:', a.shape)
