
import os
os.environ['CURL_CA_BUNDLE'] = ''  #for transformer error

from transformers import BertModel, BertConfig
import torch
from torch import nn
import torch.nn.functional as F



# bert_version = "bert-base-uncased"
# model_path = "/data3/zxx/pretrained_models/bert-base-uncased/"
# model_path = "/data3/zxx/pretrained_models/bertMLM/4/"
#bert_base_cased = BertModel.from_pretrained(bert_version)  # Instantiate model using the trained weights
# bert_base_uncased = BertModel.from_pretrained(model_path)
# config = BertConfig.from_pretrained(model_path)
# config.num_hidden_layers = 11
#model = BertModel.from_pretrained(bert_version, config=config)  # auto skip unused layers

# model = BertModel.from_pretrained(model_path, config=config)

# for param_name in model.state_dict():
#     sub_param, full_param = model.state_dict()[param_name], bert_base_cased.state_dict()[param_name] # type: torch.Tensor, torch.Tensor
#     assert (sub_param.cpu().numpy() == full_param.cpu().numpy()).all(), param_name


# model.encoder.layer[0].load_state_dict(bert_base_cased.encoder.layer[10].state_dict())
# model.encoder.layer[1].load_state_dict(bert_base_cased.encoder.layer[11].state_dict())





# class BertTop(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = model.encoder
#         self.pooler = model.pooler
    
#     def forward(self, x):
#         encoder_outputs = self.encoder(x)
#         sequence_output = encoder_outputs[0]
#         output = self.pooler(sequence_output)

#         return output



class BertTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = model.pooler
        self.embd = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        sequence_output = encoder_outputs[0]
        output = self.pooler(sequence_output)
        output = self.embd(output)
        output = F.normalize(output, p=2, dim=1)

        return output



class BertTopSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = model.encoder
        #self.pooler = model.pooler

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        sequence_output = encoder_outputs[0]
        #sequence_output = encoder_outputs[0]
        #output = self.pooler(sequence_output)
        # output = F.normalize(output, p=2, dim=1)

        return sequence_output



if __name__ == '__main__':
    import transformers
    #print(transformers.__file__)

    print(bert_base_uncased)
    # print(model.encoder)

    # model.encoder.layer[0].load_state_dict(bert_base_cased.encoder.layer[10].state_dict())
    # model.encoder.layer[1].load_state_dict(bert_base_cased.encoder.layer[11].state_dict())


    # num_params = sum(p.numel() for p in model.pooler.parameters())
    # print(num_params)
    # num_params2 = sum(p.numel() for p in model.parameters())
    # print(num_params2)




    print(bert_base_uncased.encoder.layer[10:12])
    bt = BertTop()
    bts = BertTopSeq()
    a = torch.randn((128, 64, 768))
    print(a.shape)
    y = bt(a) #bt(a)
    y2 = bts(a)
    print('HHHHHHHHHHHHHHHH', y.shape, y2.shape)
    print(bt)


    torch.save(bt, 'bt4_L11.pt')
    torch.save(bts, 'bts4_L11.pt')
    bbt = torch.load('bt4_L11.pt')
    bbts = torch.load('bts4_L11.pt')

    yy = bbt(a)
    ybts = bbts(a)
    print('TTTTTTTTTTT', yy.shape, ybts.shape)






