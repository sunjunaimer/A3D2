

import torch
from torch import nn
import torch.nn.functional as F





class BertTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = model.pooler
    
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        sequence_output = encoder_outputs[0]
        output = self.pooler(sequence_output)
        # output = F.normalize(output, p=2, dim=1)

        return output
    

class BertTopSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = model.encoder
        self.pooler = model.pooler
    
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        #sequence_output = encoder_outputs[0]
        #output = self.pooler(sequence_output)
        # output = F.normalize(output, p=2, dim=1)

        return output

if __name__ == '__main__':
    # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    # print(config)
    # print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    # import transformers
    # #print(transformers.__file__)

    # print("CCCCCCCCCCCCCCCCCCCCCCCCCCC")
    # print(bert_base_cased)
    # print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
    # print(model.encoder)

    # model.encoder.layer[0].load_state_dict(bert_base_cased.encoder.layer[10].state_dict())
    # model.encoder.layer[1].load_state_dict(bert_base_cased.encoder.layer[11].state_dict())


    # num_params = sum(p.numel() for p in model.pooler.parameters())
    # print(num_params)
    # num_params2 = sum(p.numel() for p in model.parameters())
    # print(num_params2)



    # print('QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ')
    # bt = BertTop()
    a = torch.randn((128, 64, 768))

    bbt = torch.load('bt.pt')

    yy = bbt(a)
    print('TTTTTTTTTTT', yy.shape)