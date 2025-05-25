import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        
        # embd = F.normalize(embd, p=2, dim=1)
        return embd






class TextCNN2(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        #self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        #max_out3 = self.conv_block(frame_x, self.conv3)
        #all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        all_out = torch.cat((max_out1, max_out2), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        
        embd = F.normalize(embd, p=2, dim=1)
        return embd

    


class TextCNN3(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=3, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        #self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, self.in_channels, seq_len, -1)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        #max_out3 = self.conv_block(frame_x, self.conv3)
        #all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        all_out = torch.cat((max_out1, max_out2), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        
        # embd = F.normalize(embd, p=2, dim=1)
        return embd






class BERTCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        #self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )
        self.bts = torch.load('/data3/sunjun/work/code/DA/BADA1109/models/networks/bts.pt')

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        frame_x = self.bts(frame_x)
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        #max_out3 = self.conv_block(frame_x, self.conv3)
        #all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        all_out = torch.cat((max_out1, max_out2), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        
        embd = F.normalize(embd, p=2, dim=1)
        return embd




    

class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    https://github.com/jungwhank/rcnn-text-classification-pytorch/blob/master/model.py
    """
    def __init__(self, embedding_dim, hidden_size, hidden_size_linear, dropout=0.0):
        super(RCNN, self).__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = nn.Linear(embedding_dim + 2*hidden_size, hidden_size_linear)
        self.tanh = nn.Tanh()
        #self.fc = nn.Linear(hidden_size_linear, class_num)

    def forward(self, x):
        # x = |bs, seq_len|
        #x_emb = self.embedding(x)
        # x_emb = |bs, seq_len, embedding_dim|
        x = x[:, :, 2*768: 3*768]
        output, _ = self.lstm(x)
        # output = |bs, seq_len, 2*hidden_size|
        output = torch.cat([output, x], 2)
        # output = |bs, seq_len, embedding_dim + 2*hidden_size|
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = |bs, seq_len, hidden_size_linear| -> |bs, hidden_size_linear, seq_len|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = |bs, hidden_size_linear|
        #output = self.fc(output)
        # output = |bs, class_num|
        return output






class ResCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5,7], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        # self.conv1 = nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0)   
        # self.conv2 = nn.Conv2d(32, 64, (kernel_heights[1], 128), stride=1, padding=0)
        # self.conv3 = nn.Conv2d(64, out_channels, (kernel_heights[2], 578), stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout)
        # self.embd = nn.Sequential(
        #     nn.Linear(len(kernel_heights)*out_channels, embd_size),
        #     nn.ReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, (kernel_heights[1], 128), stride=1, padding=0), 
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, (kernel_heights[2], 578), stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=2),

        )
        self.linear = nn.Linear(in_features=3328, out_features=embd_size, bias=True)   # (3,5,7)
        #self.linear = nn.Linear(in_features=536384, out_features=embd_size, bias=True)   #(3, 5)

  

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        x = x.view(batch_size, 1, seq_len, -1)
        output = self.conv(x)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        
        embd = self.linear(output)
        embd = F.normalize(embd, p=2, dim=1)
        return embd





class ResCNN2(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        # self.conv1 = nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0)   
        # self.conv2 = nn.Conv2d(32, 64, (kernel_heights[1], 128), stride=1, padding=0)
        # self.conv3 = nn.Conv2d(64, out_channels, (kernel_heights[2], 578), stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout)
        # self.embd = nn.Sequential(
        #     nn.Linear(len(kernel_heights)*out_channels, embd_size),
        #     nn.ReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, (kernel_heights[1], 705), stride=1, padding=0), 
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=2),

        )
        #self.linear = nn.Linear(in_features=3328, out_features=embd_size, bias=True)   # (3,5,7)
        self.linear = nn.Linear(in_features=1856, out_features=embd_size, bias=True)   #(3, 5)

  

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        x = x.view(batch_size, 1, seq_len, -1)
        output = self.conv(x)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        
        embd = self.linear(output)
        embd = F.normalize(embd, p=2, dim=1)
        return embd





class ResCNN22(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        # self.conv1 = nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0)   
        # self.conv2 = nn.Conv2d(32, 64, (kernel_heights[1], 128), stride=1, padding=0)
        # self.conv3 = nn.Conv2d(64, out_channels, (kernel_heights[2], 578), stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout)
        # self.embd = nn.Sequential(
        #     nn.Linear(len(kernel_heights)*out_channels, embd_size),
        #     nn.ReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, (kernel_heights[0], 64), stride=1, padding=0),
            nn.BatchNorm2d(num_features=16, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, (kernel_heights[1], 705), stride=1, padding=0), 
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=2),

        )
        #self.linear = nn.Linear(in_features=3328, out_features=embd_size, bias=True)   # (3,5,7)
        self.linear = nn.Linear(in_features=928, out_features=embd_size, bias=True)   #(3, 5)

  

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        x = x.view(batch_size, 1, seq_len, -1)
        output = self.conv(x)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        
        embd = self.linear(output)
        embd = F.normalize(embd, p=2, dim=1)
        return embd



class ResCNN23(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        # self.conv1 = nn.Conv2d(in_channels, 32, (kernel_heights[0], 64), stride=1, padding=0)   
        # self.conv2 = nn.Conv2d(32, 64, (kernel_heights[1], 128), stride=1, padding=0)
        # self.conv3 = nn.Conv2d(64, out_channels, (kernel_heights[2], 578), stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout)
        # self.embd = nn.Sequential(
        #     nn.Linear(len(kernel_heights)*out_channels, embd_size),
        #     nn.ReLU(inplace=True),
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, (kernel_heights[0], 64*3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=16, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, (kernel_heights[1], 768-64*3+1), stride=1, padding=0), 
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=2),

        )
        #self.linear = nn.Linear(in_features=3328, out_features=embd_size, bias=True)   # (3,5,7)
        self.linear = nn.Linear(in_features=928, out_features=embd_size, bias=True)   #(3, 5)

  

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        x = x.view(batch_size, 1, seq_len, -1)
        output = self.conv(x)
        output = output.view(-1, output.shape[1]*output.shape[2]*output.shape[3])
        
        embd = output #self.linear(output)
        #embd = F.normalize(embd, p=2, dim=1)
        return embd


if __name__ == '__main__':
    rc = ResCNN23(768)
    x = torch.randn((128, 64, 768))
    # batch_size, seq_len, feat_dim = x.size()
    # x = x.view(batch_size, 1, seq_len, -1)
    

    #x1 = rc.conv_block(x, rc.conv1)
    # x1 = rc.conv1(x)
    # x2 = rc.conv2(x1)
    # print('AAAAAAAAAAAA:', x.shape, x1.shape, x2.shape)
    #max_out2 = self.conv_block(frame_x, self.conv2)
    output = rc(x)
    print('AAAAAAAAAAA:', output.shape)
    #print(rc)
    num_params = sum(p.numel() for p in rc.parameters())
    print(num_params)
