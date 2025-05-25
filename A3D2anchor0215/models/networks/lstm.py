import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='attention'):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze(0)

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)

        # embd = F.normalize(embd, p=2, dim=1)
        return embd




class LSTMEncoder2(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, number_layers=1, embd_method='attention'):
        super(LSTMEncoder2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=number_layers, bidirectional=False, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):

        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze(0)

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        x = x[:, :, 768*2:768*3]
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)

        embd = F.normalize(embd, p=2, dim=1)
        return embd





class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size * num_layers, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        
        h = self.dropout(final_states[0].view(1, final_states[0].shape[1], -1).squeeze(0))
        y_1 = self.linear_1(h)
        return y_1