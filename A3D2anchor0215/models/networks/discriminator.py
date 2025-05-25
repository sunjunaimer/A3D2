import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import ReverseLayerF

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, re_coe=1):
        super(DomainDiscriminator, self).__init__()
        self.re_coe = re_coe

        # self.discriminator = nn.Sequential(
        #     nn.Linear(feature_dim, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Linear(512, 512),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 2),
        #     #nn.Sigmoid(),
        # )

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 12),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(12, 2),
            #nn.Sigmoid(),
        )



    def forward(self, feature, alpha):
        reversed_input = ReverseLayerF.apply(feature, alpha*self.re_coe)
        #reversed_input = feature
        output = self.discriminator(reversed_input)
        return output



class DomainDiscriminator43(nn.Module):
    def __init__(self, feature_dim, re_coe=1):
        super(DomainDiscriminator43, self).__init__()
        self.re_coe = re_coe

        # self.discriminator = nn.Sequential(
        #     nn.Linear(feature_dim, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Linear(512, 512),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 2),
        #     #nn.Sigmoid(),
        # )

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 12),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(12, 3),
            #nn.Sigmoid(),
        )



    def forward(self, feature, alpha):
        reversed_input = ReverseLayerF.apply(feature, alpha*self.re_coe)
        #reversed_input = feature
        output = self.discriminator(reversed_input)
        return output





class DomainDiscriminator2(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminator2, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2),
            #nn.Sigmoid(),
        )



    def forward(self, feature, alpha):
        reversed_input = ReverseLayerF.apply(feature, alpha)
        output = self.discriminator(reversed_input)
        return output





class DomainDiscriminatorSig(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminatorSig, self).__init__()

        # self.discriminator = nn.Sequential(
        #     # nn.Linear(in_features=feature_dim, out_features=384),
        #     # nn.BatchNorm1d(384),
        #     # nn.ReLU(),
        #     nn.Linear(in_features=feature_dim, out_features=64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=2),
        #     #nn.Softmax(dim=1)   #nn.LogSoftmax(dim=1)
        # )


        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, feature, alpha):
        reversed_input = ReverseLayerF.apply(feature, alpha)
        output = self.discriminator(reversed_input)
        output = output.view(output.shape[0])
        return output






class DomainDiscriminatorS(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminatorS, self).__init__()

        # self.discriminator = nn.Sequential(
        #     # nn.Linear(in_features=feature_dim, out_features=384),
        #     # nn.BatchNorm1d(384),
        #     # nn.ReLU(),
        #     nn.Linear(in_features=feature_dim, out_features=64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=2),
        #     #nn.Softmax(dim=1)   #nn.LogSoftmax(dim=1)
        # )


        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2),
            #nn.Sigmoid(),
        )


        # self.discriminator = nn.Sequential(
        #     nn.Linear(feature_dim, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 2),
        # )
    

    def forward(self, feature, alpha):
        reversed_input = ReverseLayerF.apply(feature, 0.5*alpha)
        output = self.discriminator(reversed_input)

        #output = self.discriminator(feature)
        return output

