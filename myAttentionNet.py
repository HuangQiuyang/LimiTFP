# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim



class AttentionNet(nn.Module):
    def __init__(self,L,T,K,P,D,M_size):
        super(AttentionNet, self).__init__()
        
        self.L = L#100
        self.T = T#100
        self.K = K
        self.P = P
        self.D = D
        
        
        ########################################################
        #The Variable of the first attention layer(totally K layers):
        self.Vs_1 = Variable(torch.rand(P,P),requires_grad=True)
        self.Ws1_1 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        
        self.Vt_1 = Variable(torch.rand(P,P),requires_grad=True)
        self.Wt1_1 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        self.Wt3_1 = Variable(torch.randn(D,D),requires_grad=True)
        
        self.Ve_1 = Variable(torch.rand(P,P),requires_grad=True)
        self.We1_1 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        self.We3_1 = Variable(torch.rand(D,D),requires_grad=True)
        
        
        self.spatial_1 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(2, P),
            nn.Tanh()
        )
        
        self.temporal_1 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, P),
            nn.Tanh()
        )
        
        self.external_1 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, P),
            nn.Tanh()
        )
        
        self.fusion_1 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, 1),
            #nn.ReLU()
            nn.ReLU()
        )
        
        ############################################################
        
        ############################################################
        #The Variable of the second attention layer(totally K layers):
        self.Vs_2 = Variable(torch.rand(P,P),requires_grad=True)
        self.Ws1_2 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        
        self.Vt_2 = Variable(torch.rand(P,P),requires_grad=True)
        self.Wt1_2 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        self.Wt3_2 = Variable(torch.rand(D,D),requires_grad=True)
        
        self.Ve_2 = Variable(torch.rand(P,P),requires_grad=True)
        self.We1_2 = Variable(torch.rand(2*D,2*D),requires_grad=True)
        self.We3_2 = Variable(torch.rand(D,D),requires_grad=True)
        
        
        self.spatial_2 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(2, P),
            nn.Tanh()
        )
        
        self.temporal_2 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, P),
            nn.Tanh()
        )
        
        self.external_2 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, P),
            nn.Tanh()
        )
        
        self.fusion_2 = nn.Sequential(
            nn.BatchNorm2d(M_size, momentum=0.5),
            nn.Linear(3, 1),
            #nn.ReLU()
            nn.ReLU()
        )
        ######################################
        
        
        #######################################
        # Fully-connected Layer 1-1
        self.fc1_1 = nn.Sequential(
            nn.BatchNorm1d(L*P+4, momentum=0.5),
            nn.Linear(L*P+4, P),
            #nn.Linear(80, 50),
            #nn.Linear(50, P),
            nn.ReLU()
        )
        
        # Fully-connected Layer 1-2
        self.fc1_2 = nn.Sequential(
            nn.BatchNorm1d(L*P+4, momentum=0.5),
            nn.Linear(L*P+4, P),
            #nn.Linear(80, 50),
            #nn.Linear(50, P),
            nn.ReLU()
        )
        
        # Fully-connected Layer 2
        #self.fc2_1 = nn.Linear(P*K, 1)
        self.fc2_1 = nn.Sequential(
                nn.BatchNorm1d(P*K, momentum=0.5),
                nn.Linear(P*K, 1),
                nn.ReLU()
        )
    
    #X: [Batch,L,T,K] return:[Batch,L,P]
    def attention_net(self, X1,X2, e_target, t_target, f_target, E_vecs, T_vecs, F_vecs):
        L = self.L
        T = self.T
        
        BATCH_SIZE = len(e_target)
        '''
        #traget vector
        e_target = torch.randn(BATCH_SIZE,1,2*D)
        t_target = torch.randn(BATCH_SIZE,1,D)
        f_target = torch.randn(BATCH_SIZE,1,D)
        
        #mornitoring roads vectors
        E_vecs = torch.randn(L,2*D)
        T_vecs = torch.randn(T,D)
        F_vecs = torch.randn(L,D)
        '''
        
        
        ##########################################################################
        #The first attention layer
        data = X1.unsqueeze(3)#X[:,:,:,0].unsqueeze_(3) #[Batch,L,T,1]
        
        '''
        #spatial attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        spatial_correlation = torch.matmul(torch.matmul(e_target,self.Ws1_1),E_vecs.transpose(0,1))
        
        #[BATCH_SIZE, L, T, 1]
        spatial_expand = spatial_correlation.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)

        #[BATCH_SIZE, L, T, P]
        spatial_tanh = self.spatial_1(torch.cat((spatial_expand,data),3))
        
        #[Batch, L, T, P]
        score_s = torch.matmul(self.Vs_1,spatial_tanh.transpose(2,3)).transpose(2,3)
        
        #[Batch, L, T, P]
        score_s_softmax = F.softmax(score_s,dim=1)
        
        '''
        #temporal attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        temporal_s = torch.matmul(torch.matmul(e_target,self.Wt1_1),E_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        temporal_s_expand = temporal_s.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, 1, T]
        temporal_correlation = torch.matmul(torch.matmul(t_target,self.Wt3_1),T_vecs.transpose(1,2))
        #[BATCH_SIZE, L, T, 1]
        temporal_correlation_expand = temporal_correlation.expand(BATCH_SIZE,L,T).unsqueeze(3)
        #[Batch, L, T, P]
        temporal_tanh = self.temporal_1(torch.cat((temporal_s_expand,temporal_correlation_expand,data),3))
        #[Batch, L, T, P]
        score_t = torch.matmul(self.Vt_1,temporal_tanh.transpose(2,3)).transpose(2,3)
        score_t_softmax = F.softmax(score_t,dim=2)
        
        '''
        #external factors attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        external_s = torch.matmul(torch.matmul(e_target,self.We1_1),E_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        external_s_expand = external_s.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, 1, L]
        external_correlation = torch.matmul(torch.matmul(f_target,self.We3_1),F_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        external_correlation_expand = external_correlation.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, L, T, P]
        external_tanh = self.external_1(torch.cat((external_s_expand,external_correlation_expand,data),3))
        #[Batch, L, T, P]
        score_e = torch.matmul(self.Vt_1,external_tanh.transpose(2,3)).transpose(2,3)
        score_e_softmax = F.softmax(score_e,dim=1)
        
        
        hs = torch.mul(score_s_softmax, data).sum(2).unsqueeze(3)

        ht = torch.mul(score_t_softmax, data).sum(2).unsqueeze(3)
        
        he = torch.mul(score_e_softmax, data).sum(2).unsqueeze(3)

        h1_1 = self.fusion_1(torch.cat((hs,ht,he),3)).squeeze(3)
        
        
        ##########################################################################
        """
        #The second attention layer
        """
        data = X2.unsqueeze(3)  #[Batch,L,T,1]
        
        '''
        #spatial attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        spatial_correlation = torch.matmul(torch.matmul(e_target,self.Ws1_2),E_vecs.transpose(0,1))
        
        #[BATCH_SIZE, L, T, 1]
        spatial_expand = spatial_correlation.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)

        #[BATCH_SIZE, L, T, P]
        spatial_tanh = self.spatial_2(torch.cat((spatial_expand,data),3))
        
        #[Batch, L, T, P]
        score_s = torch.matmul(self.Vs_2,spatial_tanh.transpose(2,3)).transpose(2,3)
        
        #[Batch, L, T, P]
        score_s_softmax = F.softmax(score_s,dim=1)
        
        '''
        #temporal attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        temporal_s = torch.matmul(torch.matmul(e_target,self.Wt1_2),E_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        temporal_s_expand = temporal_s.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, 1, T]
        temporal_correlation = torch.matmul(torch.matmul(t_target,self.Wt3_2),T_vecs.transpose(1,2))
        #[BATCH_SIZE, L, T, 1]
        temporal_correlation_expand = temporal_correlation.expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, L, T, P]
        temporal_tanh = self.temporal_2(torch.cat((temporal_s_expand,temporal_correlation_expand,data),3))
        #[Batch, L, T, P]
        score_t = torch.matmul(self.Vt_2,temporal_tanh.transpose(2,3)).transpose(2,3)
        score_t_softmax = F.softmax(score_t,dim=2)
        
        '''
        #external factors attention calculation
        '''
        #[BATCH_SIZE, 1, L]
        external_s = torch.matmul(torch.matmul(e_target,self.We1_2),E_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        external_s_expand = external_s.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, 1, L]
        external_correlation = torch.matmul(torch.matmul(f_target,self.We3_2),F_vecs.transpose(0,1))
        #[BATCH_SIZE, L, T, 1]
        external_correlation_expand = external_correlation.transpose(1,2).expand(BATCH_SIZE,L,T).unsqueeze(3)
        
        #[Batch, L, T, P]
        external_tanh = self.external_2(torch.cat((external_s_expand,external_correlation_expand,data),3))
        #[Batch, L, T, P]
        score_e = torch.matmul(self.Vt_2,external_tanh.transpose(2,3)).transpose(2,3)
        score_e_softmax = F.softmax(score_e,dim=1)
        
        hs = torch.mul(score_s_softmax, data).sum(2).unsqueeze(3)

        ht = torch.mul(score_t_softmax, data).sum(2).unsqueeze(3)
        
        he = torch.mul(score_e_softmax, data).sum(2).unsqueeze(3)

        h1_2 = self.fusion_2(torch.cat((hs,ht,he),3)).squeeze(3)
        
        
            
        return h1_1, h1_2

    def forward(self, X1,X2, e_target, t_target, f_target, E_vecs, T_vecs, F_vecs,time_info_tensor):
        
        K = self.K
        P = self.P
        BATCH_SIZE = len(e_target)
        #X=torch.randn(BATCH_SIZE,L,T,K)
        h1_1, h1_2 = self.attention_net(X1,X2, e_target, t_target, f_target, E_vecs, T_vecs, F_vecs)
        
        #time_info_tensor = torch.tensor(time_info).unsqueeze(0).expand(BATCH_SIZE,5).float()
        
        x1_1 = self.fc1_1(torch.cat((h1_1.view(BATCH_SIZE,-1),time_info_tensor),1))
        x1_2 = self.fc1_2(torch.cat((h1_2.view(BATCH_SIZE,-1),time_info_tensor),1))
        
        #x1_1 = self.fc1_1(h1_1.view(BATCH_SIZE,-1))
        #x1_2 = self.fc1_2(h1_2.view(BATCH_SIZE,-1))
        
        x_c = torch.cat((x1_1, x1_2), 1)
        x_c = x_c.view(-1,P*K)
        
        #x_c = torch.cat((x_c,time_info_tensor),1)
        out = self.fc2_1(x_c)
        
        
        return out






