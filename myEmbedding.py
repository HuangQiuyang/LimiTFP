# coding:utf-8
#import networkx as nx
#from node2vec import Node2Vec
import pandas as pd
from itertools import combinations
import torch
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import datetime

'''
def spatial_embedding():
    # Create a graph 这里可以给出自己的graph
    road = pd.read_csv('./data/road1.csv')
    print('start graph')
    g = nx.Graph()
    g.add_nodes_from(list(road['FID']))  # 将路段当成节点，加入图中
    edgeslist =[]
    #根据nodes字段生成边
    #nodelist = list(set(road['nodes1']).union(set(road['nodes2'])))
    nodelist = list(set(road['from']).union(set(road['to'])))

    for i in nodelist:
        temp = list(road[(road['from']==i)|(road['to']==i)]['FID'])
        edgeslist.extend(list(combinations(temp, 2)))
    g.add_edges_from(edgeslist)
    print('start node2vec')
    node2vec = Node2Vec(g, dimensions=64, walk_length=10,p=1,q=0.1, num_walks=20, workers=8)
    print('start fit')
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=20)
    # Save embeddings for later use
    print('start save embedding')
    model.wv.save_word2vec_format('./data/road_embedding1.txt')
    # Save model for later use
    model.save('embeddingmodel')
    # 12756 128
'''   
    
class EmbedNet(nn.Module):
    def __init__(self, n_input, d_hidden, n_output):
        super(EmbedNet, self).__init__()
        #self.bn_input = nn.BatchNorm1d(n_input, momentum=0.5)   # 给 input 的 BN

        self.encoder = nn.Sequential(
            
            nn.Linear(n_input, d_hidden),
                    )
        self.decoder = nn.Sequential(
            
            nn.Linear(d_hidden, n_output),
            #nn.Softmax()
            #nn.ReLU(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        #x = self.bn_input(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded


#道路属性embedding
def attr_embedding_train(n_input, d_hidden, n_output):
    road = pd.read_csv('./data/road1.csv')
    print('start...')
    net = EmbedNet(n_input, d_hidden, n_output)
    
    data = torch.tensor(road.values).float()


    type_onehot = torch.zeros(len(road),16).scatter_(1,data[:,1].unsqueeze(1).long(),1).float()
    
    x = torch.cat((type_onehot,data[:,2:]),1)
    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    
    #best_fit = 0
    #best_loss = 
    for epoch in range(80):

        encoded,out = net(x)

        loss = loss_func(out, x)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        print(loss)
    
    '''
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight
    '''
    return encoded,out

'''
#时间embedding
def time_embedding_train(day,t_interval,t_num):

    #t_interval = 15#时间片长度 min
    #t_num = 4#前n个时间片，前n天的t时刻，前n周的t时刻
    t_list = []
    #d2 = d1 + datetime.timedelta(hours = 8)
    #day = datetime.datetime(2017,6,1)  #a.__format__('%Y-%m-%d')  a.strftime("%Y%m%d")
    #week = day.isoweekday()
    for i in range(30):
        date = day + datetime.timedelta(days = i)
        
        for t in range(int(24*60/t_interval)):
            week = date.isoweekday()
            hour = int(((t+1)*t_interval-1)/60)
            is_weekend = 0
            if week >5:
                is_weekend = 1
                
            t_list.append([week-1,hour,is_weekend,t])
            
    
    data = torch.tensor(t_list).float().unsqueeze(1)

    num = len(t_list)
    week_onehot = torch.zeros(num,7).scatter_(1,data[:,0].unsqueeze(1).long(),1).float()
    day_onehot = torch.zeros(num,7*t_num).scatter_(1,data[:,1].unsqueeze(1).long(),1).float()

    hour_onehot = torch.zeros(num,24).scatter_(1,data[:,2].unsqueeze(1).long(),1).float()
    is_weekend_onehot = torch.zeros(num,2).scatter_(1,data[:,3].unsqueeze(1).long(),1).float()
    t_onehot = torch.zeros(num,int(24*60/t_interval)).scatter_(1,data[:,4].unsqueeze(1).long(),1).float()
    
    x = torch.cat((week_onehot,day_onehot,hour_onehot,is_weekend_onehot,t_onehot),1)
    
    
    net = EmbedNet(x.size()[1], 64, x.size()[1])
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    
    #best_fit = 0
    #best_loss = 
    for epoch in range(80):

        encoded,out = net(x)

        loss = loss_func(out, x)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        print(loss)

    return net
'''
  
#external factors embedding
def external_embedding_train(n_input, d_hidden, n_output):
    road = pd.read_csv('./data/poi.csv', header=None)
    print('start...')
    net = EmbedNet(n_input, d_hidden, n_output)
    
    x = torch.tensor(road.iloc[:,1:].values).float()
    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    
    #best_fit = 0
    #best_loss = 
    for epoch in range(800):

        encoded,out = net(x)

        loss = loss_func(out, x)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        print(loss)
    
    '''
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight
    '''
    return encoded,out


'''
if __name__ == '__main__':
    #encoded,out = attr_embedding_train(0,128,64,128)
    #print(out)
    #spatial_embedding()
    #spatial_embedding()
    #t = encoded.detach().numpy()
    net = time_embedding_train(129, 64, 129)
    #result = encoded.detach().numpy()
    #encoded,out = external_embedding_train(17, 64, 17)
'''


