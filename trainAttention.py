# -*- coding: utf-8 -*-
import torch
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim
import datetime
import pandas as pd
import os
import time
import random

from myEmbedding import EmbedNet
from myAttentionNet import AttentionNet



def get_data(road_id,interval,num,day,t):
    t_interval = interval#时间片长度 min
    t_num = num#每个属性的时间片数量
    date = day
    #d2 = d1 + datetime.timedelta(hours = 8)
    #day = datetime.datetime(2017,6,1)  #a.__format__('%Y-%m-%d')  a.strftime("%Y%m%d")
    #week = day.isoweekday()
    #for i in range(30):
    #    date = day + datetime.timedelta(days = i)
        
    #for t in range(int(24*60/t_interval)):
    
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()
    
    data1 = road_id.copy()#pd.read_csv('./data/whole_road_id.csv', header = None)
    data2 = road_id.copy()#pd.read_csv('./data/whole_road_id.csv', header = None)

    
    for num in range(t_num):
        if t-num-1 >= 0:
            data1[str(num)],data2[str(num)] = read_data(date,(t-num-1),t_interval)
        else:
            data1[str(num)],data2[str(num)] = read_data((date - datetime.timedelta(days = 1)), int(24*60/t_interval + (t-num-1)), t_interval)
    
    for num in range(t_num):
        data1[str(num+t_num)],data2[str(num+t_num)] = read_data((date - datetime.timedelta(days = num+1)),t,t_interval)

        
    for num in range(t_num): 
        data1[str(num+2*t_num)],data2[str(num+2*t_num)] = read_data((date - datetime.timedelta(days = 7*(num+1))),t,t_interval)

    data1['label'],data2['label'] = read_data(date,t,t_interval)
    
            
    return data1,data2


# 返回对应日期对应时间片每条道路的数据
def read_data(date,t,t_interval):
    month = date.month
    day = date.day 
    #path1 = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':month, 'day':day}
    #path2 = './data/speed/2017%(month)02d/2017%(month)02d%(day)02dspeed.csv' %{'month':month, 'day':day}
    
    key = str(month)+','+str(day)
    #data1 = volume_data[key]
    #data2 = speed_data[key] 
    
    data1 = volume_data[key].iloc[:,(t*int(t_interval/5)+1):(t*int(t_interval/5)+1+int(t_interval/5))].sum(1)
    data2 = speed_data[key].iloc[:,(t*int(t_interval/5)+1):(t*int(t_interval/5)+1+int(t_interval/5))].max(1)

    return data1,data2


def read_all_data():
    
    first_day = datetime.datetime(2017,5,1)
    for i in range(90):
        date = first_day + datetime.timedelta(days = i)
        month = date.month
        day = date.day 
        path1 = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':month, 'day':day}
        path2 = './data/speed/2017%(month)02d/2017%(month)02d%(day)02dspeed.csv' %{'month':month, 'day':day}
    
        data1 = pd.read_csv(path1,header=None)
        data2 = pd.read_csv(path2,header=None)
        key = str(month)+','+str(day)
        print('read '+key)
        volume_data[key] = data1
        speed_data[key] = data2
    
    return

#过滤掉没有车通过的路段
def filter_road():
    date = datetime.datetime(2017,5,20)
    path = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':date.month, 'day':date.day}
    data = pd.read_csv(path,header=None).iloc[:,1:]
    sumdata = data
    num = 0
    for i in range(70):
        print(i)
        date = date + datetime.timedelta(days = 1)
        #date = datetime.datetime(2017,6,20)
        if date > datetime.datetime(2017,7,29):
            break
        
        path = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':date.month, 'day':date.day}
        data = pd.read_csv(path,header=None).iloc[:,1:]
        num+=1
        sumdata=sumdata+data
            
    sum_list = sumdata.sum(1)/num
    
    sum_list.to_csv('./data/filter.csv')

#过滤掉没有车通过的路段
def history_ave():
    date = datetime.datetime(2017,5,20)
    path = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':date.month, 'day':date.day}
    data = pd.read_csv(path,header=None)#.iloc[:,1:]
    sumdata = data
    num = 0
    for i in range(70):
        print(i)
        date = date + datetime.timedelta(days = 1)
        #date = datetime.datetime(2017,6,20)
        if date > datetime.datetime(2017,7,29):
            break
        
        path = './data/volumn/2017%(month)02d/2017%(month)02d%(day)02dvolumn.csv' %{'month':date.month, 'day':date.day}
        data = pd.read_csv(path,header=None)#.iloc[:,1:]
        num+=1
        sumdata=sumdata+data
        
    result = pd.DataFrame()
    for i in range(24):
        result[str(i)] = sumdata[12*i+1]
        for j in range(1,12):
            result[str(i)] = result[str(i)] + sumdata[12*i+j+1]
    result = result / num

    
    result.to_csv('./data/history_ave.csv')



def get_sptial_vectors(m_list):
    path = './data/Spatial_vectors.csv' 
    
    #(N,2*D)
    e_target_List = pd.read_csv(path,header=None)
    E_vecs = e_target_List[e_target_List[0].isin(m_list)]
    
    return e_target_List, E_vecs



def get_time_vectors(date,t_interval,t_num,t):
    
    T_num = int(24*60/t_interval)#一天内时间片总数量
    week = date.isoweekday()
    hour = int(( (t+1) * t_interval - 1) / 60)
    is_weekend = 0
    if week >5:
        is_weekend = 1
        
    t_target = [week-1,0,hour,is_weekend,t]
    
    T_vecs = []
    
    T_vecs.append(t_target)
    
    for num in range(t_num):
        if t-num-1 >= 0:
            week = date.isoweekday()
            hour = int(( (t-num) * t_interval - 1) / 60)
            is_weekend = 0
            if week >5:
                is_weekend = 1
                
            T_vecs.append([week-1,0, hour,is_weekend,(t-num-1)])
        else:
            week = (date - datetime.timedelta(days = 1)).isoweekday()
            hour = int(((T_num + t-num)*t_interval-1)/60)
            is_weekend = 0
            if week >5:
                is_weekend = 1
            T_vecs.append([week-1,1,hour,is_weekend ,(T_num + t-num-1)])
    
    for num in range(t_num):
        week = (date - datetime.timedelta(days = num+1)).isoweekday()
        hour = int(((t+1)*t_interval-1)/60)
        is_weekend = 0
        if week >5:
            is_weekend = 1
            
        T_vecs.append([week-1,num+1,hour,is_weekend, t])

        
    for num in range(t_num): 
        week = (date - datetime.timedelta(days =  7*(num+1))).isoweekday()
        hour = int(((t+1)*t_interval-1)/60)
        is_weekend = 0
        if week >5:
            is_weekend = 1
            
            
        T_vecs.append([week-1,7*(num+1),hour,is_weekend, t])
        
    data = torch.tensor(T_vecs).float()

    t_len = len(T_vecs)
    week_onehot = torch.zeros(t_len,7).scatter_(1,data[:,0].unsqueeze(1).long(),1).float()
    day_onehot = torch.zeros(t_len,7*t_num+1).scatter_(1,data[:,1].unsqueeze(1).long(),1).float()
    hour_onehot = torch.zeros(t_len,24).scatter_(1,data[:,2].unsqueeze(1).long(),1).float()
    is_weekend_onehot = torch.zeros(t_len,2).scatter_(1,data[:,3].unsqueeze(1).long(),1).float()
    t_onehot = torch.zeros(t_len,T_num).scatter_(1,data[:,4].unsqueeze(1).long(),1).float()
    
    x = torch.cat((week_onehot,day_onehot,hour_onehot,is_weekend_onehot,t_onehot),1)

    
    return x#t_target,T_vecs


def get_external_vectors(m_list):
    path = './data/external_vectors.csv' 
    
    #(N,1,2*D)
    f_target_List = pd.read_csv(path,header=None)
    F_vecs = f_target_List[f_target_List[0].isin(m_list)]
    
    return f_target_List, F_vecs


def time_embedding_train(day,day_num,t_interval,t_num):
    
    #t_interval = 15#时间片长度 min
    #t_num = 4#前n个时间片，前n天的t时刻，前n周的t时刻
    #d2 = d1 + datetime.timedelta(hours = 8)
    #day = datetime.datetime(2017,6,1)  #a.__format__('%Y-%m-%d')  a.strftime("%Y%m%d")
    #week = day.isoweekday()
    x = torch.tensor([])
    for i in range(day_num):
        date = day + datetime.timedelta(days = i)
        
        for t in range(int(24*60/t_interval)):
            temp = get_time_vectors(date,t_interval,t_num,t)
            x = torch.cat((x,temp),0)
    
    
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
    
    '''
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight
    '''
    return net



    
def train():
    
    interval = 60 #时间片长度
    num = 8 #历史数据长度
    M = int(6404 *0.05) #选择监测路段数量
    #d2 = d1 + datetime.timedelta(hours = 8)
    day = datetime.datetime(2017,7,1)  #a.__format__('%Y-%m-%d')  a.strftime("%Y%m%d")
    
    BATCH_SIZE = 256 
    N = 6404
    
    
    road_list = list(pd.read_csv('./data/road_id.csv', header = None)[0]) #添加道路id
    
    monitoring_road = pd.read_csv('./data/mornitoring_selection.csv', header = None) #监测路段信息
    
    #shuf_m_road = list(range(len(monitoring_road)))
    #random.shuffle(shuf_m_road)A
    
    #m_list = monitoring_road.iloc[0:M,:][0].values.tolist()#list(monitoring_road[monitoring_road[0].isin(shuf_m_road[0:M]) ][0])
    
    m_list = list(monitoring_road[monitoring_road[2]<=M][0]) #选择top M 条监测路段
    
    e_target_List, E_vecs = get_sptial_vectors(m_list) #获取空间embedding 向量
    f_target_List, F_vecs = get_external_vectors(m_list) #获取external embedding 向量
    
    #e_target = torch.tensor(e_target_List.iloc[:,1:].values).unsqueeze(1).float()
    #f_target = torch.tensor(f_target_List.iloc[:,1:].values).unsqueeze(1).float()
    E_vecs = torch.tensor(E_vecs.iloc[:,1:].values).float()
    F_vecs = torch.tensor(F_vecs.iloc[:,1:].values).float()
    
    whole_road_id = pd.read_csv('./data/whole_road_id.csv', header = None) #预测道路
    timeNet = time_embedding_train(datetime.datetime(2017,5,1),60,interval,num)
    
    
    
    attentionNet = AttentionNet(len(m_list),3*num,2,5,64,M)
        
    optimizer = torch.optim.Adam(attentionNet.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    loss_list_train = []
    loss_list_test = []
    
    
    
    
    #开始训练
    for epoch in range(1):
        
        for i in range(7):
            print('train-第 '+str(i)+' 天')
            
            date = day + datetime.timedelta(days = i)
            
            time_slice = list(range(int(24*60/interval)))
            random.shuffle(time_slice)
            
            #for batch_road in range(int(N/BATCH_SIZE)):
            
            X1 = torch.tensor([])
            X2 = torch.tensor([])
            
            label = torch.tensor([])
            
            
            
            e_target = torch.tensor([])
            f_target = torch.tensor([])
            
            t_target = torch.tensor([])
            
            T_vecs = torch.tensor([])
        
            time_info_tensor = torch.tensor([])
        
            for t in time_slice:
                print('train-第 '+str(i)+' 天, 第 ' + str(t)+' 个时间片('+str(int(24*60/interval))+')')
    
                week = date.isoweekday()
                hour = int(( (t+1) * interval - 1) / 60)
                is_weekend = 0
                if week >5:
                    is_weekend = 1
                    
                time_info = [week-1,hour,is_weekend,t]
                
                time_info_tensor = torch.cat((time_info_tensor,torch.tensor(time_info).unsqueeze(0).expand(N,4).float()),0)
            
                
                data1,data2 = get_data(whole_road_id,interval,num,date,t)
    
                
                m_data1 = data1[data1[0].isin(m_list)]
                m_data2 = data2[data1[0].isin(m_list)]
                
                data1 = data1[data1[0].isin(road_list)]
                data2 = data2[data2[0].isin(road_list)]
                
                time_tensor,decode = timeNet(get_time_vectors(date,interval,num,t))
                
    
                X1 = torch.cat((X1,torch.tensor(m_data1.iloc[:,1:-1].values).unsqueeze(0).expand(N,M,3*num).float()),0)
                X2 = torch.cat((X2,torch.tensor(m_data2.iloc[:,1:-1].values).unsqueeze(0).expand(N,M,3*num).float()),0)
                
                label = torch.cat((label,torch.tensor(data1.iloc[:,-1].values).unsqueeze(1).float()),0)
                
                e_target = torch.cat((e_target,torch.tensor(e_target_List.iloc[:,1:].values).unsqueeze(1).float()),0)
                f_target = torch.cat((f_target,torch.tensor(f_target_List.iloc[:,1:].values).unsqueeze(1).float()),0)            
            
                t_target = torch.cat((t_target,time_tensor[0].unsqueeze(0).unsqueeze(1).expand(N,1,64).float()),0)
                
                T_vecs = torch.cat((T_vecs,time_tensor[1:].unsqueeze(0).expand(N,3*num,64).float()),0)
            
            
            data_length = len(label)
            shuffle_list = list(range(data_length))
            
            random.shuffle(shuffle_list)
            
            for batch in range(int(data_length/BATCH_SIZE)):
                start_time = time.time()
                batch_index = shuffle_list[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                
                out = attentionNet(X1[batch_index],X2[batch_index], e_target[batch_index], t_target[batch_index], f_target[batch_index], E_vecs, T_vecs[batch_index], F_vecs,time_info_tensor[batch_index]).float()
        
                loss = loss_func(out, label[batch_index])      # mean square error
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward(retain_graph=True)                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients
                end_time = time.time()
                MAE = abs(out-label[batch_index]).mean()
                MAPE = abs((out-label[batch_index]).int()/(label[batch_index]+1)).mean()
                sum_out = out.sum().long().data
                label_sum = label[batch_index].long().sum().data
                
                loss_list_train.append([i,float(loss.data),float(MAE.data),float(MAPE.data),float(sum_out),float(label_sum)])            
                print(i,batch,round((end_time-start_time),2),int(data_length/BATCH_SIZE),loss.data,MAE.data,MAPE.data,sum_out,label_sum)
                        
    pd.DataFrame(loss_list_train).to_csv('./data/result/test/train-loss-0.05-60min-5k-1.csv')
        
        
    #############################################################################################
    '''test'''
    
    test_day = datetime.datetime(2017,7,15)
    #开始测试
    for i in range(7):
        print('test-第 '+str(i)+' 天')
        
        date = test_day + datetime.timedelta(days = i)
        
        time_slice = list(range(int(24*60/interval)))
        random.shuffle(time_slice)
        
        #for batch_road in range(int(N/BATCH_SIZE)):
        
        X1 = torch.tensor([])
        X2 = torch.tensor([])
        
        label = torch.tensor([])
        road_index = torch.tensor([])
        time_index = torch.tensor([])
            
        e_target = torch.tensor([])
        f_target = torch.tensor([])
        
        t_target = torch.tensor([])
        
        T_vecs = torch.tensor([])
        time_info_tensor = torch.tensor([])
        
    
        for t in time_slice:
            print('test-第 '+str(i)+' 天, 第 ' + str(t)+' 个时间片('+str(int(24*60/interval))+')')

            week = date.isoweekday()
            hour = int(( (t+1) * interval - 1) / 60)
            is_weekend = 0
            if week >5:
                is_weekend = 1
                
            time_info = [week-1,hour,is_weekend,t]
        
            time_info_tensor = torch.cat((time_info_tensor,torch.tensor(time_info).unsqueeze(0).expand(N,4).float()),0)
            
            data1,data2 = get_data(whole_road_id,interval,num,date,t)

            
            m_data1 = data1[data1[0].isin(m_list)]
            m_data2 = data2[data1[0].isin(m_list)]
            
            data1 = data1[data1[0].isin(road_list)]
            data2 = data2[data2[0].isin(road_list)]
            
            time_tensor,decode = timeNet(get_time_vectors(date,interval,num,t))
            

            X1 = torch.cat((X1,torch.tensor(m_data1.iloc[:,1:-1].values).unsqueeze(0).expand(N,M,3*num).float()),0)
            X2 = torch.cat((X2,torch.tensor(m_data2.iloc[:,1:-1].values).unsqueeze(0).expand(N,M,3*num).float()),0)
            
            label = torch.cat((label,torch.tensor(data1.iloc[:,-1].values).unsqueeze(1).float()),0)
        
            road_index = torch.cat((road_index,torch.tensor(data1.iloc[:,0].values).float()),0)
            time_index = torch.cat((time_index,torch.tensor(t).expand(len(data1),1).float()),0)
                
            e_target = torch.cat((e_target,torch.tensor(e_target_List.iloc[:,1:].values).unsqueeze(1).float()),0)
            f_target = torch.cat((f_target,torch.tensor(f_target_List.iloc[:,1:].values).unsqueeze(1).float()),0)            
        
            t_target = torch.cat((t_target,time_tensor[0].unsqueeze(0).unsqueeze(1).expand(N,1,64).float()),0)
            
            T_vecs = torch.cat((T_vecs,time_tensor[1:].unsqueeze(0).expand(N,3*num,64).float()),0)
        
        
        data_length = len(label)
        shuffle_list = list(range(data_length))
        
        random.shuffle(shuffle_list)
        
        out_record=[]
        label_record = []
        
        id_record = []
        time_record = []
        
        for batch in range(int(data_length/BATCH_SIZE)):
            start_time = time.time()
            batch_index = shuffle_list[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            
            out = attentionNet(X1[batch_index],X2[batch_index], e_target[batch_index], t_target[batch_index], f_target[batch_index], E_vecs, T_vecs[batch_index], F_vecs,time_info_tensor[batch_index]).float()
    
            loss = loss_func(out, label[batch_index])      # mean square error
            #optimizer.zero_grad()               # clear gradients for this training step
            #loss.backward(retain_graph=True)                     # backpropagation, compute gradients
            #optimizer.step()                    # apply gradients
            end_time = time.time()
            test_time = end_time-start_time
            MAE = abs(out.int()-label[batch_index].int()).float().mean()
            MAPE = abs((out-label[batch_index]).int()/(label[batch_index]+1)).mean()
            sum_out = out.sum().long().data
            label_sum = label[batch_index].long().sum().data
            
            #out_label_record.append(list(torch.cat((out.data,label[batch_index].data),1)))
            out_record = out_record + list(out.data.view(-1).numpy())
            label_record = label_record + list(label[batch_index].data.view(-1).numpy())
            id_record = id_record + list(road_index[batch_index].data.view(-1).numpy())
            time_record = time_record + list(time_index[batch_index].data.view(-1).numpy())
            
            loss_list_test.append([i,float(loss.data),float(MAE.data),float(MAPE.data),float(sum_out),float(label_sum)])            
            print(i,batch,round(test_time,2),int(data_length/BATCH_SIZE),loss.data,MAE.data,MAPE.data,sum_out,label_sum)
        pd.DataFrame({'id':id_record,'time':time_record,'out':out_record,'label':label_record}).to_csv('./data/result/test/out-label-day-'+str(i)+'.csv')
    pd.DataFrame(loss_list_test).to_csv('./data/result/test/test-loss-0.05-60min-5k-1.csv')
        
    
    return attentionNet,timeNet
            


volume_data = {}
speed_data = {}           
print('read all data...')       
read_all_data() 
print('train .....')
attentionNet,timeNet = train()
