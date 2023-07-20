import pandas as pd
import os
import math

import torch
from torch import vmap
from torch.utils.data import DataLoader

from models import Teacher_model

class Data():
    def __init__(self):
        self.X, self.t = None, None

# 人工データの作成
def sythetic_data(n, d, noise_std=0.0):
    with torch.no_grad():
        model = Teacher_model(d)
        if os.path.isfile(f'teacher_models/teacher_model_weight_input_dim={d}.pth'):
            model.load_state_dict(torch.load(f'teacher_models/teacher_model_weight_input_dim={d}.pth'))
        else:
            for p in model.parameters():
                teacher_param = torch.normal(0, 10, p.shape)
                p.data = teacher_param
            torch.save(model.state_dict(), f'teacher_models/teacher_model_weight_dim={d}.pth')
        noise = torch.normal(0, noise_std, size=(n, 1)) #第三引数はtuple
        X = torch.normal(0, 1, size=(n, d))
        fx = model(X)
        #fx = model(X) + noise #ノイズありの時はまた別の感じになる
        fx = vmap(sigmoid)(fx)
        # label = torch.where(fx >= 0.5, 1, 0) #teacher modelが最後にシグモイド層を持つとき
        # BCELossを使う時のラベルは[0,1]，HingeLossを使う時は[-1,1]
        # label = torch.where(fx >= 0, 1, -1)
        
        label = torch.where(fx >= 0.5, 1, 0)
        # print(label.size())

    return X, label.reshape(-1, 1)

# データローダの作成
def make_dataloader(data, bs):
    dataset = []
    for input, label in zip(data.X, data.t):
        dataset.append((input, label))

    return DataLoader(dataset, batch_size = bs, shuffle = True)

# 人工データの入手
def get_datas(num, dim, noise_std):
    path = f'sythetic_datas/num={num}_dim={dim}_noise_std={noise_std}'
    if os.path.isfile(path):
        data = pd.read_pickle(path)
    else:
        data = Data()
        data.X, data.t = sythetic_data(num, dim, noise_std)
        path = f'sythetic_datas/num={num}_dim={dim}_noise_std={noise_std}'
        # pd.to_pickle(data, path)
    
    return data
    
def sigmoid(a):
    e = math.e
    s = 1 / (1 + e**-a)
    return s
 
def smoothed_hinge(x, y):
    z = x * y
    if z <= 0:
        return 1/2 - z
    elif 0 < z < 1:
        return ((1-z)**2) / 2
    elif 1 <= z:
        return 0 
    
if __name__ == '__main__':
    d = 2
    data = Data()
    model = Teacher_model(d)
 
    # train_num = 2**8
    # test_num = 2**13
    """
    num = 10*d
    dim = 2
    noise_std = 0.1
    data.X, data.t = sythetic_data(num, dim, noise_std)
    
    path = f'sythetic_datas/num={num}_dim={dim}_noise_std={noise_std}'
    pd.to_pickle(data, path)

    train_loader = make_dataloader(data, 3)
    for X, t in train_loader:
        print(X.shape, t)
    """
    X, t = sythetic_data(10, 5, 0.0)
    print(X.size(), t.size())
    