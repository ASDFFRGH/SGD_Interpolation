import torch
from torch import nn, vmap

# inputの次元は適宜変換する
#Teacher model　中間層の幅
class Teacher_model(nn.Module):
    def __init__(self, d):
        super ().__init__()
        self.d = d
        self.m = 128
        self.linear_relu = nn.Sequential(
            nn.Linear(d, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1),
        )

    def forward(self, x):
        logit = self.linear_relu(x) / self.m
        return logit
        
class Student_model(nn.Module):
    def __init__(self, d, m, model_type):
        super().__init__()
        self.model_type = model_type
        self.m = m
        self.linear_relu = nn.Sequential(
            nn.Linear(d, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1),
        )
        
        if self.model_type == 'ntk' or self.model_type == 'mf': print(f'Learning regime is {self.model_type}')
        else: exit(print('モデルタイプが正しく設定されていません'))
    
    def forward(self, x):
        if self.model_type == 'ntk':  logit = self.linear_relu(x)/(self.m**0.5)
        elif self.model_type == 'mf': logit = self.linear_relu(x)/self.m
        return logit
        
if __name__ == '__main__':
    import os
    import math
    def sigmoid(a):
        e = math.e
        s = 1 / (1 + e**-a)
        return s
    
    d=10
    teacher_model = Teacher_model(d)
    print(teacher_model)
    for p in teacher_model.parameters():
        # print(p)
        teacher_param = torch.normal(0, 5, p.shape)
        p.data = teacher_param
        # p.data = torch.ones_like(p)
        # print(p)
    
    # print(teacher_model)
    # print(list(teacher_model.parameters()))

    # student_model = Student_model(d, 8, 'ntk')
    # st_model2 = Student_model(d, 1024, 'ntk')
    
    # print(student_model)
    # print(list(student_model.parameters()))
    # print(st_model2.model_type)

    # X = torch.normal(0, 10**3, size=(10, d))
    X = torch.normal(0, 1, size=(10, d)) 
    print(X)
    print(teacher_model(X))
    print(vmap(sigmoid)(teacher_model(X)))
    # print(st_model2(X))
    # print(st_model3(X))
    