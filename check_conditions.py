import os
from dataclasses import asdict

import torch
import torch.optim as optim
from torch import vmap

from models import Student_model
from utils import get_datas, make_dataloader, sigmoid
from setup import setup

sup = setup()
s = asdict(sup)
for key, val in s.items():
    print(f'{key}: {val}')

#modelのインストール
with torch.no_grad():
        model = Student_model(sup.data_dimention, sup.mid_layer_size, sup.model_type)
        stc_model = Student_model(sup.data_dimention, sup.mid_layer_size, sup.model_type)
        det_model = Student_model(sup.data_dimention, sup.mid_layer_size, sup.model_type)
        if os.path.isfile(f'teacher_models/teacher_model_weight_input_dim={sup.data_dimention}.pth'):
            model.load_state_dict(torch.load(f'teacher_models/teacher_model_weight_input_dim={sup.data_dimention}.pth'))
            stc_model.load_state_dict(torch.load(f'teacher_models/teacher_model_weight_input_dim={sup.data_dimention}.pth'))
            det_model.load_state_dict(torch.load(f'teacher_models/teacher_model_weight_input_dim={sup.data_dimention}.pth'))

data = get_datas(sup.train_size, sup.data_dimention, sup.noise_std)
train_loader = make_dataloader(data, sup.batch_size)

model.train()
stc_model.eval()
det_model.eval()
loss_func = torch.nn.BCELoss()
 
stc_inputs, stc_target = next(iter(train_loader))    

det_inputs, det_target = data.X, data.t
stc_target = stc_target.to(torch.float32)
det_target = det_target.to(torch.float32)
print(det_target)
print(stc_inputs.size(), stc_target.size())
print(det_inputs.size(), det_target.size())

stc_outputs = vmap(sigmoid)(stc_model(stc_inputs))
det_outputs = vmap(sigmoid)(det_model(det_inputs))
det_loss = loss_func(det_outputs, det_target)
stc_loss = loss_func(stc_outputs, stc_target)
det_loss.backward()
stc_loss.backward()
for p, stc_p, det_p in zip(model.parameters(), stc_model.parameters(), det_model.parameters()):
    epsilon = stc_p.grad.detach() - det_p.grad.detach()
    p = p - sup.eta*epsilon
