import time
from dataclasses import asdict

import torch
import torch.optim as optim
from torch import vmap

from models import Student_model, Teacher_model
from utils import get_datas, make_dataloader, sigmoid
from setup import setup

# model training
def train(epoch):
    # torch.cuda.synchronize()
    time_ep = time.time()
    model.train()
    total = 0
    train_loss = 0.0
    correct = 0.0    
    
    for batch_idx, (inputs, target) in enumerate(train_loader):
        # print(inputs, target)
        target = target.to(torch.float32)
        input_size = inputs.shape[0]
        outputs = vmap(sigmoid)(model(inputs))
        # print(f'outputs = {outputs}')
        # pred = torch.where(outputs >= 0, 1, -1)
        pred = torch.where(outputs >= 0.5, 1, 0)

        loss = loss_func(outputs, target)
        # print(loss)
        train_loss += loss.data
        correct += pred.eq(target).sum()
        total += input_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # torch.cuda.synchronize()   
    time_ep = time.time() - time_ep
    return train_loss/(batch_idx+1), 100*correct/total, time_ep

# model test
def test(epoch):
    model.eval()
    total = 0
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            target = target.to(torch.float32)
            input_size = inputs.shape[0]
            outputs = vmap(sigmoid)(model(inputs))
            # pred = torch.where(outputs >= 0, 1, -1)
            pred = torch.where(outputs >= 0.5, 1, 0)

            loss = loss_func(outputs, target)
            test_loss += loss.data
            correct += pred.eq(target).sum()
            total += input_size

    return test_loss/(batch_idx+1), 100*correct/total

sup = setup()
s = asdict(sup)
for key, val in s.items():
    print(f'{key}: {val}')

model = Student_model(sup.data_dimention, sup.mid_layer_size, sup.model_type)
data = get_datas(sup.train_size, sup.data_dimention, sup.noise_std)
train_loader = make_dataloader(data, sup.batch_size)

data = get_datas(sup.test_size, sup.data_dimention, sup.noise_std)
test_loader = make_dataloader(data, sup.batch_size)

# loss_func = torch.nn.HingeEmbeddingLoss()
# loss_func = vmap(smoothed_hinge)()
loss_func = torch.nn.BCELoss()
if sup.model_type == 'mf':
    c = 1
elif sup.model_type == 'ntk':
    c = 0
#learning rateの決定
learning_rate = sup.eta * (sup.mid_layer_size ** c)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=sup.momentum, weight_decay=sup.weight_decay)

for epoch in range(sup.epochs):
    train_loss, train_acc, time_ep = train(epoch)
    print("epoch", epoch+1, "lr:{:.4f}".format(optimizer.param_groups[0]['lr']), " train_loss:{:.5f}".format(train_loss), "train_acc:{:.2f}".format(train_acc), "time:{:.3f}".format(time_ep))

    if (epoch+1)%5 == 0:
        test_loss, test_acc = test(epoch)
        print('test_loss:{:.5f}'.format(test_loss), 'test_acc:{:.2f}'.format(test_acc))

torch.save(model.state_dict(), f'results/trained_model_dim={sup.data_dimention}_regime:{sup.model_type}.pth')