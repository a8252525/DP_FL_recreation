#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
import random
import math
import copy
from tqdm import tqdm
import time
import syft as sy
from torch.utils.tensorboard import SummaryWriter


# In[2]:


class Arguments():
    def __init__(self):
        self.batch_size = 60
        self.test_batch_size = 64
        self.best_lr_list = []
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 5
        self.save_model = False
        self.gamma = 0.1
        self.alpha_max = float
        self.init_alpha_max = 0.1
        self.epsilon = 8
        self.clip_threshold = 0.01
        self.split = 120
        
        
        #federated arg
        self.n_workers = 10
        self.rounds = 20
        self.client_data_number = 600
        

args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# In[3]:


hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
# simulation functions
def connect_to_workers(n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]


workers = connect_to_workers(n_workers=args.n_workers)


# In[4]:


temp = torch.utils.data.DataLoader(
    datasets.MNIST('~/data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size,
    shuffle = True,
    pin_memory = True
)


train_loader = []
for i, (data, target) in tqdm(enumerate(temp)):
    train_loader.append((data.to(device), target.to(device)))

    
#send data to all the client first
train_loader_send = []
for n in range(args.n_workers):
    unit = len(train_loader)//args.n_workers
    if n ==0:
        for (data, target) in train_loader[:unit]:
            train_loader_send.append((data.send(workers[n]), target.send(workers[n])))
    else:
        for (data, target) in train_loader[(n-1)*unit:n*unit]:
            train_loader_send.append((data.send(workers[n]), target.send(workers[n])))



test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size,
    pin_memory = True
)


# In[5]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 34, 5, 1)
        self.conv2 = nn.Conv2d(34, 64, 5, 1)
        self.fc1 = nn.Linear(20*20*64, 512)
        self.fc2 = nn.Linear(512, 10)
        self.drop = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 20*20*64)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# model is not exactully the same as the paper since it did not mention the unit of fc


# In[6]:


def load_grad(temp, model):
    for net1,net2 in zip(model.named_parameters(),temp.named_parameters()):
        net2[1].grad = net1[1].grad.clone()


# In[7]:


def noisy_max (loss_list, p_nmax, clip_threshold):
    neg_loss_array = np.array([-x for x in loss_list])
    noise = np.random.laplace(0, clip_threshold/p_nmax, len(neg_loss_array))
    noisy_loss = neg_loss_array + noise
    best_loss_index = np.argmax(noisy_loss)
    return best_loss_index


# In[8]:


def add_grad_noise(model, noise):
    for i, param in enumerate(model.parameters()):
        param.grad.add_(noise[i])

def sub_grad_noise(model, noise):
    for i, param in enumerate(model.parameters()):
        param.grad.sub_(noise[i])


# In[9]:


def create_grad_Gaussian_noise(model, device, p_ng, clip_threshold, batch_size):
    noise = []
    # remembe that torch.normal(mean, std) use std 
    for param in model.parameters():
        noise.append(torch.normal(0, clip_threshold/math.sqrt(2 * p_ng), param.grad.size(), device=device)/batch_size)
    return noise


# In[14]:


def set_model_list(args, model, Net):
    model_list = []
    for i in range(args.n_workers):
        temp = Net().to(device)
        temp.load_state_dict(model.state_dict())
        model_list.append(temp)
    return model_list


# In[10]:


def aggregate_model(args, model_list):
    new_model_state = model_list[0].state_dict()
    #sum the weight of the model
    for m in model_list[1:]:
        state_m = m.state_dict()
        #add with new_model_state
        
        for key in state_m:
            new_model_state[key]  = new_model_state[key] + state_m[key]

    for key in new_model_state:
        new_model_state[key] /=  args.n_workers
    
    return new_model_state


# In[11]:


def best_step_size_model(args, model, device, train_loader, p_ng):
    
    r = np.random.randint(920)
    step_size_loader = train_loader[r:r+5]
    
    
    best_loss = math.inf
    best_lr = 0
    best_model = Net().to(device)
    
    
    if not args.best_lr_list:
        args.alpha_max = min(args.alpha_max, 0.1)
    elif len(args.best_lr_list) % 10 == 0:
        args.alpha_max = (1+args.gamma) * max(args.best_lr_list)
        del args.best_lr_list[:]

    #while lr_index == 0, means choose the noise add on gradient again.    

    noise = create_grad_Gaussian_noise(model, device, p_ng, args.clip_threshold, args.batch_size)
    index = 0
    args.epsilon -= p_ng
    if args.epsilon < 0:
        return model, p_ng
    
    while index == 0:
        temp_loss_list = []
        temp_model_list = []
        temp_lr_list = []
        add_grad_noise(model, noise)
        
        for i in np.linspace(0, args.alpha_max, 21):
            temp = Net().to(device)
            temp_loss = 0
            temp.load_state_dict(model.state_dict())
            #load_state_dict will not copy the grad, so you need to copy it here.
            load_grad(temp, model)
            

            temp_optimizer = optim.SGD(temp.parameters(), lr=i)
            temp_optimizer.step()
            #optimizer will be new every time, so if you have state in optimizer, it will need load state from the old optimzer.

            for (data, target) in step_size_loader:
                data,target = data.to(device), target.to(device)
                output = model(data)
                temp_loss += F.nll_loss(output, target).item()

            temp_loss_list.append(temp_loss)
            temp_model_list.append(temp)
            temp_lr_list.append(i)
        
        #choose the best lr with noisy max
        index = noisy_max(temp_loss_list, math.sqrt(2*p_nmax), args.clip_threshold)
        args.epsilon -= p_nmax
        if args.epsilon < 0:
            return model, p_ng
        
        # if index == 0, means we need to add the noise again and cost more epsilon
        if index == 0:
            #delete the original noise and add new noise
            sub_grad_noise(model, noise)
            # create new noise, and also sub the epsilon of new noise
            p_ng = (1+args.gamma) * p_ng
            noise = create_grad_Gaussian_noise(model, device, p_ng, args.clip_threshold, args.batch_size)
            args.epsilon -= (args.gamma * p_ng)
            if args.epsilon < 0:
                break
        else :
            best_model.load_state_dict(temp_model_list[index].state_dict())
            best_loss = temp_loss_list[index]
            best_lr = temp_lr_list[index]
            
    args.best_lr_list.append(best_lr)
#     print("best learning rate:", best_lr)
#     print("best loss:", best_loss)


    return best_model, p_ng


# In[12]:


def train(args, device, model, train_loader, rounds, worker_index, p_ng):
    
    unit = len(train_loader)//args.n_workers
    client_batch_number = args.client_data_number // args.batch_size
    if worker_index == 0 and rounds == 0:
        start = 0
    else:
        start = unit * worker_index + rounds * client_batch_number
    end = start + client_batch_number
    
    client_data_loader = train_loader[start:end]
    
    model.train()
    for batch_idx, (data, target) in enumerate(client_data_loader):
        data,target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_threshold)
        # Chose the best step size(learning rate)
        batch_best_model, p_ng = best_step_size_model(args, model, device, train_loader, p_ng)
        if args.epsilon < 0:
            break
        
        model.load_state_dict(batch_best_model.state_dict())
        model.zero_grad()
        #remember to zero_grad or the grad will accumlate and the model will explode
        
        if batch_idx % args.log_interval == 0:
            print('Train rounds: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\talpha_max: {:.6f}\tepsilon: {:.2f}'.format(
                rounds, batch_idx * args.batch_size, len(train_loader) * args.batch_size ,
                100. * batch_idx * args.batch_size / (len(train_loader) * args.batch_size), loss.item(), args.alpha_max, args.epsilon))
    return model, p_ng


# In[15]:


def test(args, device, model, test_loader, r_number, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / (len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    
    
    writer.add_scalar('Accuracy', accuracy,r_number)
    writer.add_scalar('Loss', test_loss, r_number)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\tepsilon: {:.2f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset)), args.epsilon))


# In[ ]:


#%%time
# model = Net().to(device)
args.best_lr_list = []

args.alpha_max = args.init_alpha_max
args.epsilon = 8
p_ng, p_nmax = args.epsilon / (2 * args.split), args.epsilon / (2 * args.split) 
# for epoch in range(1, args.epochs + 1):

start = time.time()

# for client in workers
#     while epsilon is over 0, keep training
server_model = Net().to(device)
temp_model = Net().to(device)

logdir = "/root/notebooks/tensorflow/logs/DPAGD/F_DPAGD_v2"
writer = SummaryWriter(logdir)

for r in range(args.rounds):
    temp_model_list = []
    # set model into model list
    server_model_list = set_model_list(args, server_model, Net)
    
    
    #train on all the client
    for worker_index, (worker, model) in enumerate(zip(workers, server_model_list)):
        print("Now is worker {}".format(worker_index))
        
        args.alpha_max = args.init_alpha_max
        args.epsilon = 8
        p_ng = args.epsilon / (2 * args.split)
        del args.best_lr_list[:]
        
        while args.epsilon > 0:
            worker_best_model, p_ng = train(args, device, model, train_loader , r, worker_index, p_ng)
        #append model trained by client into list
        temp_model.load_state_dict(worker_best_model.state_dict())
        temp_model_list.append(temp_model)
        
    temp_stat_dict = aggregate_model(args, temp_model_list)
    server_model.load_state_dict(temp_stat_dict)
    test(args, device, server_model, test_loader, r, writer)
    

print("Spend time:{:.1f}".format(time.time() - start))

if (args.save_model):
    torch.save(model.state_dict(), "mnist_cnn.pt")


# In[ ]:




