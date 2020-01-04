# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.models as models
import os

USE_CUDA = True
global_step_train = 0
n_epochs = 30
batch_size = 64
summary_prefix = 'test_leaky_with_bias'
summary_dir = 'runs/{0}'.format(summary_prefix + datetime.now().strftime("%b %d %Y %H:%M:%S"))
use_leaky_routing = True

# %%
class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=dataset_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=dataset_transform)
        
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)        


# %%
class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        self.writer.add_histogram('conv1/weight', self.conv.weight, global_step_train)
        self.writer.add_histogram('conv1/bias', self.conv.bias, global_step_train)

        return F.relu(self.conv(x))
    def add_writer(self, writer):
        self.writer = writer


# %%
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=64, in_channels=256, out_channels=8, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=4) 
                          for _ in range(num_capsules)])
    
    def forward(self, x):
        self.writer.add_histogram('conv2/weight', torch.stack([capsule.weight for capsule in self.capsules]), global_step_train)
        self.writer.add_histogram('conv2/bias', torch.stack([capsule.bias for capsule in self.capsules]), global_step_train)

        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=4)
        u = u.view(x.size(0), -1, 64 * 12 * 12)
        u = torch.transpose(u, 1, 2)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    def add_writer(self, writer):
        self.writer = writer


# %%
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=64 * 12 * 12, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        self.bias = nn.Parameter(torch.randn(num_capsules, out_channels))

    def forward(self, x):
        self.writer.add_histogram('dight_caps/W', self.W, global_step_train)
        self.writer.add_histogram('dight_caps/bias', self.bias, global_step_train)

        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(batch_size, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            if use_leaky_routing:
                c_ij = self.leaky_softmax(b_ij).unsqueeze(4)
            else:
                c_ij = F.softmax(b_ij).unsqueeze(4)
            #c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4)
        
        self.writer.add_histogram('dight_caps/b_ij', b_ij, global_step_train)
        
        return v_j.squeeze(1) + torch.stack([self.bias] * batch_size, dim=0).unsqueeze(3)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-2, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    def leaky_softmax(self, logits):
        leak = torch.zeros_like(logits)
        leak = torch.sum(leak, dim=2, keepdim=True)
        leaky_logits = torch.cat([leak, logits], axis=2)
        leaky_routing = F.softmax(leaky_logits, dim=2)
        return leaky_routing[:,:,1:,:]

    def add_writer(self, writer):
        self.writer = writer


# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
        return masked


# %%
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        masked = self.decoder(output, data)
        return output, masked
    
    def loss(self, data, x, target):
        return self.margin_loss(x, target)
    
    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = ((F.relu(0.9 - v_c))**2).view(batch_size, -1)
        right = ((F.relu(v_c - 0.1))**2).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def add_writer(self, writer):
        self.digit_capsules.add_writer(writer)
        self.conv_layer.add_writer(writer)
        self.primary_capsules.add_writer(writer)


# %%
class Writer():
    def __init__(self, writer):
        self.writer = writer
    def add_scalar(self, *args):
        if global_step_train % 10 == 1:
            self.writer.add_scalar(*args)
    def add_scalar_force(self, *args):
        self.writer.add_scalar(*args)
    def add_histogram(self, *args):
        if global_step_train % 10 == 1:
            self.writer.add_histogram(*args)
    def close(self):
        self.writer.close()
    def flush(self):
        self.writer.flush()


# %%
capsule_net = CapsNet()
if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters())

def load_model(saved_model_path):
    checkpoint = torch.load(saved_model_path)
    capsule_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    global global_step_train
    global summary_dir
    global_step_train = checkpoint['global_step_train']
    summary_dir = checkpoint['summary_dir']

# %%
mnist = Mnist(batch_size)
writer = Writer(SummaryWriter(summary_dir))
capsule_net.add_writer(writer)

for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    print('--------------- epoch ', epoch, '---------------')
    for batch_id, (data, target) in enumerate(mnist.train_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        global_step_train += 1
            
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data.item()
        
        train_accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / data.shape[0]

        writer.add_scalar('train/accuracy', train_accuracy, global_step_train)
        writer.add_scalar('train/loss', loss.data.item(), global_step_train)

        if batch_id % 100 == 0:
            print ("train accuracy:", train_accuracy)
            torch.save({
                'model_state_dict': capsule_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step_train': global_step_train,
                'summary_dir': summary_dir
            }, os.path.join(summary_dir, 'saved_model'))
        
    print ("train loss: ", train_loss / len(mnist.train_loader))
    
    capsule_net.eval()
    test_loss = 0
    correct_predictions = 0.0
    total_predictions = 0.0
    for batch_id, (data, target) in enumerate(mnist.test_loader):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target)

        test_loss += loss.data.item()
        correct_predictions += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        total_predictions += data.shape[0]
    
    print ('test accuracy: ', correct_predictions / total_predictions, 'correct: ', correct_predictions, 'total: ', total_predictions)
    print ("test loss: ", test_loss / len(mnist.test_loader))
    
    writer.add_scalar_force('test/accuracy', correct_predictions / total_predictions, global_step_train)
    writer.add_scalar_force('test/loss', test_loss / len(mnist.test_loader), global_step_train)

writer.close()

