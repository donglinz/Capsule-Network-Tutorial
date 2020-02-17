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
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import torchvision.models as models
import os
import args_parser
# TODO pass parameters into each network
# TODO reconstruction layer
args = args_parser.get_args()

summary_prefix = '{0}_{1}{2} '.format(args.action, args.dataset, args.profile_category)
summary_dir = 'runs/{0}/{1}'.format(args.dataset, summary_prefix + datetime.now().strftime("%b %d %Y %H:%M:%S"))
global_step_train = 0

def summary_variables(writer, tag, variable):
    writer.add_scalar(tag+'/max', variable.max(), global_step_train)
    writer.add_scalar(tag+'/min', variable.min(), global_step_train)
    writer.add_scalar(tag+'/mean', variable.mean(), global_step_train)
    writer.add_histogram(tag, variable, global_step_train)

class DataLoader:
    def __init__(self):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
        
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=dataset_transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=dataset_transform)
        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=dataset_transform)
            test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=dataset_transform)
        elif args.dataset == 'f-mnist':
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=dataset_transform)
            test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=dataset_transform)
        elif args.dataset == 'svhn':
            train_dataset = datasets.SVHN('./data', split='train', download=True, transform=dataset_transform)
            test_dataset = datasets.SVHN('./data', split='test', download=True, transform=dataset_transform)
        
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)        

class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )
                        

    def forward(self, x):
        summary_variables(self.writer, 'conv1/weight', self.conv.weight)
        summary_variables(self.writer, 'conv1/bias', self.conv.bias)

        result = F.relu(self.conv(x))
        if args.profile_eval_by_category:
            images = result.transpose(0, 1)
            self.writer.add_images('conv1/channels', images, global_step_train)
        return result
    def add_writer(self, writer):
        self.writer = writer

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=64, in_channels=256, out_channels=8, kernel_size=9):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2) 
                          for _ in range(num_capsules)])
    
    def forward(self, x):
        summary_variables(self.writer, 'conv2/weight', torch.stack([capsule.weight for capsule in self.capsules]))
        summary_variables(self.writer, 'conv2/bias', torch.stack([capsule.bias for capsule in self.capsules]))

        u = [capsule(x) for capsule in self.capsules]
        width = u[0].shape[-1]
        u = torch.stack(u, dim=4)

        if args.profile_eval_by_category:
            self.writer.add_images('conv2/output_avg', self.get_images_avg(u), global_step_train)
            self.writer.add_images('conv2/output', self.get_images(u), global_step_train)
        
        u = u.view(x.size(0), -1, self.num_capsules * width * width)
        u = torch.transpose(u, 1, 2)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    def add_writer(self, writer):
        self.writer = writer

    def get_images_avg(self, u):
        return u.mean(dim=1).permute(3, 0, 1, 2)
    
    def get_images(self, u):
        if args.dataset == 'svhn':
            return u.permute(0,2,3,4,1).reshape(1,8,8,-1).permute(3,0,1,2)
        else:
            return u.permute(0,2,3,4,1).reshape(1,6,6,-1).permute(3,0,1,2)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=64 * 8 * 8, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.out_channels = out_channels

        self.W = nn.Parameter(torch.empty(1, num_routes, num_capsules, out_channels, in_channels).normal_(mean=0, std=0.01))
        self.bias = nn.Parameter(torch.empty(num_capsules, out_channels).normal_(mean=0, std=0.01))

    def forward(self, x):
        summary_variables(self.writer, 'dight_caps/W', self.W)
        summary_variables(self.writer, 'dight_caps/bias', self.bias)

        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(3)

        W = torch.cat([self.W] * batch_size, dim=0)
        
        u_hat = (W * x).sum(dim=-1, keepdim=True)

        b_ij = Variable(torch.zeros(batch_size, self.num_routes, self.num_capsules, 1))
        if args.use_cuda:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            if args.leaky_routing:
                c_ij = self.leaky_softmax(b_ij).unsqueeze(4)
            else:
                c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            s_j = s_j + torch.stack([self.bias] * batch_size, dim=0).unsqueeze(1).unsqueeze(4)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4)
        
        summary_variables(self.writer, 'dight_caps/b_ij', b_ij)
        summary_variables(self.writer, 'dight_caps/c_ij', c_ij)

        if args.profile_eval_by_category:
            if args.dataset == 'svhn':
                self.writer.add_images('dight_caps/c_ij_distribution', c_ij.squeeze(3).squeeze(3).view(1, 128, -1).unsqueeze(0), global_step_train)
            else:
                self.writer.add_images('dight_caps/c_ij_distribution', c_ij.squeeze(3).squeeze(3).view(1, 96, -1).unsqueeze(0), global_step_train)
            #self.writer.add_images('dight_caps/c_ij_distribution_raw', c_ij.squeeze(3).squeeze(3).unsqueeze(0), global_step_train)
            self.plot_cij_by_capsule(c_ij)
            self.plot_cij_all(c_ij)
            
        return v_j.squeeze(1)
    
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
    
    def plot_cij_by_capsule(self, c_ij):
        if args.dataset == 'svhn':
            routes = c_ij.reshape(8,8,32,10).mean(dim=0).mean(dim=0)[:,args.profile_category]
        else:
            routes = c_ij.reshape(6,6,32,10).mean(dim=0).mean(dim=0)[:,args.profile_category]
        for i in range(32):
            self.writer.add_scalar('digit_caps/c_capsule{0}'.format(i), routes[i], global_step_train)

    def plot_cij_all(self, c_ij):
        if args.dataset == 'svhn':
            routes = c_ij.reshape(2048, 10)[:,args.profile_category]
        else:
            routes = c_ij.reshape(1152, 10)[:,args.profile_category]

        for i in range(2048):
            self.writer.add_scalar('digit_caps/c_ij{0}'.format(i), routes[i], global_step_train)
    def add_writer(self, writer):
        self.writer = writer

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
        classes = F.softmax(classes, dim=1)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if args.use_cuda:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
        return masked

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(in_channels = 1 if args.dataset == 'mnist' or args.dataset == 'f-mnist' else 3)
        self.primary_capsules = PrimaryCaps(num_capsules = args.num_capsules)
        if args.dataset == 'mnist' or args.dataset == 'f-mnist':
            self.digit_capsules = DigitCaps(num_routes = 32 * 6 * 6)
        elif args.dataset == 'svhn':
            self.digit_capsules = DigitCaps(num_routes = 8 * 8 * 32)
        elif args.dataset == 'cifar10':
            self.digit_capsules = DigitCaps(num_routes = 64 * 8 * 8)
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
    def add_image(self, *args):
        self.writer.add_image(*args)
    def add_images(self, *args):
        self.writer.add_images(*args)
    def close(self):
        self.writer.close()
    def flush(self):
        self.writer.flush()

capsule_net = CapsNet()
if args.use_cuda:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters(), weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.50)

def load_model_ckpt(saved_model_path):
    if args.use_cuda:
        checkpoint = torch.load(saved_model_path)
    else:
        checkpoint = torch.load(saved_model_path, map_location=torch.device('cpu'))
    return checkpoint
    # capsule_net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_step(model, optimizer, scheduler, data, target):
    optimizer.zero_grad()
    output, masked = capsule_net(data)
    loss = capsule_net.loss(data, output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_loss = loss.data.item()
    train_accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / data.shape[0]
    writer.add_scalar('train/accuracy', train_accuracy, global_step_train)
    writer.add_scalar('train/loss', train_loss, global_step_train)
    return train_loss, train_accuracy

def eval_step(model, writer, data, target):
    output, masked = model(data)
    loss = model.loss(data, output, target)

    loss = loss.data.item()
    correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
    total = data.shape[0]

    return loss, correct, total

dataset = DataLoader()
#load_model('/home/donglin/Capsule-Network-Tutorial/runs/test_bias Jan 05 2020 04:50:29/saved_model')
writer = Writer(SummaryWriter(summary_dir))
capsule_net.add_writer(writer)

def train_experiment():
    global global_step_train
    for epoch in range(args.epoch):
        capsule_net.train()
        total_train_loss = 0
        print('--------------- epoch ', epoch, '---------------')
        for batch_id, (data, target) in enumerate(dataset.train_loader):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            global_step_train += 1
                
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            
            loss, train_accuracy = train_step(capsule_net, optimizer, scheduler, data, target)
            total_train_loss += loss
            
            if batch_id % args.save_step == 0:
                torch.save({
                    'model_state_dict': capsule_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(summary_dir, 'saved_model'))
            
        print ("train loss: ", total_train_loss / len(dataset.train_loader))
        
        capsule_net.eval()
        test_loss = 0
        correct_predictions = 0.0
        total_predictions = 0.0
        for batch_id, (data, target) in enumerate(dataset.test_loader):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)

            if args.use_cuda:
                data, target = data.cuda(), target.cuda()

            loss, correct, total = eval_step(capsule_net, writer, data, target)

            test_loss += loss
            correct_predictions += correct
            total_predictions += total
        
        print ('test accuracy: ', correct_predictions / total_predictions, 'correct: ', correct_predictions, 'total: ', total_predictions)
        print ("test loss: ", test_loss / len(dataset.test_loader))
        
        writer.add_scalar_force('test/accuracy', correct_predictions / total_predictions, global_step_train)
        writer.add_scalar_force('test/loss', test_loss / len(dataset.test_loader), global_step_train)

def eval_experiment():
    global global_step_train
    ckpt = load_model_ckpt(args.saved_model)
    capsule_net.load_state_dict(ckpt['model_state_dict'])

    capsule_net.eval()
    test_loss = 0
    correct_predictions = 0.0
    total_predictions = 0.0
    for batch_id, (data, target) in enumerate(dataset.test_loader):
        if args.profile_eval_by_category and target.item() != args.profile_category:
            continue

        global_step_train += 1

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if args.use_cuda:
            data, target = data.cuda(), target.cuda()

        loss, correct, total = eval_step(capsule_net, writer, data, target)

        test_loss += loss
        correct_predictions += correct
        total_predictions += total

    print ('test accuracy: ', correct_predictions / total_predictions, 'correct: ', correct_predictions, 'total: ', total_predictions)
    print ("test loss: ", test_loss / len(dataset.test_loader))
    
    writer.add_scalar_force('test/accuracy', correct_predictions / total_predictions, global_step_train)
    writer.add_scalar_force('test/loss', test_loss / len(dataset.test_loader), global_step_train)

if args.action == 'train':
    train_experiment()

if args.action == 'eval':
    eval_experiment()

writer.close()

