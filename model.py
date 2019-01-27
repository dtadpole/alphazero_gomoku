#!/usr/bin/env python

import io
import os
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
#import onnx_tensorrt.backend as backend
import math
import random
import timeit
from args import parse_args
from game import Game


class Model(nn.Module):

    def __init__(self, state_shape, action_size, args):

        super(Model, self).__init__()
        
        self.CHANNELS    = args.model_channels
        self.CHANNELS_P  = int(args.model_channels/2)
        self.CHANNELS_V  = int(args.model_channels/4)
        
        self._state_layer, self._size_x, self._size_y  = state_shape
        self._action_size                              = action_size
        self._args                                     = args

        if args.model_cuda >= 0 and args.model_tensorrt != 2:
            self._device = torch.device("cuda:" + str(args.model_cuda))
            self.cuda(self._device)

        self.lr_base       = args.model_lr_base
        self.lr_multiplier = 1.0

        self.kl_target     = args.model_kl_target

        if args.model_tensorrt != 2:
            self.build_network(args)
    
            if args.model_tensorrt:
                self.build_trt_engine(self.state_dict(), dtype=np.dtype(self._args.play_dtype))


    def build_network(self, args):
    
        if args.model_arch == 'resnet':
            
            # resnet
            self.resnet_input_conv = nn.Conv2d(self._state_layer, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_input_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_01a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_01a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_01b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_01b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_02a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_02a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_02b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_02b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_03a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_03a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_03b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_03b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_04a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_04a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_04b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_04b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_05a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_05a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_05b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_05b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_06a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_06a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_06b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_06b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_07a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_07a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_07b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_07b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_08a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_08a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_08b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_08b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_09a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_09a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_09b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_09b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_10a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_10a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_10b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_10b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_11a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_11a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_11b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_11b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_12a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_12a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_12b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_12b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_13a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_13a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_13b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_13b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_14a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_14a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_14b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_14b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_15a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_15a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_15b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_15b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_16a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_16a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_16b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_16b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_17a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_17a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_17b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_17b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_18a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_18a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_18b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_18b_bn   = nn.BatchNorm2d(self.CHANNELS)

            self.resnet_19a_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_19a_bn   = nn.BatchNorm2d(self.CHANNELS)
            self.resnet_19b_conv = nn.Conv2d(self.CHANNELS, self.CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet_19b_bn   = nn.BatchNorm2d(self.CHANNELS)

            # policy head
            self.pc     = nn.Conv2d(self.CHANNELS, self.CHANNELS_P, kernel_size=1, stride=1, bias=False)
            self.pc_bn  = nn.BatchNorm2d(self.CHANNELS_P)

            self.p1     = nn.Linear(self.CHANNELS_P*(self._size_x)*(self._size_y), self._action_size)

            # value head
            self.vc     = nn.Conv2d(self.CHANNELS, self.CHANNELS_V, kernel_size=1, stride=1, bias=False)
            self.vc_bn  = nn.BatchNorm2d(self.CHANNELS_V)

            self.v1     = nn.Linear(self.CHANNELS_V*(self._size_x)*(self._size_y), 256)
            self.v2     = nn.Linear(256, 1)

        elif args.model_arch == 'conv4':

            self.conv1  = nn.Conv2d(self._state_layer, self.CHANNELS, kernel_size=3, stride=1, padding=1)
            self.conv2  = nn.Conv2d(self.CHANNELS, self.CHANNELS*2, kernel_size=3, stride=1, padding=1)
            self.conv3  = nn.Conv2d(self.CHANNELS*2, self.CHANNELS*4, kernel_size=3, stride=1, padding=1)
            self.conv4  = nn.Conv2d(self.CHANNELS*4, self.CHANNELS*4, kernel_size=3, stride=1, padding=1)

            # policy head
            self.pc     = nn.Conv2d(self.CHANNELS*4, self.CHANNELS_P, kernel_size=1, stride=1)
            self.p1     = nn.Linear(self.CHANNELS_P*(self._size_x)*(self._size_y), self._action_size)

            # value head
            self.vc     = nn.Conv2d(self.CHANNELS*4, self.CHANNELS_V, kernel_size=1, stride=1)
            self.v1     = nn.Linear(self.CHANNELS_V*(self._size_x)*(self._size_y), 64)
            self.v2     = nn.Linear(64, 1)
            
        elif args.model_arch == 'conv3':

            self.conv1  = nn.Conv2d(self._state_layer, self.CHANNELS, kernel_size=3, stride=1, padding=1)
            self.conv2  = nn.Conv2d(self.CHANNELS, self.CHANNELS*2, kernel_size=3, stride=1, padding=1)
            self.conv3  = nn.Conv2d(self.CHANNELS*2, self.CHANNELS*4, kernel_size=3, stride=1, padding=1)
            #self.conv4  = nn.Conv2d(self.CHANNELS*4, self.CHANNELS*4, kernel_size=3, stride=1, padding=1)

            # policy head
            self.pc     = nn.Conv2d(self.CHANNELS*4, self.CHANNELS_P, kernel_size=1, stride=1)
            self.p1     = nn.Linear(self.CHANNELS_P*(self._size_x)*(self._size_y), self._action_size)

            # value head
            self.vc     = nn.Conv2d(self.CHANNELS*4, self.CHANNELS_V, kernel_size=1, stride=1)
            self.v1     = nn.Linear(self.CHANNELS_V*(self._size_x)*(self._size_y), 64)
            self.v2     = nn.Linear(64, 1)
            
        else:
            
            raise Exception("unknown model arch : %s" % args.model_arch)


        if self._args.model_cuda >= 0 and self._args.model_tensorrt != 2:
            self.to(self._device)

        if args.model_tensorrt != 2:
            self.optimizer = optim.Adam(self.parameters(),
                                        lr = self.lr_base * self.lr_multiplier,
                                        weight_decay=args.model_weight_decay)

            if self._args.model_precision == 'half':
                self.model_half()
            elif self._args.model_precision == 'double':
                self.model_double()
            else:
                self.model_float()
            

            
    def forward(self, s):
        
        #intermediate = s
        
        if self._args.model_arch == 'resnet':

            s = self.resnet_input_conv(s)
            
            #intermediate = s

            s = F.relu(self.resnet_input_bn(s))

            orig = s
            s = F.relu(self.resnet_01a_bn(self.resnet_01a_conv(s)))
            s = F.relu(self.resnet_01b_bn(self.resnet_01b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_02a_bn(self.resnet_02a_conv(s)))
            s = F.relu(self.resnet_02b_bn(self.resnet_02b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_03a_bn(self.resnet_03a_conv(s)))
            s = F.relu(self.resnet_03b_bn(self.resnet_03b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_04a_bn(self.resnet_04a_conv(s)))
            s = F.relu(self.resnet_04b_bn(self.resnet_04b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_05a_bn(self.resnet_05a_conv(s)))
            s = F.relu(self.resnet_05b_bn(self.resnet_05b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_06a_bn(self.resnet_06a_conv(s)))
            s = F.relu(self.resnet_06b_bn(self.resnet_06b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_07a_bn(self.resnet_07a_conv(s)))
            s = F.relu(self.resnet_07b_bn(self.resnet_07b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_08a_bn(self.resnet_08a_conv(s)))
            s = F.relu(self.resnet_08b_bn(self.resnet_08b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_09a_bn(self.resnet_09a_conv(s)))
            s = F.relu(self.resnet_09b_bn(self.resnet_09b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_10a_bn(self.resnet_10a_conv(s)))
            s = F.relu(self.resnet_10b_bn(self.resnet_10b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_11a_bn(self.resnet_11a_conv(s)))
            s = F.relu(self.resnet_11b_bn(self.resnet_11b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_12a_bn(self.resnet_12a_conv(s)))
            s = F.relu(self.resnet_12b_bn(self.resnet_12b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_13a_bn(self.resnet_13a_conv(s)))
            s = F.relu(self.resnet_13b_bn(self.resnet_13b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_14a_bn(self.resnet_14a_conv(s)))
            s = F.relu(self.resnet_14b_bn(self.resnet_14b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_15a_bn(self.resnet_15a_conv(s)))
            s = F.relu(self.resnet_15b_bn(self.resnet_15b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_16a_bn(self.resnet_16a_conv(s)))
            s = F.relu(self.resnet_16b_bn(self.resnet_16b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_17a_bn(self.resnet_17a_conv(s)))
            s = F.relu(self.resnet_17b_bn(self.resnet_17b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_18a_bn(self.resnet_18a_conv(s)))
            s = F.relu(self.resnet_18b_bn(self.resnet_18b_conv(s)) + orig)

            orig = s
            s = F.relu(self.resnet_19a_bn(self.resnet_19a_conv(s)))
            s = F.relu(self.resnet_19b_bn(self.resnet_19b_conv(s)) + orig)

            # s = s.view(-1, self.CHANNELS*(self._size_x)*(self._size_y))

            # policy head
            policy = F.relu(self.pc_bn(self.pc(s)))                                  # policy conv2d 1x1
            policy = policy.view(-1, self.CHANNELS_P*(self._size_x)*(self._size_y))  # flattern
            policy = self.p1(policy)                                                 # dense
            policy = F.log_softmax(policy, dim=1)                                    # softmax

            # value head
            value  = F.relu(self.vc_bn(self.vc(s)))                                  # value conv2d 1x1
            value  = value.view(-1, self.CHANNELS_V*(self._size_x)*(self._size_y))   # flattern
            value  = F.relu(self.v1(value))                                          # dense
            value  = self.v2(value)                                                  # dense
            value  = torch.tanh(value)                                               # tanh

        elif self._args.model_arch == 'conv4':

            s = s.view(-1, self._state_layer, self._size_x, self._size_y)             # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv1(s))                                                # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv2(s))                                                # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv3(s))                                                # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv4(s))                                                # batch_size x num_channels x s_x x s_y

            # policy head
            policy = F.relu(self.pc(s))                                              # policy conv2d 1x1
            policy = policy.view(-1, self.CHANNELS_P*(self._size_x)*(self._size_y))  # flattern
            policy = self.p1(policy)                                                 # dense
            policy = F.log_softmax(policy, dim=1)                                    # softmax

            # value head
            value  = F.relu(self.vc(s))                                              # value conv2d 1x1
            value  = value.view(-1, self.CHANNELS_V*(self._size_x)*(self._size_y))   # flattern
            value  = F.relu(self.v1(value))                                          # dense
            value  = self.v2(value)                                                  # dense
            value  = torch.tanh(value)                                               # tanh

        elif self._args.model_arch == 'conv3':
            
            s = s.view(-1, self._state_layer, self._size_x, self._size_y)             # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv1(s))                                                # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv2(s))                                                # batch_size x num_channels x s_x x s_y
            s = F.relu(self.conv3(s))                                                # batch_size x num_channels x s_x x s_y

            # policy head
            policy = F.relu(self.pc(s))                                              # policy conv2d 1x1
            policy = policy.view(-1, self.CHANNELS_P*(self._size_x)*(self._size_y))  # flattern
            policy = self.p1(policy)                                                 # dense
            policy = F.log_softmax(policy, dim=1)                                    # softmax

            # value head
            value  = F.relu(self.vc(s))                                              # value conv2d 1x1
            value  = value.view(-1, self.CHANNELS_V*(self._size_x)*(self._size_y))   # flattern
            value  = F.relu(self.v1(value))                                          # dense
            value  = self.v2(value)                                                  # dense
            value  = torch.tanh(value)                                               # tanh

        else:
            
            raise Exception("unknown model arch : %s" % args.model_arch)
            
        return policy, value
        #return policy, value, intermediate
    

    
    def model_half(self):
        self._dtype = np.float16
        self.half()  # convert to half precision
        for layer in self.modules():
          #if isinstance(layer, nn.BatchNorm2d):
            layer.half()
    
    def model_float(self):
        self._dtype = np.float32
        self.float()  # convert to float precision
        for layer in self.modules():
          #if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    
    def model_double(self):
        self._dtype = np.float64
        self.double()  # convert to double precision
        for layer in self.modules():
          #if isinstance(layer, nn.BatchNorm2d):
            layer.double()


    def model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_learning_rate(self, lr):
        """Sets the learning rate to the given value"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def game_train(self, examples):

        boards, predvs, pis, advs, nlogp, vs = list(zip(*[examples[i] for i in range(len(examples))]))
        if self._dtype == np.float16:
            boards          = torch.HalfTensor(np.array(boards))
            pred_vs         = torch.HalfTensor(np.array(predvs))
            target_pis      = torch.HalfTensor(np.array(pis))
            target_vs       = torch.HalfTensor(np.array(vs))
            #old_advs        = torch.HalfTensor(np.array(advs))
            #old_nlogp       = torch.HalfTensor(np.array(nlogp))
        elif self._dtype == np.float64:
            boards          = torch.DoubleTensor(np.array(boards))
            pred_vs         = torch.DoubleTensor(np.array(predvs))
            target_pis      = torch.DoubleTensor(np.array(pis))
            target_vs       = torch.DoubleTensor(np.array(vs))
            #old_advs        = torch.DoubleTensor(np.array(advs))
            #old_nlogp       = torch.DoubleTensor(np.array(nlogp))
        else:
            boards          = torch.FloatTensor(np.array(boards))
            pred_vs         = torch.FloatTensor(np.array(predvs))
            target_pis      = torch.FloatTensor(np.array(pis))
            target_vs       = torch.FloatTensor(np.array(vs))
            #old_advs        = torch.FloatTensor(np.array(advs))
            #old_nlogp       = torch.FloatTensor(np.array(nlogp))

        # predict
        if self._args.model_cuda >= 0 and self._args.model_tensorrt != 2:
            boards, pred_vs, target_pis, target_vs = boards.contiguous().cuda(device=self._device), \
                                                     pred_vs.contiguous().cuda(device=self._device), \
                                                     target_pis.contiguous().cuda(device=self._device), \
                                                     target_vs.contiguous().cuda(device=self._device)
            #boards, pred_vs, target_pis, target_vs, old_advs, old_nlogp = boards.contiguous().cuda(device=self._device), \
            #                                                              pred_vs.contiguous().cuda(device=self._device), \
            #                                                              target_pis.contiguous().cuda(device=self._device), \
            #                                                              target_vs.contiguous().cuda(device=self._device), \
            #                                                              old_advs.contiguous().cuda(device=self._device), \
            #                                                              old_nlogp.contiguous().cuda(device=self._device)

        prev_policy, prev_value = self(boards)
        #entropy_prev = -torch.mean(torch.sum(prev_policy * torch.exp(prev_policy), 1))
        
        entropy_list = [-np.sum(list(filter(lambda v:v>0, k)) * np.log(list(filter(lambda v:v>0, k)))) for k in pis]
        entropy_mcts = np.mean(entropy_list)

        for epoch in range(self._args.model_epochs):

            # compute output
            curr_policy, curr_value = self(boards)

            entropy_curr  = -torch.mean(torch.sum(curr_policy * torch.exp(curr_policy), 1))

            if self._args.model_ppo2:

                #print("old_nlogp", old_nlogp.shape)
                #print("old_advs", old_advs.shape)
                policy_nlogp = -target_pis*curr_policy
                ratio = torch.exp(policy_nlogp - old_nlogp)
                #print("policy_nlogp", policy_nlogp.shape)
                #print("ratio", ratio.shape)
                #print(policy_nlogp)
                #print(old_nlogp)
                #print(old_advs)
                
                pi_loss = self.loss_pi(target_pis, curr_policy)

                pg_loss1 = old_advs * ratio
                pg_loss2 = old_advs * torch.clamp(ratio, 1.0 - self._args.model_ppo2_clip, 1.0 + self._args.model_ppo2_clip)
                pg_loss  = torch.mean(torch.sum(torch.max(pg_loss1, pg_loss2), -1))
                #print("pg_loss1", pg_loss1.shape)
                #print("pg_loss2", pg_loss2.shape)
                #print("pg_loss", pg_loss.shape)

                #print("pred_vs", pred_vs.shape)
                #print("curr_value", curr_value.shape)

                vpred_clipped = pred_vs + torch.clamp(torch.squeeze(curr_value) - pred_vs,
                                                      -self._args.model_ppo2_clip,
                                                      self._args.model_ppo2_clip)
                #print("vpred_clipped", vpred_clipped.shape)
                
                vf_loss1 = (torch.squeeze(curr_value) - target_vs) ** 2
                vf_loss2 = (torch.squeeze(vpred_clipped) - target_vs) ** 2
                #print("vf_loss1", vf_loss1.shape)
                #print("vf_loss2", vf_loss2.shape)

                vf_loss = .5 * torch.mean(torch.max(vf_loss1, vf_loss2))
                #print("vf_loss", vf_loss.shape)
                
                total_loss = pg_loss + vf_loss * self._args.model_coef_value - entropy_curr * self._args.model_coef_entropy

                #print("pg_loss %.3f, vf_loss %.3f, entropy %.3f" % (pg_loss.detach().cpu().numpy(),
                #                                                    vf_loss.detach().cpu().numpy(),
                #                                                    entropy_curr.detach().cpu().numpy()))
                
            else:
                pi_loss = self.loss_pi(target_pis, curr_policy)
                pg_loss = 0
                vf_loss = self.loss_v(target_vs, curr_value)
                total_loss = pi_loss + vf_loss * self._args.model_coef_value - entropy_curr * self._args.model_coef_entropy

            self.optimizer.zero_grad()

            # compute gradient and do SGD step
            curr_lr = self.lr_base * self.lr_multiplier
            self.set_learning_rate(curr_lr)
            #self.optimizer = optim.Adam(self.parameters(), lr = self.lr_base * self.lr_multiplier, weight_decay=1e-4)

            # optimize
            total_loss.backward()
            self.optimizer.step()

            new_policy, new_value = self(boards)

            # calculate policy entropy
            #entropy_curr = -torch.mean(torch.sum(curr_policy * torch.exp(curr_policy), 1))
            entropy_new = -torch.mean(torch.sum(new_policy * torch.exp(new_policy), 1))
            
            #print(prev_policy)
            #print(new_policy)
            kl_list = np.sum(np.exp(prev_policy.data.cpu().numpy()) * (prev_policy.data.cpu().numpy() - new_policy.data.cpu().numpy()), axis=1)
            #print(kl_list)
            kl = np.mean(kl_list)

            print("EPOCH %d - Loss: %.3f   [Pi : %.3f, %.3f, %.3f]   [V : %.3f]   [Entropy: %.3f -> %.3f]   [LR: %.5f, KL: %.5f]" % (epoch+1, total_loss, pi_loss, pg_loss, entropy_mcts, vf_loss, entropy_curr, entropy_new, curr_lr, kl))

            if kl > self.kl_target * 4:  # early stopping if D_KL diverges badly
                print("KL Out of Bound: %.5f" % kl)
                break

        # adaptively adjust the learning rate
        if kl > self.kl_target * 2 and self.lr_multiplier > 1 / self._args.model_lr_range:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_target / 2 and self.lr_multiplier < self._args.model_lr_range:
            self.lr_multiplier *= 1.5

        print("END EPOCH [LR_MULTIPLIER : %.5f, KL : %.5f]" % (self.lr_multiplier, kl))


    def game_predict(self, board):

        # preparing input
        if self._dtype == np.float16:
            boards = torch.HalfTensor(board)
        elif self._dtype == np.float64:
            boards = torch.DoubleTensor(board)
        else:
            boards = torch.FloatTensor(board)
            
        if self._args.model_cuda >= 0 and self._args.model_tensorrt != 2:
            boards = boards.contiguous().cuda(device=self._device)
        boards = boards.view(-1, self._state_layer, self._size_x, self._size_y)
        #self.eval()
        with torch.no_grad():
            #policy, value, intermediate = self(boards)
            policy, value = self(boards)

        #print("policy :", policy)
        #print("value  :", value)
        #return np.exp(policy.data.cpu().numpy()[0]), value.data.cpu().numpy()[0], intermediate
        return np.exp(policy.data.cpu().numpy()[0]), value.data.cpu().numpy()[0]


    def loss_pi(self, targets, outputs):
        #print("pi targets : %s" % targets)
        #print("pi outputs : %s" % outputs)
        policy_loss = -torch.sum(targets*outputs, dim=1)
        #print("pi loss : %s" % policy_loss)
        return torch.mean(policy_loss)


    def loss_v(self, targets, outputs):
        #print("v targets : %s" % targets)
        #print("v outputs : %s" % outputs)
        loss = F.mse_loss(targets, outputs.view(-1))
        #print(loss)
        return loss

    def to_base64(self):
        buffer = io.BytesIO()
        torch.save({
            'state_dict' : self.state_dict(),
            #'lr_multiplier' : self.lr_multiplier
        }, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def from_base64(self, base64_model):
        
        buffer = io.BytesIO(base64.b64decode(base64_model))
        map_location = None if (self._args.model_cuda >= 0 and self._args.model_tensorrt != 2) else 'cpu'
        checkpoint = torch.load(buffer, map_location=map_location)
        
        model = Model((self._state_layer, self._size_x, self._size_y), self._action_size, self._args)
        if self._args.model_tensorrt != 2:
            model.load_state_dict(checkpoint['state_dict'])
            #model.lr_multiplier = checkpoint['lr_multiplier']
            return model
        else:
            model.build_trt_engine(checkpoint['state_dict'], dtype=np.dtype(self._args.play_dtype))
            return model
            
    
    def save_model(self, filepath='model/model.data'):
        # rename file if exist
        if os.path.isfile(filepath+'.bak'):
            os.rename(filepath+'.bak', filepath+'.bak2')
        if os.path.isfile(filepath):
            os.rename(filepath, filepath+'.bak')
        # save current model to filepath
        torch.save({
            'state_dict' : self.state_dict(),
            'lr_multiplier' : self.lr_multiplier
        }, filepath)

    def load_model(self, filepath='model/model.data'):
        if os.path.isfile(filepath):
            try:
                map_location = None if self._args.model_cuda >= 0 and self._args.model_tensorrt != 2 else 'cpu'
                checkpoint = torch.load(filepath, map_location=map_location)
                model = Model((self._state_layer, self._size_x, self._size_y), self._action_size, self._args)
                model.load_state_dict(checkpoint['state_dict'])
                model.lr_multiplier = checkpoint['lr_multiplier'] if 'lr_multiplier' in checkpoint else 1.0
                return model
            except Exception as e:
                return self.load_model(filepath + '.bak')
        else:
            return None

        
    def game_predict_trt(self, board):
        
        h_game_input    = board.astype(np.dtype(self._args.play_dtype))
        h_policy_output = np.empty(self._action_size, dtype = np.dtype(self._args.play_dtype))
        h_value_output  = np.empty(1, dtype = np.dtype(self._args.play_dtype))
        #h_intermediate_output = np.empty(self.CHANNELS * self._size_x * self._size_y, dtype = np.dtype(self._args.play_dtype))

        #print(h_game_input.nbytes, h_policy_output.nbytes, h_value_output.nbytes)
        
        #d_game_input    = cuda.mem_alloc(h_game_input.nbytes)
        #d_policy_output = cuda.mem_alloc(h_policy_output.nbytes)
        #d_value_output  = cuda.mem_alloc(h_value_output.nbytes)
        
        #stream = cuda.Stream()

        cuda.memcpy_htod_async(self.trt_d_game_input, h_game_input, self.trt_stream)

        self.trt_context.execute_async(bindings=[int(self.trt_d_game_input), 
                                                 int(self.trt_d_policy_output),
                                                 int(self.trt_d_value_output)],
                                                 #int(self.trt_d_value_output),
                                                 #int(self.trt_d_intermediate_output)],
                                       stream_handle=self.trt_stream.handle)

        cuda.memcpy_dtoh_async(h_policy_output, self.trt_d_policy_output, self.trt_stream)
        cuda.memcpy_dtoh_async(h_value_output, self.trt_d_value_output, self.trt_stream)
        #cuda.memcpy_dtoh_async(h_intermediate_output, self.trt_d_intermediate_output, self.trt_stream)

        # Synchronize the stream
        self.trt_stream.synchronize()

        #return h_policy_output, h_value_output, h_intermediate_output
        return h_policy_output, h_value_output
        


    def _trt_res_layer(self, input_layer, weights, name="01", dtype=np.float32):
    
        if dtype == np.dtype('float32'):
            trt_type = trt.float32
            #print('float32')
        elif dtype == np.dtype('float16'):
            trt_type = trt.float16
            #print('float16')
        elif dtype == np.dtype('int32'):
            trt_type = trt.int32
            #print('int32')
        elif dtype == np.dtype('int8'):
            trt_type = trt.int8
            #print('int8')
        else:
            raise Exception("Unknown data type [%s]" % dtype)
            
        # conv x_a
        conv_x_a_w = weights['resnet_'+name+'a_conv.weight'].detach().cpu().numpy().astype(dtype)
        conv_x_a   = self.trt_network.add_convolution(input_layer.get_output(0),
                                                      num_output_maps=self.CHANNELS,
                                                      kernel_shape=(3, 3), kernel=conv_x_a_w,
                                                      bias=trt.Weights())
        conv_x_a.stride    = (1, 1)
        conv_x_a.padding   = (1, 1)
        conv_x_a.precision = trt_type

        conv_x_a_bn_w = weights['resnet_'+name+'a_bn.weight'].detach().cpu().numpy().astype(dtype)
        conv_x_a_bn_b = weights['resnet_'+name+'a_bn.bias'].detach().cpu().numpy().astype(dtype)
        conv_x_a_bn   = self.trt_network.add_scale(conv_x_a.get_output(0),
                                                   mode=trt.tensorrt.ScaleMode.CHANNEL,
                                                   shift=conv_x_a_bn_b, scale=conv_x_a_bn_w,
                                                   power=np.ones_like(conv_x_a_bn_w, dtype=dtype))
        conv_x_a_bn.precision = trt_type

        conv_x_a_actv           = self.trt_network.add_activation(conv_x_a_bn.get_output(0), trt.ActivationType.RELU)
        conv_x_a_actv.precision = trt_type

        # conv_x_b
        conv_x_b_w = weights['resnet_'+name+'b_conv.weight'].detach().cpu().numpy().astype(dtype)
        conv_x_b   = self.trt_network.add_convolution(conv_x_a_actv.get_output(0),
                                                      num_output_maps=self.CHANNELS,
                                                      kernel_shape=(3, 3), kernel=conv_x_b_w,
                                                      bias=trt.Weights())
        conv_x_b.stride    = (1, 1)
        conv_x_b.padding   = (1, 1)
        conv_x_b.precision = trt_type

        conv_x_b_bn_w = weights['resnet_'+name+'b_bn.weight'].detach().cpu().numpy().astype(dtype)
        conv_x_b_bn_b = weights['resnet_'+name+'b_bn.bias'].detach().cpu().numpy().astype(dtype)
        conv_x_b_bn   = self.trt_network.add_scale(conv_x_b.get_output(0),
                                                   mode=trt.tensorrt.ScaleMode.CHANNEL,
                                                   shift=conv_x_b_bn_b, scale=conv_x_b_bn_w,
                                                   power=np.ones_like(conv_x_b_bn_w, dtype=dtype))
        conv_x_b_bn.precision = trt_type

        conv_x_b_res  = self.trt_network.add_elementwise(conv_x_b_bn.get_output(0),
                                                         input_layer.get_output(0),
                                                         trt.tensorrt.ElementWiseOperation.SUM)
        conv_x_b_res.precision = trt_type

        conv_x_b_actv = self.trt_network.add_activation(conv_x_b_res.get_output(0), trt.ActivationType.RELU)
        conv_x_b_actv.precision = trt_type

        return conv_x_b_actv
    

    def build_trt_engine(self, state_dict, dtype=np.float32):
        
        if self._args.model_arch != "conv3" and self._args.model_arch != "conv4" and self._args.model_arch != "resnet":
            raise Exception("Unsupport model arch [%s]" % self._args.model_arch)

        if dtype == np.dtype('float32'):
            trt_type = trt.float32
            #print('float32')
        elif dtype == np.dtype('float16'):
            trt_type = trt.float16
            #print('float16')
        elif dtype == np.dtype('int32'):
            trt_type = trt.int32
            #print('int32')
        elif dtype == np.dtype('int8'):
            trt_type = trt.int8
            #print('int8')
        else:
            raise Exception("Unknown data type [%s]" % dtype)
            
        #print(trt_type, trt.nptype(trt_type))
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.TRT_LOGGER)
        #print(self.trt_runtime)

        self.trt_builder = trt.Builder(self.TRT_LOGGER)
        self.trt_builder.max_batch_size = 1
        self.trt_builder.max_workspace_size = 2**20
        if dtype == np.dtype('float16'):
            self.trt_builder.fp16_mode = True
        #self.trt_builder.strict_type_constraints = True
        
        self.trt_network = self.trt_builder.create_network()

        # Configure the network layers based on the weights provided.
        # In this case, the weights are imported from a pytorch model. 
        # Add an input layer. The name is a string, dtype is a TensorRT dtype,
        # and the shape can be provided as either a list or tuple.
        input_tensor = self.trt_network.add_input(name="INPUT", dtype=trt_type,
                                                  shape=(self._state_layer, self._size_x, self._size_y))
        #input_tensor.precision = trt_type

        #weights = dict(self.named_parameters())
        weights = state_dict

        if self._args.model_arch == "conv3":
            
            # common convolution network
            conv1_w = weights['conv1.weight'].detach().cpu().numpy().astype(dtype)
            conv1_b = weights['conv1.bias'].detach().cpu().numpy().astype(dtype)
            conv1   = self.trt_network.add_convolution(input=input_tensor, num_output_maps=self.CHANNELS,
                                                       kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
            conv1.stride    = (1, 1)
            conv1.padding   = (1, 1)
            conv1.precision = trt_type

            #intermediate = conv1
            #intermediate.get_output(0).name = "INTERMEDIATE"
            
            conv1_actv           = self.trt_network.add_activation(conv1.get_output(0), trt.ActivationType.RELU)
            conv1_actv.precision = trt_type

            conv2_w = weights['conv2.weight'].detach().cpu().numpy().astype(dtype)
            conv2_b = weights['conv2.bias'].detach().cpu().numpy().astype(dtype)
            conv2   = self.trt_network.add_convolution(conv1_actv.get_output(0), num_output_maps=self.CHANNELS*2,
                                                       kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
            conv2.stride    = (1, 1)
            conv2.padding   = (1, 1)
            conv2.precision = trt_type

            conv2_actv           = self.trt_network.add_activation(conv2.get_output(0), trt.ActivationType.RELU)
            conv2_actv.precision = trt_type

            conv3_w = weights['conv3.weight'].detach().cpu().numpy().astype(dtype)
            conv3_b = weights['conv3.bias'].detach().cpu().numpy().astype(dtype)
            conv3   = self.trt_network.add_convolution(conv2_actv.get_output(0), num_output_maps=self.CHANNELS*4,
                                                       kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
            conv3.stride    = (1, 1)
            conv3.padding   = (1, 1)
            conv3.precision = trt_type

            conv3_actv           = self.trt_network.add_activation(conv3.get_output(0), trt.ActivationType.RELU)
            conv3_actv.precision = trt_type

            #conv4_w = weights['conv4.weight'].detach().cpu().numpy().astype(dtype)
            #conv4_b = weights['conv4.bias'].detach().cpu().numpy().astype(dtype)
            #conv4   = self.trt_network.add_convolution(conv3_actv.get_output(0), num_output_maps=self.CHANNELS*4,
            #                                           kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
            #conv4.stride    = (1, 1)
            #conv4.padding   = (1, 1)
            #conv4.precision = trt_type

            #conv4_actv           = self.trt_network.add_activation(conv4.get_output(0), trt.ActivationType.RELU)
            #conv4_actv.precision = trt_type

            # policy head
            pc_w = weights['pc.weight'].detach().cpu().numpy().astype(dtype)
            pc_b = weights['pc.bias'].detach().cpu().numpy().astype(dtype)
            pc   = self.trt_network.add_convolution(conv3_actv.get_output(0), num_output_maps=self.CHANNELS_P,
                                                    kernel_shape=(1, 1), kernel=pc_w, bias=pc_b)
            pc.stride    = (1, 1)
            pc.precision = trt_type

            pc_actv           = self.trt_network.add_activation(pc.get_output(0), trt.ActivationType.RELU)
            pc_actv.precision = trt_type

            p1_w = weights['p1.weight'].detach().cpu().numpy().astype(dtype)
            p1_b = weights['p1.bias'].detach().cpu().numpy().astype(dtype)
            p1   = self.trt_network.add_fully_connected(pc_actv.get_output(0), num_outputs=self._action_size,
                                                        kernel=p1_w, bias=p1_b)
            p1.precision = trt_type

            p1_softmax = self.trt_network.add_softmax(p1.get_output(0))
            p1_softmax.get_output(0).name = "POLICY"
            p1_softmax.precision = trt_type

            # value head
            vc_w = weights['vc.weight'].detach().cpu().numpy().astype(dtype)
            vc_b = weights['vc.bias'].detach().cpu().numpy().astype(dtype)
            vc   = self.trt_network.add_convolution(conv3_actv.get_output(0), num_output_maps=self.CHANNELS_V,
                                                    kernel_shape=(1,1), kernel=vc_w, bias=vc_b)
            vc.stride    = (1, 1)
            vc.precision = trt_type

            vc_actv           = self.trt_network.add_activation(vc.get_output(0), trt.ActivationType.RELU)
            vc_actv.precision = trt_type

            v1_w = weights['v1.weight'].detach().cpu().numpy().astype(dtype)
            v1_b = weights['v1.bias'].detach().cpu().numpy().astype(dtype)
            v1   = self.trt_network.add_fully_connected(vc_actv.get_output(0), num_outputs=64, kernel=v1_w, bias=v1_b)
            v1.precision = trt_type

            v1_actv           = self.trt_network.add_activation(v1.get_output(0), trt.ActivationType.RELU)
            v1_actv.precision = trt_type

            v2_w = weights['v2.weight'].detach().cpu().numpy().astype(dtype)
            v2_b = weights['v2.bias'].detach().cpu().numpy().astype(dtype)
            v2   = self.trt_network.add_fully_connected(v1_actv.get_output(0), num_outputs=1, kernel=v2_w, bias=v2_b)
            v2.precision = trt_type

            v2_tanh = self.trt_network.add_activation(v2.get_output(0), trt.ActivationType.TANH)
            v2_tanh.get_output(0).name = "VALUE"
            v2_tanh.precision = trt_type

            #print(p1_softmax.precision)
            #print(v2_tanh.precision)
            
        if self._args.model_arch == "conv4":
            
            # common convolution network
            conv1_w = weights['conv1.weight'].detach().cpu().numpy().astype(dtype)
            conv1_b = weights['conv1.bias'].detach().cpu().numpy().astype(dtype)
            conv1   = self.trt_network.add_convolution(input=input_tensor, num_output_maps=self.CHANNELS,
                                                       kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
            conv1.stride    = (1, 1)
            conv1.padding   = (1, 1)
            conv1.precision = trt_type

            #intermediate = conv1
            #intermediate.get_output(0).name = "INTERMEDIATE"
            
            conv1_actv           = self.trt_network.add_activation(conv1.get_output(0), trt.ActivationType.RELU)
            conv1_actv.precision = trt_type

            conv2_w = weights['conv2.weight'].detach().cpu().numpy().astype(dtype)
            conv2_b = weights['conv2.bias'].detach().cpu().numpy().astype(dtype)
            conv2   = self.trt_network.add_convolution(conv1_actv.get_output(0), num_output_maps=self.CHANNELS*2,
                                                       kernel_shape=(3, 3), kernel=conv2_w, bias=conv2_b)
            conv2.stride    = (1, 1)
            conv2.padding   = (1, 1)
            conv2.precision = trt_type

            conv2_actv           = self.trt_network.add_activation(conv2.get_output(0), trt.ActivationType.RELU)
            conv2_actv.precision = trt_type

            conv3_w = weights['conv3.weight'].detach().cpu().numpy().astype(dtype)
            conv3_b = weights['conv3.bias'].detach().cpu().numpy().astype(dtype)
            conv3   = self.trt_network.add_convolution(conv2_actv.get_output(0), num_output_maps=self.CHANNELS*4,
                                                       kernel_shape=(3, 3), kernel=conv3_w, bias=conv3_b)
            conv3.stride    = (1, 1)
            conv3.padding   = (1, 1)
            conv3.precision = trt_type

            conv3_actv           = self.trt_network.add_activation(conv3.get_output(0), trt.ActivationType.RELU)
            conv3_actv.precision = trt_type

            conv4_w = weights['conv4.weight'].detach().cpu().numpy().astype(dtype)
            conv4_b = weights['conv4.bias'].detach().cpu().numpy().astype(dtype)
            conv4   = self.trt_network.add_convolution(conv3_actv.get_output(0), num_output_maps=self.CHANNELS*4,
                                                       kernel_shape=(3, 3), kernel=conv4_w, bias=conv4_b)
            conv4.stride    = (1, 1)
            conv4.padding   = (1, 1)
            conv4.precision = trt_type

            conv4_actv           = self.trt_network.add_activation(conv4.get_output(0), trt.ActivationType.RELU)
            conv4_actv.precision = trt_type

            # policy head
            pc_w = weights['pc.weight'].detach().cpu().numpy().astype(dtype)
            pc_b = weights['pc.bias'].detach().cpu().numpy().astype(dtype)
            pc   = self.trt_network.add_convolution(conv4_actv.get_output(0), num_output_maps=self.CHANNELS_P,
                                                    kernel_shape=(1, 1), kernel=pc_w, bias=pc_b)
            pc.stride    = (1, 1)
            pc.precision = trt_type

            pc_actv           = self.trt_network.add_activation(pc.get_output(0), trt.ActivationType.RELU)
            pc_actv.precision = trt_type

            p1_w = weights['p1.weight'].detach().cpu().numpy().astype(dtype)
            p1_b = weights['p1.bias'].detach().cpu().numpy().astype(dtype)
            p1   = self.trt_network.add_fully_connected(pc_actv.get_output(0), num_outputs=self._action_size,
                                                        kernel=p1_w, bias=p1_b)
            p1.precision = trt_type

            p1_softmax = self.trt_network.add_softmax(p1.get_output(0))
            p1_softmax.get_output(0).name = "POLICY"
            p1_softmax.precision = trt_type

            # value head
            vc_w = weights['vc.weight'].detach().cpu().numpy().astype(dtype)
            vc_b = weights['vc.bias'].detach().cpu().numpy().astype(dtype)
            vc   = self.trt_network.add_convolution(conv4_actv.get_output(0), num_output_maps=self.CHANNELS_V,
                                                    kernel_shape=(1,1), kernel=vc_w, bias=vc_b)
            vc.stride    = (1, 1)
            vc.precision = trt_type

            vc_actv           = self.trt_network.add_activation(vc.get_output(0), trt.ActivationType.RELU)
            vc_actv.precision = trt_type

            v1_w = weights['v1.weight'].detach().cpu().numpy().astype(dtype)
            v1_b = weights['v1.bias'].detach().cpu().numpy().astype(dtype)
            v1   = self.trt_network.add_fully_connected(vc_actv.get_output(0), num_outputs=64, kernel=v1_w, bias=v1_b)
            v1.precision = trt_type

            v1_actv           = self.trt_network.add_activation(v1.get_output(0), trt.ActivationType.RELU)
            v1_actv.precision = trt_type

            v2_w = weights['v2.weight'].detach().cpu().numpy().astype(dtype)
            v2_b = weights['v2.bias'].detach().cpu().numpy().astype(dtype)
            v2   = self.trt_network.add_fully_connected(v1_actv.get_output(0), num_outputs=1, kernel=v2_w, bias=v2_b)
            v2.precision = trt_type

            v2_tanh = self.trt_network.add_activation(v2.get_output(0), trt.ActivationType.TANH)
            v2_tanh.get_output(0).name = "VALUE"
            v2_tanh.precision = trt_type

            #print(p1_softmax.precision)
            #print(v2_tanh.precision)
            
        elif self._args.model_arch == "resnet":

            # common convolution network
            input_conv_w = weights['resnet_input_conv.weight'].detach().cpu().numpy().astype(dtype)
            input_conv   = self.trt_network.add_convolution(input=input_tensor, num_output_maps=self.CHANNELS,
                                                            kernel_shape=(3, 3), kernel=input_conv_w,
                                                            bias=trt.Weights())

            input_conv.stride    = (1, 1)
            input_conv.padding   = (1, 1)
            input_conv.precision = trt_type

            #intermediate = input_conv
            #intermediate.get_output(0).name = "INTERMEDIATE"
            
            input_conv_bn_w = weights['resnet_input_bn.weight'].detach().cpu().numpy().astype(dtype)
            input_conv_bn_b = weights['resnet_input_bn.bias'].detach().cpu().numpy().astype(dtype)
            input_conv_bn   = self.trt_network.add_scale(input_conv.get_output(0),
                                                         mode=trt.tensorrt.ScaleMode.CHANNEL,
                                                         shift=input_conv_bn_b,
                                                         scale=input_conv_bn_w,
                                                         power=np.ones_like(input_conv_bn_w, dtype=dtype))
            input_conv_bn.precision = trt_type
            
            input_conv_actv           = self.trt_network.add_activation(input_conv_bn.get_output(0), trt.ActivationType.RELU)
            input_conv_actv.precision = trt_type

            # resnet
            res_layer = input_conv_actv
            for i in range(1, 20):
                name = "%02d" % i
                res_layer = self._trt_res_layer(res_layer, weights, name=name, dtype=dtype)

            # policy head
            pc_w = weights['pc.weight'].detach().cpu().numpy().astype(dtype)
            pc   = self.trt_network.add_convolution(res_layer.get_output(0),
                                                    num_output_maps=self.CHANNELS_P,
                                                    kernel_shape=(1, 1), kernel=pc_w,
                                                    bias=trt.Weights())
            pc.stride    = (1, 1)
            pc.precision = trt_type

            pc_bn_w = weights['pc_bn.weight'].detach().cpu().numpy().astype(dtype)
            pc_bn_b = weights['pc_bn.bias'].detach().cpu().numpy().astype(dtype)
            pc_bn   = self.trt_network.add_scale(pc.get_output(0),
                                                 mode=trt.tensorrt.ScaleMode.CHANNEL,
                                                 shift=pc_bn_b, scale=pc_bn_w,
                                                 power=np.ones_like(pc_bn_w, dtype=dtype))
            pc_bn.precision = trt_type
            
            pc_actv           = self.trt_network.add_activation(pc_bn.get_output(0), trt.ActivationType.RELU)
            pc_actv.precision = trt_type

            p1_w = weights['p1.weight'].detach().cpu().numpy().astype(dtype)
            p1_b = weights['p1.bias'].detach().cpu().numpy().astype(dtype)
            p1   = self.trt_network.add_fully_connected(pc_actv.get_output(0), num_outputs=self._action_size,
                                                        kernel=p1_w, bias=p1_b)
            p1.precision = trt_type

            p1_softmax = self.trt_network.add_softmax(p1.get_output(0))
            p1_softmax.get_output(0).name = "POLICY"
            p1_softmax.precision = trt_type

            # value head
            vc_w = weights['vc.weight'].detach().cpu().numpy().astype(dtype)
            vc   = self.trt_network.add_convolution(res_layer.get_output(0),
                                                    num_output_maps=self.CHANNELS_V,
                                                    kernel_shape=(1,1), kernel=vc_w,
                                                    bias=trt.Weights())
            vc.stride    = (1, 1)
            vc.precision = trt_type

            vc_bn_w = weights['vc_bn.weight'].detach().cpu().numpy().astype(dtype)
            vc_bn_b = weights['vc_bn.bias'].detach().cpu().numpy().astype(dtype)
            vc_bn   = self.trt_network.add_scale(vc.get_output(0),
                                                 mode=trt.tensorrt.ScaleMode.CHANNEL,
                                                 shift=vc_bn_b, scale=vc_bn_w,
                                                 power=np.ones_like(vc_bn_w, dtype=dtype))
            vc_bn.precision = trt_type

            vc_actv           = self.trt_network.add_activation(vc_bn.get_output(0), trt.ActivationType.RELU)
            vc_actv.precision = trt_type

            v1_w = weights['v1.weight'].detach().cpu().numpy().astype(dtype)
            v1_b = weights['v1.bias'].detach().cpu().numpy().astype(dtype)
            v1   = self.trt_network.add_fully_connected(vc_actv.get_output(0), num_outputs=256, kernel=v1_w, bias=v1_b)
            v1.precision = trt_type

            #self.trt_network.mark_output(v1.get_output(0))
            #self.trt_engine = self.trt_builder.build_cuda_engine(self.trt_network)
            #print(self.trt_engine)

            v1_actv           = self.trt_network.add_activation(v1.get_output(0), trt.ActivationType.RELU)
            v1_actv.precision = trt_type

            v2_w = weights['v2.weight'].detach().cpu().numpy().astype(dtype)
            v2_b = weights['v2.bias'].detach().cpu().numpy().astype(dtype)
            v2   = self.trt_network.add_fully_connected(v1_actv.get_output(0), num_outputs=1, kernel=v2_w, bias=v2_b)
            v2.precision = trt_type

            v2_tanh = self.trt_network.add_activation(v2.get_output(0), trt.ActivationType.TANH)
            v2_tanh.get_output(0).name = "VALUE"
            v2_tanh.precision = trt_type

            #print(p1_softmax.precision)
            #print(v2_tanh.precision)

        self.trt_network.mark_output(p1_softmax.get_output(0))
        self.trt_network.mark_output(v2_tanh.get_output(0))
        #self.trt_network.mark_output(intermediate.get_output(0))

        self.trt_engine = self.trt_builder.build_cuda_engine(self.trt_network)
        #print(self.trt_engine)
        #print(self.trt_engine.get_binding_shape(0))
        #print(self.trt_engine.get_binding_shape(1))
        #print(self.trt_engine.get_binding_shape(2))

        self.trt_context = self.trt_engine.create_execution_context()
        #print(self.trt_context)
            
        # Create a stream in which to copy inputs/outputs and run inference.
        # Allocate device memory for inputs and outputs.
        self.trt_stream = cuda.Stream()
        #print(self.trt_stream)

        # Allocate device memory for inputs and outputs.
        game = Game(self._args)
        h_game_input    = game.state_normalized().astype(np.dtype(self._args.play_dtype))
        h_policy_output = np.empty(game.action_size(), dtype = np.dtype(self._args.play_dtype))
        h_value_output  = np.empty(1, dtype = np.dtype(self._args.play_dtype))
        #h_intermediate_output  = np.empty(self.CHANNELS*self._size_x*self._size_y, dtype = np.dtype(self._args.play_dtype))
        #print(h_game_input.nbytes, h_policy_output.nbytes, h_value_output.nbytes)

        self.trt_d_game_input    = cuda.mem_alloc(h_game_input.nbytes)
        self.trt_d_policy_output = cuda.mem_alloc(h_policy_output.nbytes)
        self.trt_d_value_output  = cuda.mem_alloc(h_value_output.nbytes)
        #self.trt_d_intermediate_output  = cuda.mem_alloc(h_intermediate_output.nbytes)
        
        print("Loaded TENSORRT Engine [%s] [%s, %s, %s] [%d, %d, %d]" % (trt_type,
        #print("Loaded TENSORRT Engine [%s] [%s, %s, %s, %s] [%d, %d, %d, %d]" % (trt_type,
                                                                         self.trt_engine.get_binding_shape(0), 
                                                                         self.trt_engine.get_binding_shape(1),
                                                                         self.trt_engine.get_binding_shape(2),
                                                                         #self.trt_engine.get_binding_shape(3),
                                                                         h_game_input.nbytes,
                                                                         h_policy_output.nbytes,
                                                                         h_value_output.nbytes))
                                                                         #h_value_output.nbytes,
                                                                         #h_intermediate_output.nbytes))

        if self._args.model_tensorrt == 2:
            #self.conv1  = None
            #self.conv2  = None
            #self.conv3  = None
            #self.conv4  = None

            # policy head
            self.pc     = None
            self.p1     = None

            # value head
            self.vc     = None
            self.v1     = None
            self.v2     = None
            
            torch.cuda.empty_cache()

            print("Cleaned ORIGINAL Model")
            

            
            
model = None
game = None

def test_model():
    
    global model, game

    args = parse_args()

    game  = Game(args)
    model = Model(game.state_shape(), game.action_size(), args)

    print(model)
    params = dict(model.named_parameters())
    print(len(params))
    #print(params)
    model_size = model.model_size()
    total_size = 0
    for name, p in params.items():
        if p.requires_grad:
            layer_shape = p.size()
            layer_size  = np.prod(list(layer_shape))
            total_size += layer_size
            print(name, layer_shape, layer_size)
    print("##### Total Parameters : [%d, %d] #####" % (model_size, total_size))
    
    #print(model.state_dict())
    
    print("CONVERT TO BASE64")
    base64_model = model.to_base64()
    #print(base64_model)
    print("CONVERT FROM BASE64")
    model = model.from_base64(base64_model)
    print(model)

    print("-"*30)
    print("CONVER TO ONNX")
    print("Dummy Input :", game.state_normalized().shape)
    dummy_input = torch.randn(game.state_normalized().shape)
    if args.model_cuda >= 0 and args.model_tensorrt != 2:
        dummy_input = dummy_input.contiguous().cuda()
    #print(dummy_input)
    #onnx_buffer = io.BytesIO()
    #torch.onnx.export(model, dummy_input, onnx_buffer, export_params=True)
    #torch.onnx.export(model, dummy_input, "model.onnx")
    #print(onnx_buffer.getvalue())
    #print("ONNX Model Size : ", len(onnx_buffer.getvalue()))

    print("-"*30)
    print("INFERENCE WITH ORIGINAL MODEL")
    
    #policy, value, intermediate = model.game_predict(game.state_normalized())
    policy, value = model.game_predict(game.state_normalized())
    print(policy)
    print(value)
    #print(intermediate)
    
    print("-"*30)
    print("INFERENCE WITH TENSORRT")
    #onnx_inf_model = onnx.load("model.onnx")
    #onnx_inf_model = onnx.load(io.BytesIO(onnx_buffer.getvalue()))
    #print(onnx_inf_model)
    #engine = backend.prepare(onnx_inf_model, device='CUDA:'+str(args.model_cuda))
    #print(engine)

    model.build_trt_engine(model.state_dict(), dtype=np.dtype(args.play_dtype))
    #policy_output, value_output, intermediate_output = model.game_predict_trt(game.state_normalized())
    policy_output, value_output = model.game_predict_trt(game.state_normalized())
    
    #intermediate_output = np.reshape(intermediate_output, (model.CHANNELS, model._size_x, model._size_y))

    print(policy_output)
    print(value_output)
    #print(intermediate_output)

    #inputs, outputs, bindings, stream = model.allocate_buffers(engine)
    #with engine.create_execution_context() as context:
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        #[output] = model.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        #print("Prediction: " + str(output))


        
 
    # timeit
    game = Game(args)
    model = Model(game.state_shape(), game.action_size(), args)
    
    t = timeit.timeit("model.game_predict(game.state_normalized())", globals=globals(), number=args.timeit_count)
    print("timeit [pytorch] : %s ms  [%d] [total %s s]" % (round((t/args.timeit_count)*1000,2), args.timeit_count, round(t,2)))

    # timeit
    game = Game(args)
    model = Model(game.state_shape(), game.action_size(), args)
    model.build_trt_engine(model.state_dict(), dtype=np.dtype(args.play_dtype))
    
    t = timeit.timeit("model.game_predict_trt(game.state_normalized())", globals=globals(), number=args.timeit_count)
    print("timeit [tensorrt] : %s ms  [%d] [total %s s]" % (round((t/args.timeit_count)*1000,2), args.timeit_count, round(t,2)))

        
if __name__ == "__main__":
    test_model()

