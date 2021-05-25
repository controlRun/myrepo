import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# W(权重)量化
class B1Weight(Function):
    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input
    
    
# A(激活)量化
class B1ActQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    
class B2ActQuantize(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        #input= input.data.clamp_(-1.0, 1.0)
        input = BN_convparams(input) 
        x = input
        num_bits=2
        v0 = 1
        v1 = 2
        v2 = -0.5
        y = 2.**num_bits - 1.
        x = x.add(v0).div(v1)
        x = x.mul(y).round_()
        x = x.div(y)
        x = x.add(v2)
        x = x.mul(v1)
        output = x
        return output
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input.ge(1)] = 0
        #grad_input[input.le(-1)] = 0
        return grad_input
    
# ********************* W(模型参数)量化(四/二值) ***********************
def meancenter_clamp_convparams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub_(mean)        # W中心化(C方向)
    w.data.clamp_(-1.0, 1.0)  # W截断
    return w

class B1WeightQuantizer(nn.Module):
    def __init__(self):
        super(B1WeightQuantizer, self).__init__()

    def binary(self, input):
        output = B1Weight.apply(input)
        return output

    def forward(self, input):
        # **************************************** W二值 *****************************************
        output = meancenter_clamp_convparams(input)  # W中心化+截断
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
        # **************** α(缩放因子) ****************
        alpha = E
        # ************** W —— +-1 **************
        output = self.binary(output)
        # ************** W * α **************
        #output = output * alpha  # 若不需要α(缩放因子)，注释掉即可

        return output
    
class Round(Function):
    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def BN_convparams(w):
    bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
    bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
    bw.data.clamp_(-1.0, 1.0)  # W截断
    
    return bw
    
class B2WeightQuantizer(nn.Module):
    def __init__(self):
        super(B2WeightQuantizer, self).__init__()

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        output = BN_convparams(input)  # W中心化+截断
        
        #E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        # **************** α(缩放因子) ****************
        #alpha = E
        
        x = output
        num_bits=2
        v0 = 1
        v1 = 2
        v2 = -0.5
        y = 2.**num_bits - 1.
        x = x.add(v0).div(v1)
        x = x.mul(y).round_()
        x = x.div(y)
        x = x.add(v2)
        x = x.mul(v1)
        
        #x = x * alpha  # 若不需要α(缩放因子)，注释掉即可
        
        return x   

class B1Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(B1Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_quantizer = B1WeightQuantizer()
        
    
    def b1activation(self, input):
        output = B1ActQuantize.apply(input)
        return output
    
    def forward(self, input):
        w = self.weight
        bw = self.weight_quantizer(w)
        
        a = input
        ba = self.b1activation(a)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
    
    
class B2Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(B2Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight_quantizer = B2WeightQuantizer()
        #self.ba_hook = []
        #self.bw_hook = []
        
    def b2activation(self, input):
        output = B2ActQuantize.apply(input)
        return output
    
    def forward(self, input):
        w = self.weight
        bw = self.weight_quantizer(w)
        a = input
        ba = self.b2activation(a)
        

        
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output    
    
    
def add_quant_op(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if name in []:
                if child.bias is not None:
                    quant_conv = B1Conv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = B1Conv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
            elif name in ["conv1","conv2","conv3","conv4"]:
                if child.bias is not None:
                    quant_conv = B2Conv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = B2Conv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv


def prepare(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(model)
    return model

