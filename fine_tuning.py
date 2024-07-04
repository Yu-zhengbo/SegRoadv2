import torch
from torch import nn
from nets.segformer import SegFormer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.depoly = False
    def forward(self, x):
        if self.depoly:
            return self.act(self.rep_conv(x))
        
        return self.act(self.bn(self.conv(x)))


    def _fuse_bn_tensor(self, conv,bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self,'rep_conv'):
            return
        kernel,bias = self._fuse_bn_tensor(self.conv,self.bn)
        self.rep_conv = nn.Conv2d(kernel.shape[1], kernel.shape[0], kernel_size=kernel.shape[2:], stride=self.conv.stride, padding=self.conv.padding, bias=True)
        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias

        self.__delattr__('conv')
        self.__delattr__('bn')
        self.depoly = True





if __name__ == "__main__":
    model   = SegFormer(num_classes=2, phi='b0', pretrained=False)
    model.eval()
    input = torch.randn(2,3,512,512)
    output = model(input)
    print(output.shape)
    # for name, module in model.named_children():
    #     print('children module:', name, module)

    # model = ConvModule(3,6,k=3, s=1, p=1, g=1, act=True)
    # model.eval()
    # input = torch.randn(2,3,512,512)
    # output1 = model(input)
    # model.switch_to_deploy()
    # output2 = model(input)
    # print(torch.sum(output1-output2))