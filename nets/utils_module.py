import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)




class DecoderBlock_x(nn.Module):
    def __init__(self, in_channels, n_filters,group=1, BatchNorm=nn.BatchNorm2d, inp=False):
        super(DecoderBlock_x, self).__init__()
        self.conv1 = nn.Conv2d(in_channels//group, in_channels // 2//group, 1,bias=False)
        self.bn1 = BatchNorm(in_channels // 2//group)
        self.relu1 = nn.ReLU()
        self.inp = inp
        self.group = group
        self.deconv1 = nn.Conv2d(
            in_channels // 2//group, in_channels // 4//group, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 2//group, in_channels // 4//group, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 2//group, in_channels // 4//group, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 2//group, in_channels // 4//group, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 2//group)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 2//group, n_filters//group, 1,bias=False)
        self.bn3 = BatchNorm(n_filters//group)
        self.relu3 = nn.ReLU()
        self.deploy = False
        self._init_weight()

    def forward(self, x, inp = False):
        if self.deploy:
            B,_,H,W = x.shape
            x = x.view(B*self.group,-1,H,W)
            x = self.relu1(self.rep_conv1(x))

            x1 = self.rep_deconv1(x)
            x = self.transform(x)
            x3 = self.inv_transform(self.rep_deconv3(x))
            x = torch.cat((x1, x3), 1)
            if self.inp:
                x = F.interpolate(x, scale_factor=2)
            x = self.bn2(x)
            x = self.relu2(x)

            x = self.relu3(self.rep_conv3(x))
            x = x.view(B,-1,H,W)
            return x
        
        B,_,H,W = x.shape
        x = x.view(B*self.group,-1,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x) + self.deconv2(x)
        x = self.transform(x)
        x3 = self.inv_transform(self.deconv3(x)+self.deconv4(x))
        x = torch.cat((x1, x3), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(B,-1,H,W)
        return x

    def transI_fusebn(self,kernel, bn):
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

    def transII_fuseconv(self,kernel1,kernel2,b1,b2):
        kernel1 = F.pad(kernel1,(0,0,4,4),mode='constant',value=0)
        kernel2 = F.pad(kernel2,(4,4,0,0),mode='constant',value=0)
        return kernel1+kernel2,b1+b2


    def switch_to_deploy(self):
        kernel, bias = self.transI_fusebn(self.conv1.weight,self.bn1)
        self.rep_conv1 = nn.Conv2d(kernel.shape[1],kernel.shape[0],1,1,0,bias=True)
        self.rep_conv1.weight.data = kernel
        self.rep_conv1.bias.data = bias
        
        kernel, bias = self.transI_fusebn(self.conv3.weight,self.bn3)
        self.rep_conv3 = nn.Conv2d(kernel.shape[1],kernel.shape[0],1,1,0,bias=True)
        self.rep_conv3.weight.data = kernel
        self.rep_conv3.bias.data = bias

        kernel,bias = self.transII_fuseconv(self.deconv1.weight,self.deconv2.weight,
                                            self.deconv1.bias,self.deconv2.bias)
        self.rep_deconv1 = nn.Conv2d(kernel.shape[1],kernel.shape[0],9,1,4,bias=True)
        self.rep_deconv1.weight.data = kernel
        self.rep_deconv1.bias.data = bias


        kernel,bias = self.transII_fuseconv(self.deconv4.weight,self.deconv3.weight,
                                            self.deconv4.bias,self.deconv3.bias)
        self.rep_deconv3 = nn.Conv2d(kernel.shape[1],kernel.shape[0],9,1,4,bias=True)
        self.rep_deconv3.weight.data = kernel
        self.rep_deconv3.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('bn1')
        self.__delattr__('conv3')
        self.__delattr__('bn3')
        self.__delattr__('deconv1')
        self.__delattr__('deconv2')
        self.__delattr__('deconv3')
        self.__delattr__('deconv4')
        
        self.deploy = True
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def transform(self, x):
        if not hasattr(self,'pad_length'):
            self.pad_length = int((math.sqrt(2*x.shape[2]**2)-x.shape[2])/2)+1
        x = F.pad(x, (self.pad_length, self.pad_length, self.pad_length, self.pad_length), mode='constant', value=0)

        angle = -45/180*math.pi
        # 创建一个坐标变换矩阵
        transform_matrix = torch.tensor([
                [math.cos(angle),math.sin(-angle),0],
                [math.sin(angle),math.cos(angle),0]])
     
        transform_matrix = transform_matrix.unsqueeze(0).repeat(x.shape[0],1,1).to(x.device)

        grid = F.affine_grid(transform_matrix, # 旋转变换矩阵
                        x.shape,
                        align_corners=True)	# 变换后的tensor的shape(与输入tensor相同)

        x = F.grid_sample(x, # 输入tensor，shape为[B,C,W,H]
                            grid, mode='nearest',
                            align_corners=True)# 上一步输出的gird,shape为[B,C,W,H]
        return x

    def inv_transform(self, x):
        angle = -45/180*math.pi
        transform_matrix = torch.tensor([
            [math.cos(-angle),math.sin(angle),0],
            [math.sin(-angle),math.cos(-angle),0]])
        transform_matrix = transform_matrix.unsqueeze(0).repeat(x.shape[0],1,1).to(x.device)
        grid = F.affine_grid(transform_matrix, # 旋转变换矩阵
                        x.shape,
                        align_corners=True)	# 变换后的tensor的shape(与输入tensor相同)

        x = F.grid_sample(x, # 输入tensor，shape为[B,C,W,H]
                            grid, mode='nearest',
                            align_corners=True)# 上一步输出的gird,shape为[B,C,W,H]
        return x[:,:,self.pad_length:-self.pad_length,self.pad_length:-self.pad_length]





if __name__ == "__main__":
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from thop import profile
    model = DecoderBlock_x(64,64,4)
    input = torch.randn(2,64,256,256)
    input_1 = torch.randn(2,64,256,256)
    model.eval()

    # t1 = time.time()
    # for _ in range(100):
    #     model(input)
    # t2 = time.time()

    output1 = model(input)
    output_1 = model(input_1)
    model.switch_to_deploy()
    output2 = model(input)
    model.eval()

    # t3 = time.time()
    # for _ in range(100):
    #     model(input)
    # t4 = time.time()

    # print(t2-t1,t4-t3)

    print(torch.sum(output1-output2))
    print(torch.sum(output1-output_1))
    """
    tensor(-0.0084, grad_fn=<SumBackward0>)
    tensor(-16931.8867, grad_fn=<SumBackward0>)
    """
    