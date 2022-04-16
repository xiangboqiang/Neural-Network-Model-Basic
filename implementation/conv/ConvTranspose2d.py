



# 网上有关nn.ConvTranspose2d资料较少，这里做做实验

import torch
import torch.nn as nn

IC = 2
OC = 2
K  = 2
S  = 2
P  = 0

IH = 3
IW = 3

class MyDeconv():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(MyDeconv, self)
        self.ic = in_channels
        self.oc = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = kernel_size - padding - 1
        self.weight = torch.ones(self.oc, self.ic, self.k, self.k)
    def load(self, weight):
        print(f'w raw shape {self.weight.shape}')
        self.weight = weight
        print(f'w changed shape {self.weight.shape}')
    def __call__(self, x):
        return self.forward(x)
    def forward(self,x):
        IH = x.shape[2] + (self.stride-1)*(x.shape[2]-1)
        IW = x.shape[3] + (self.stride-1)*(x.shape[3]-1)
        OH = (IH + 2*self.padding - self.k)//1 + 1
        OW = (IH + 2*self.padding - self.k)//1 + 1
        y = torch.zeros(1, self.ic, OH, OW)
        ''' upsampling '''
        newx = torch.zeros(1,self.oc,IH,IW)
        for oc in range(self.oc):
            for ih in range(0,IH,self.stride):
                for iw in range(0,IW,self.stride):
                    newx[0,oc,ih,iw] = x[0,oc,ih//self.stride,iw//self.stride]
        x = newx
        ''' padding '''
        pad = nn.ZeroPad2d(self.padding)
        x = pad(x)
        print(f'x is {x}')
        self.weight = torch.rot90(self.weight,2,dims=(2,3))
        for ic in range(self.ic):
            for oc in range(self.oc):
                for r in range(OH):
                    for c in range(OW):
                        # print(f'x is {x[0,oc,r:r+self.k,c:c+self.k]}')
                        # print(f'w is {self.weight[oc,ic,:,:]}')
                        y[0, ic,r,c] += torch.mul(
                            x[0,oc,r:r+self.k,c:c+self.k],
                            self.weight[oc,ic,:,:]
                        ).sum()
        return y

if __name__ == "__main__":
    
    deconv_torch = nn.ConvTranspose2d(
        in_channels=IC, 
        out_channels=OC, 
        kernel_size=K, 
        stride=S, 
        padding = P,
        bias=False)
    
    
    print(deconv_torch)
    print(f'w raw shape is {deconv_torch.weight.shape}')
    deconv_torch.weight = nn.Parameter(torch.ones(IC,OC,K,K).float())
    deconv_torch.weight = nn.Parameter(torch.arange(IC*OC*K*K).view(IC,OC,K,K).float())
    print(f'w changed shape is {deconv_torch.weight.shape}')
    print(deconv_torch.weight)


    x = torch.ones(1,IC,IH,IW)
    x = torch.arange(0,IC*IH*IW).view(1,IC,IH,IW).float()
    print(x)
    print(f'x shape is {x.shape}')
    y = deconv_torch(x)
    print(f'y shape is {y.shape}')
    print(y)


    print('- - '*10)

    deconv_np = MyDeconv(
        in_channels=OC, 
        out_channels=IC, 
        kernel_size=K, 
        stride=S, 
        padding = 0,
        bias=False
        )
    print(f'x shape is {x.shape}')
    deconv_np.load(deconv_torch.weight)
    
    print(f'w shape is {deconv_torch.weight.shape}')
    y = deconv_np(x)
    print(f'y shape is {y.shape}')
    print(y)
    






