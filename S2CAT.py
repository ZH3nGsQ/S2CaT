import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from thop import profile
from thop import clever_format


class DSCNN(nn.Module):
    def __init__(self ,channels, band_reduce):
        super().__init__()
        # PU=49,IP=97,SA=99,HS=69,HR=85,TR=29
        self.dwconv1 = nn.Conv3d(channels,channels,kernel_size=(1,1,band_reduce))
        self.conv1 = nn.Conv3d(channels,channels,kernel_size=1)
        self.conv2 = nn.Conv3d(channels,channels,kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(channels,channels,kernel_size=3, padding=1, dilation=1)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        self.net = nn.Sequential(nn.BatchNorm3d(channels),
                                 nn.ReLU6(),
                                )

    def forward(self, x):
        x_ = self.dwconv1(x)
        x1 = self.conv1(x_)
        x2 = self.conv2(x_)
        x3 = self.conv3(x_)
        x_ = x1 + x2 + x3
        x_ = self.pwconv(x_)
        x = x + self.net(x_)
        return x

class SpatialOperation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.block(x)

class Attention(nn.Module):
    def __init__(self,heads,patch,drop):
        super().__init__()
        self.heads = heads
        self.patch=patch
        self.scale = patch**-1
        self.reorganize = nn.Sequential(nn.Conv3d(1,3*heads,
                                                    kernel_size=(3,3,1),
                                                    padding=(1,1,0),
                                                    bias=False),
                                          Rearrange('b h x y s -> b s (h x y)'),
                                          nn.Dropout(drop))
        self.oper_q = nn.Sequential(
            SpatialOperation(heads),
            ChannelOperation(heads),
        )
        self.down_k = nn.Conv2d(self.heads,self.heads,
                                  kernel_size=(3,1),padding=(1,0),stride=(4,1),
                                  groups=self.heads,bias=False)
        self.down_v = nn.Conv2d(self.heads,self.heads,
                                  kernel_size=(3,1),padding=(1,0),stride=(4,1),
                                  groups=self.heads,bias=False)
        self.out = nn.Sequential(nn.Conv3d(in_channels=heads,
                                                out_channels=1,
                                                kernel_size=(3,3,1),
                                                padding=(1,1,0),bias=False),
                                      nn.Dropout(drop),
                                      Rearrange('b c x y s-> b c s x y'),
                                      nn.LayerNorm((patch,patch)),
                                      Rearrange('b c s x y->b c x y s')
                                      )
    def forward(self,x):
        qkv = self.reorganize(x)  # 4,49,486
        qkv = qkv.chunk(3,dim=-1)   # 4,49,162
        q,k,v = map(lambda a: rearrange(a,'b s (h d) -> b h s d',h=self.heads),
                    qkv)    # 4，2，49，81
        q = self.oper_q(q)
        k = self.down_k(k)    # 4，2，13，81
        dots = torch.einsum('bhid,bhjd->bhij',q,k) * self.scale    # 4，2，49，13
        attn = dots.softmax(dim=-1) # 4，2，49，13
        v = self.down_v(v)    # 4，2，13，81
        out = torch.einsum('bhij,bhjd->bhid',attn,v)    # 4，2，49，81
        out = rearrange(out,'b c s (x y) -> b c x y s ', x=self.patch,y=self.patch)  # 4，2，9，9，49
        out = self.out(out)    # 4，1，9，9，49
        return out

class DEformer(nn.Module):
    def __init__(self,heads,patch,drop):
        super().__init__()
        self.attention = Attention(heads,patch,drop)
        self.ffn = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=1,
                                           kernel_size=(3,3,1),
                                           padding=(1,1,0),
                                           bias=False),
                                 nn.ReLU6(),
                                 nn.Dropout(drop)
                                 )
    def forward(self,x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x
    
class CaTfusion(nn.Module):
    def __init__(self,channels,patch,heads,drop,fc_dim,band_reduce):
        super().__init__()
        self.dscnn = DSCNN(channels, band_reduce)
        self.deformer = nn.Sequential(nn.Conv3d(channels,1,
                                                     kernel_size=(1,1,7),
                                                     padding=(0,0,3),
                                                     stride=(1,1,1)),
                                           DEformer(heads,patch,drop)
                                           )
        self.local = nn.Sequential(nn.Conv3d(channels,channels,
                                                 kernel_size=(3,3,
                                                              band_reduce),
                                                 padding=(1,1,0),
                                                 groups=channels),
                                       nn.BatchNorm3d(channels), 
                                       nn.ReLU6()
                                       )
        self.glob = nn.Sequential(nn.Conv3d(1,channels,
                                                  kernel_size=(3,3,
                                                               band_reduce),
                                                  padding=(1,1,0)),
                                        nn.BatchNorm3d(channels),
                                        nn.ReLU6()
                                        )
        self.out = nn.Sequential(nn.Conv3d(2*channels ,fc_dim,kernel_size=1),
                                nn.BatchNorm3d(fc_dim),
                                nn.ReLU6()
                                )


    def forward(self,x):
        x_cnn = self.dscnn(x)  # 4,16,9,9,49
        x_te = self.deformer(x)    # 4,1,9,9,49
        cnn_out = self.local(x_cnn)   # 4,16,9,9,1
        te_out = self.glob(x_te)  # 4,16,9,9,1
        out = self.out(torch.cat((cnn_out,te_out),dim=1))
        # out = self.out(cnn_out)
        return out

class SSfusion(nn.Module):
    def __init__(self,channels, band_reduce):
        super().__init__()
        self.channels = channels
        self.c = channels // 4
        self.spectral1 = nn.Sequential(nn.Conv3d(self.c,self.c, kernel_size=(1,1,3), padding=(0,0,1), groups=self.c),
                                                 nn.BatchNorm3d(self.c),    # [3,1],[1,0],[9,4]
                                                 nn.ReLU6()
                                                 )
        self.spectral2 = nn.Sequential(nn.Conv3d(self.c,self.c, kernel_size=(1,1,7), padding=(0,0,3), groups=self.c),
                                                 nn.BatchNorm3d(self.c),    # [7,3],[3,1],[13,6]
                                                 nn.ReLU6()
                                                 )
        self.spectral3 = nn.Sequential(nn.Conv3d(self.c,self.c, kernel_size=(1,1,11), padding=(0,0,5), groups=self.c),
                                                 nn.BatchNorm3d(self.c),    # [11,5],[5,2]
                                                 nn.ReLU6()
                                                 )
        self.spectral4 = nn.Sequential(nn.Conv3d(self.c,self.c, kernel_size=(1,1,15), padding=(0,0,7), groups=self.c),
                                                 nn.BatchNorm3d(self.c),    # [15,7],[7,3]
                                                 nn.ReLU6()
                                                 )
        # self.spectral5 = nn.Sequential(
        #     nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 9), padding=(0, 0, 4), groups=self.c),
        #     nn.BatchNorm3d(self.c),  # [15,7],[7,3]
        #     nn.ReLU6()
        #     )
        # self.spectral6 = nn.Sequential(
        #     nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 11), padding=(0, 0, 5), groups=self.c),
        #     nn.BatchNorm3d(self.c),  # [15,7],[7,3]
        #     nn.ReLU6()
        #     )
        # self.spectral7 = nn.Sequential(
        #     nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 13), padding=(0, 0, 6), groups=self.c),
        #     nn.BatchNorm3d(self.c),  # [15,7],[7,3]
        #     nn.ReLU6()
        #     )
        # self.spectral8 = nn.Sequential(
        #     nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 15), padding=(0, 0, 7), groups=self.c),
        #     nn.BatchNorm3d(self.c),  # [15,7],[7,3]
        #     nn.ReLU6()
        #     )
        self.spatial1 = nn.Sequential(nn.Conv3d(channels, 8, kernel_size=(1, 1, band_reduce), stride=(1, 1, 1)),
                                      nn.BatchNorm3d(8, eps=0.001, momentum=0.1, affine=True),
                                      nn.Mish())
        self.spatial2 = nn.Conv3d(8, 4, padding=(1, 1, 0),
                                  kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(12, eps=0.001, momentum=0.1, affine=True),
            nn.Mish()
        )
        self.spatial3 = nn.Conv3d(12, 4, padding=(1, 1, 0),
                                  kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(16, eps=0.001, momentum=0.1, affine=True),
            nn.Mish()
        )
     
    def forward(self,x):
        x1 = self.spectral1(x[:,0:self.c,:])
        x2 = self.spectral2(x[:,self.c:2*self.c,:])
        x3 = self.spectral3(x[:,2*self.c:3*self.c,:])
        x4 = self.spectral4(x[:,3*self.c:,:])
        gmspe = torch.cat((x1,x2,x3,x4),dim=1)
        y1 = self.spatial1(x)
        y2 = self.spatial2(y1)
        y3 = torch.cat((y1, y2), dim=1)
        y3 = self.batch_norm2(y3)
        y4 = self.spatial3(y3)
        y5 = torch.cat((y1, y2, y4), dim=1)
        caspa = self.batch_norm3(y5)
        output = torch.mul(gmspe, caspa)
        return output

class S2CAT(nn.Module):
    def __init__(self,channels=16,patch=9,bands=200,num_class=16,
                 fc_dim=16,heads=2,drop=0.1):
        super().__init__()
        self.band_reduce = (bands - 7) // 2 + 1
        self.stem = nn.Conv3d(1,channels,kernel_size=(1,1,7),
                                            padding=0,stride=(1,1,2))
        self.ssfusion = SSfusion(channels, self.band_reduce)
        
        self.catfusion = CaTfusion(channels,patch,heads,drop,fc_dim,self.band_reduce)
       
        self.fc = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                nn.Flatten(),
                                nn.Linear(fc_dim, num_class)
                                )


                                
    def forward(self,x):
        # x.shape = [batch_size,1,patch_size,patch_size,spectral_bands]
        # b,_,_,_,_ = x.shape
        x = self.stem(x)
        x = self.ssfusion(x)
        feature = self.catfusion(x)
        return self.fc(feature)


if __name__ == '__main__':
    model = S2CAT(bands=200,num_class=16)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    input = torch.randn(4, 1, 9, 9, 200).cuda()
    y = model(input)
    print(y.shape)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"flops:{flops}, params:{params}")