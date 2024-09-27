## by Xiangyu Rui ##

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.nn.parameter import Parameter
from model_fusionnet import FusionNet
from scipy.io import loadmat, savemat
from torch.optim import SGD, Adam
import torch.nn.functional as nF
from psutils import *
from torchvision.transforms import Compose
import torch.utils.data as data
import os
from os.path import join 
import random as random
import argparse
import time
from PIL import Image

parser = argparse.ArgumentParser(description = 'PSDip')
parser.add_argument('-sensor', default="QB", type=str)
parser.add_argument('-lam', default=0.1, type=float) # trade off parameter \lambda in Eq. (8)
parser.add_argument('-seed', default=0, type=int)
parser.add_argument('-init', action='store_true') # set true to initialize f_\theta 
args = parser.parse_args()
print(args)

class Param(nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.X = Parameter(data=data.clone())
    
    def forward(self,):
        return self.X

def mypadRepli(x, pd):
    s1,s2,s3,s4 = x.shape
    y = torch.zeros((s1,s2,s3+2*pd,s4+2*pd)).to(x.device).type(x.dtype)
    y[:,:,:pd,:pd] = x[:,:, :1,:1]
    y[:,:,:pd,pd:pd+s4] = x[:,:, :1, :]
    y[:,:,:pd,pd+s4:] = x[:,:, :1, -1:]
    
    y[:,:,pd:pd+s3,:pd] = x[:,:, :, :1]
    y[:,:,pd:pd+s3,pd:pd+s4] = x
    y[:,:,pd:pd+s3,pd+s4:] = x[:,:, :, -1:]

    y[:,:,pd+s3:,:pd] = x[:,:, -1:, :1]
    y[:,:,pd+s3:,pd:pd+s4] = x[:,:, -1:, :]
    y[:,:,pd+s3:,pd+s4:] = x[:,:, -1:, -1:]
    
    return y
    
device = torch.device("cuda")

lam = args.lam
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

sensor = args.sensor
rtype = 'full' # full resolution
savedir = join("./results", sensor+'_'+rtype)
os.makedirs(savedir, exist_ok=True)

imgdir = join(savedir, "img_show") # visualize the intermediate products
os.makedirs(imgdir, exist_ok=True)

saveGdir = join("./results", sensor+'_'+rtype)

channel = {'QB':4}

# bands are used for generating the pseudo color images of intermediate products
bands = {'QB':torch.tensor([1,3,0], dtype=torch.int)}

saveimage = True # save the pseudo color images of intermediate products or not

for dindex in range(20):
    ###### process data ######
    print('======== ',dindex+1,' ========')

    data = loadmat("/blabla/"+sensor+"/full/Test(HxWxC)_"+sensor.lower()+"_data_fr"+str(dindex+1)+".mat") # please change the this line according to your own data dir

    lrms = np.float32(data["ms"]/2047.) # LRMS
    I_MS = np.float32(data["lms"]/2047.) # upsampled LRMS provided by the dataset

    pan = np.float32(data["pan"]/2047.) # PAN
    Tlrms = torch.from_numpy(lrms).permute(2,0,1).unsqueeze(0).to(device)
    Tpan = torch.from_numpy(pan).unsqueeze(0).unsqueeze(0).to(device)

    imgdir_ = join(imgdir, str(dindex+1))
    os.makedirs(imgdir_, exist_ok=True)      

    del data

    lh, lw, ch = lrms.shape
    hh, hw = pan.shape
    ratio = int(hh/lh)

    lrms_up = np.float32(imresize(lrms, scalar_scale=ratio, method='bicubic')) # upsample LRMS
    Tlrms_up = torch.from_numpy(lrms_up).permute(2,0,1).unsqueeze(0).to(device)

    P3D = torch_hist_mapping_simple(Tlrms, Tpan).to(device) + 1e-2 # extended PAN \hat{P} in Eq. (4)

    ###### kernel ######
    h = genMTF(ratio, sensor, ch)
    h = torch.from_numpy(np.float32(h)).permute(2,0,1).unsqueeze(1).to(device)  # [C, 1, ks, ks]
    hp = genMTF_pan(ratio, sensor)
    hp = torch.from_numpy(np.float32(hp)).unsqueeze(0).unsqueeze(0).to(device)  # [1 ,1, ks, ks]

    ks = h.shape[-1]
    pd = int((ks - 1)/2)
    # blur operator
    blur = Compose([partial(mypadRepli, pd = pd), partial(nF.conv2d, weight=h, groups=ch)]) 

    s0 = 2
    # downsample operator
    down = lambda x : x[:,:,s0::ratio, s0::ratio]
    
    ###### other data ######
    BP3D = blur(P3D)

    ###### define net ######
    Xopt = Param(Tlrms_up.detach().clone()).cuda()
    optX = SGD(Xopt.parameters(), lr=2)  # 2

    Gnet = FusionNet(channel[sensor]).cuda()  
    optG = Adam(Gnet.parameters(), lr = 1e-3)
    
    G_loss = []
    start_time = time.time()

    ## initializing f_\theta
    if args.init:
        for i in range(8000):
            Gnet.train()

            G = Gnet(Tlrms_up, Tpan)

            loss = torch.norm(Tlrms_up-G*BP3D) # L_{init} in Eq. (13)
            
            optG.zero_grad()
            loss.backward()
            optG.step()
            
            G_loss.append(loss.item())
            
            if saveimage:
                if (i+1)%10 == 0 and (i+1)<200:
                    G = ((G - G.min())/(G.max() - G.min())*255).detach().cpu().squeeze(0)
                    G = torch.index_select(G, dim=0, index=bands[sensor])
                    img = Image.fromarray(np.uint8(G.permute(1,2,0).numpy()), "RGB")
                    img.save(join(imgdir_, 'G_'+str(i+1)+'.png'), )

            if (i+1)%200 == 0:
                if saveimage:
                    G = ((G - G.min())/(G.max() - G.min())*255).detach().cpu().squeeze(0)
                    G = torch.index_select(G, dim=0, index=bands[sensor])
                    img = Image.fromarray(np.uint8(G.permute(1,2,0).numpy()), "RGB")
                    img.save(join(imgdir_, 'G_'+str(i+1)+'.png'))
                with torch.no_grad():
                    G = Gnet(Tlrms_up, Tpan)
                    tX = (G*P3D).detach().squeeze(0).permute(1,2,0).cpu().numpy()

                print(f"Iter {i+1:d}  loss: {loss.item():.3e}")
        
        torch.save({'model':Gnet.state_dict(), 'opt':optG.state_dict()}, join(savedir, str(dindex+1) + "_Gpre.pth"))

    cks = torch.load(join(saveGdir, str(dindex+1) + "_Gpre.pth"))
    Gnet.load_state_dict(cks['model'])

    # G_{init}
    with torch.no_grad():
        G_init = Gnet(Tlrms_up, Tpan).detach().cpu().squeeze(0).permute(1,2,0).numpy()

    all_loss = []

    del optG
    optG = Adam(Gnet.parameters(), lr = 1e-3)

    # iterative updating
    for i in range(3000): 
        Gnet.train()

        ## update X 
        X = Xopt()
        G = Gnet(X.detach().clone(), Tpan.detach()).detach()
        # L_\X(\X, \theta_{t-1}) in Eq. (9)
        loss = torch.sum(torch.pow(down(blur(X)) - Tlrms.detach(), 2)) + lam*torch.sum(torch.pow(X-G*P3D.detach(),2))

        # Algorithm 1 line 2
        optX.zero_grad()
        loss.backward()
        optX.step()

        ## update \theta
        X = Xopt().detach()
        G = Gnet(X, Tpan.detach())
        # L_\theta(\X_t, \theta) in Eq. (11)
        loss = torch.sum(torch.pow(down(blur(X)) - Tlrms.detach(), 2)) + lam*torch.sum(torch.pow(X-G*P3D.detach(),2)) 

        # Algorithm1 line 3
        optG.zero_grad()
        loss.backward()
        optG.step()
        
        all_loss.append(loss.item())

        X = Xopt().detach().squeeze(0).permute(1,2,0).cpu().numpy()
        
        if saveimage:
            if (i+1)%10 == 0 and (i+1)<200:
                X = Xopt().detach().cpu().squeeze(0)*255
                X = torch.index_select(X, dim=0, index=bands[sensor])
                img = Image.fromarray(np.uint8(X.permute(1,2,0).numpy()), "RGB")
                img.save(join(imgdir_, 'X_'+str(i+1)+'.png'))

        if (i+1)%200 == 0:      
            print(f"Iter {i+1:d} loss: {loss.item():.4e}")

            if saveimage:
                G = ((G - G.min())/(G.max() - G.min())*255).detach().cpu().squeeze(0)
                G = torch.index_select(G, dim=0, index=bands[sensor])
                img = Image.fromarray(np.uint8(G.permute(1,2,0).numpy()), "RGB")
                img.save(join(imgdir_, 'G_'+str(i+8001)+'.png'))
                
                X = Xopt().detach().cpu().squeeze(0)*255
                X = torch.index_select(X, dim=0, index=bands[sensor])
                img = Image.fromarray(np.uint8(X.permute(1,2,0).numpy()), "RGB")
                img.save(join(imgdir_, 'X_'+str(i+1)+'.png'))
            
    our_time = time.time() - start_time
    print(f"Time: {our_time:.3f}")
    G_loss = np.array(G_loss)
    all_loss = np.array(all_loss)

    # final output HRMS X
    X = Xopt().detach()

    # final output coefficient G
    G = Gnet(X, Tpan.detach()).detach()

    # save data
    torch.save({'model':Gnet.state_dict()}, join(savedir, str(dindex+1) + "_Gfinal.pth"))
    savemat(join(savedir, str(dindex+1) + ".mat"), {'I_myDIP': X.squeeze(0).permute(1,2,0).cpu().numpy(), 'G': G.squeeze(0).permute(1,2,0).cpu().numpy(), \
                                                    'G_loss':G_loss, 'all_loss':all_loss, 'G_init':G_init, 'Time':our_time})

