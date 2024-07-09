##############################################################################
# This is a copy from https://sites.google.com/site/vivonegemine/code
##############################################################################

import numpy as np
from scipy import ndimage
from math import ceil, log10, floor
import torch.fft as FFT
import torch
import torch.nn.functional as nF

def my_psnr(img, ref):
    p = 0
    ch = ref.shape[-1]
    for i in range(ch):
        p += 10*log10(1/np.mean(np.power(img[:,:,i] - ref[:,:,i], 2)))

    return p/ch

######### operator ######################################################################
def solve_proxconv(Y1, Y2, FB, FBT, alpha, beta):
    # \alpha ||X - Y_1||_F^2 + \beta ||XB - Y_2||_F^2
    ch = Y1.shape[1]
    UP = alpha*FFT.rfft2(Y1) + beta*FFT.rfft2(Y2)*FB
    DOWN = alpha + beta*FB*FBT
    X = FFT.irfft2(UP/DOWN)
    return X

def ST_upsample(Y, sf, s0, Nways):
    # X torch tensor of size [B, C, H, W]
    X = torch.zeros(Nways).type(Y.dtype).to(Y.device)
    X[:,:,s0::sf, s0::sf] = Y.clone()
    return X

def Bfilter(Y, FB):
    X = FFT.irfft2(FFT.rfft2(Y)*FB)
    return X


######### filter ########################################################################
def fir_filter_wind(Hd,w):
    
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    #h=h/np.sum(h)
    
    return h

def gaussian2d (N, std):
    t=np.arange(-(N-1)/2,(N+1)/2)
    #t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d (N, beta):
    t=np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    #t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

def genMTF(ratio, sensor, nbands):
    
    N = 41
        
    if sensor=='QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1' or sensor=='WV4':
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='WV2':
        GNyq = np.asarray([0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27],dtype='float32')
    elif sensor=='WV3':
        GNyq = np.asarray([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315],dtype='float32') 
    else:
        GNyq = 0.3 * np.ones(nbands)
        
    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1/ratio

    h = np.zeros((N,N,nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[ii])))
        H=gaussian2d(N,alpha)
        Hd=H/np.max(H)
        w=kaiser2d(N,0.5)
        h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
        
    return h

def genMTF_pan(ratio, sensor):
    N = 41

    if sensor == 'QB':
        GNyq = 0.15
    elif sensor == 'IKONOS':
        GNyq = 0.17
    elif sensor in ['GeoEye1', 'WV4']:
        GNyq = 0.16
    elif sensor == 'WV2':
        GNyq = 0.11
    elif sensor == 'WV3':
        GNyq = 0.14
    else:
        GNyq = 0.15

    """MTF"""
    fcut = 1 / ratio
    # h = np.zeros((N, N, nbands))
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = np.real(fir_filter_wind(Hd, w))

    return h


def MTF(I_MS,sensor,ratio):
    
    h = genMTF(ratio, sensor,I_MS.shape[2])
    
    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:,:,ii] = ndimage.filters.correlate(I_MS[:,:,ii],h[:,:,ii],mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)

def MTF_sim(Y, k):
    ks = k.shape[-1]
    pd = int((ks-1)/2)
    Y = nF.pad(Y, (pd,pd,pd,pd), mode='replicate')
    X = nF.conv2d(Y, k, groups=k.shape[0])
    return X


def MTF_pan(I_MS, sensor, ratio):
    h = genMTF_pan(ratio, sensor, I_MS.shape[2])

    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:, :, ii] = ndimage.filters.correlate(I_MS[:, :, ii], h, mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)

def interp23(image, ratio):
    if (2**round(np.log2(ratio)) != ratio):
        print("Error: only resize factors of power 2")
        return -1

    r = image.shape[0]
    c = image.shape[1]
    
    if (np.size(image.shape) == 3):      
        b = image.shape[2]
    else:
        b = 1
    
    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        if (b == 1):
            I1LRU = np.zeros(((2**z)*r, (2**z)*c))
        else:
            I1LRU = np.zeros(((2**z)*r, (2**z)*c, b))
            
        if first:
            if (b == 1):
                I1LRU[1:I1LRU.shape[0]:2,1:I1LRU.shape[1]:2]=image
            else:
                I1LRU[1:I1LRU.shape[0]:2,1:I1LRU.shape[1]:2,:]=image
            first = 0
        else:
            if (b == 1):
                I1LRU[0:I1LRU.shape[0]:2,0:I1LRU.shape[1]:2]=image
            else:
                I1LRU[0:I1LRU.shape[0]:2,0:I1LRU.shape[1]:2,:]=image
        
        for ii in range(b):
            if (b == 1):
                t = I1LRU
            else:
                t = I1LRU[:,:,ii]
                
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            if (b == 1):
                I1LRU = t
            else:
                I1LRU[:,:,ii] = t
            
        image = I1LRU
        
    return image


def hist_mapping_simple(lrms, pan):
    '''
    lrms : numpy [h,w,C]
    pan : numpy [H,W]
    '''
    nr, nc, C = lrms.shape
    stdY = np.std(lrms, axis=(0,1), keepdims=True)
    stdP = np.std(pan)
    mP = np.mean(pan)
    mY = np.mean(lrms, axis=(0,1), keepdims=True)
    High_pan = (stdY/stdP)*(np.repeat(np.expand_dims(pan,2), C, axis=2) - mP) + mY
    return High_pan

def torch_hist_mapping_simple(lrms, pan):
    '''
    lrms : [B,C,h,w]
    pan : numpy [B,1,H,W]
    '''
    stdY = torch.std(lrms, dim=(2,3), keepdim=True) 
    stdP = torch.std(pan, dim=(2,3), keepdim=True) 
    mP = torch.mean(pan, dim=(2,3), keepdim=True)
    mY = torch.mean(lrms, dim=(2,3), keepdim=True)
    High_pan = (stdY/stdP)*(pan.repeat(1,lrms.shape[1],1,1) - mP) + mY
    return High_pan

def mappting_block(blrms, bpan):
    """
    blrms : numpy [h,w,C]
    """
    return 0

############### imresize ##########################################################
def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

###### FFT related #####
def p2o(psf, shape): 
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = FFT.rfft2(otf)
    otf = torch.view_as_real(otf)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    otf = torch.view_as_complex(otf)
    return otf

def mconj(X):
    Y = torch.view_as_real(X).clone()
    Y[:,:,:,:,1] = -Y[:,:,:,:,1]
    Y = torch.view_as_complex(Y)
    return Y