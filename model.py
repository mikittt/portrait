import numpy as np
import math

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer import cuda, optimizers, Variable,link


class InstanceNormalization(link.Link):
    
    def __init__(self, size, eps=2e-5, dtype=np.float32):
        super(InstanceNormalization, self).__init__()
        self.add_param("gamma", size, dtype=dtype)
        self.add_param("beta", size, dtype=dtype)
        self.eps = eps
        
    def __call__(self, x):
        
        xp = cuda.get_array_module(x.data)
        mean = xp.mean(x.data, axis=(2,3), keepdims=True)
        var = xp.var(x.data, axis=(2,3), keepdims=True)

        normalized_x = (x - mean) / xp.sqrt(var + self.eps) 
        
        return self.gamma*normalized_x + self.beta


class Block(chainer.Chain):
    
    def __init__(self, n_in, N):
        super(Block,self).__init__(
            c1 = L.Convolution2D(n_in, N, 3, stride=1, pad=1),
            i1 = L.BatchNormalization(N),
            c2 = L.Convolution2D(N, N, 3, stride=1, pad=1),
            i2 = L.BatchNormalization(N),
            c3 = L.Convolution2D(N, N, 1, stride=1, pad=0),
            i3 = L.BatchNormalization(N)
        )
    
    def __call__(self, x, test=False):
        h = F.relu(self.i1(self.c1(x)))
        h = F.relu(self.i2(self.c2(h)))
        h = F.relu(self.i3(self.c3(h)))
        
        return h

class Bottleneck(chainer.Chain):
    def __init__(self, N, N_h):
        super(Bottleneck,self).__init__(
            c1 = L.Convolution2D(N, N_h, 1, stride=1, pad=0),
            b1 = L.BatchNormalization(N_h),
            c2 = L.Convolution2D(N_h, N_h, 3, stride=1, pad=1),
            b2 = L.BatchNormalization(N_h),
            c3 = L.Convolution2D(N_h, N, 1, stride=1, pad=0),
            b3 = L.BatchNormalization(N)            
        )
    
    def __call__(self, x, test=False):
        h = F.leaky_relu(self.b1(self.c1(x)))
        h = F.leaky_relu(self.b2(self.c2(h)))
        h = self.b3(self.c3(h))
        return h
        
        
class FaceSwapNet(chainer.Chain):
    
    def __init__(self):
        super(FaceSwapNet, self).__init__(
            b1 = Block(3,32),
            
            b2_1 = Block(3,32),
            b2_2 = Block(64,64),
            
            b3_1 = Block(3,32),
            b3_2 = Block(96,96),
            
            b4_1 = Block(3,32),
            b4_2 = Block(128,128),
            
            b5_1 = Block(3,32),
            b5_2 = Block(160,160),
            
            fin_conv = L.Convolution2D(160, 3, 1, stride=1, pad=0)
        )
        
    def __call__(self, x1, x2, x3, x4, x5, test=False):
        
        h1 = self.b1(x1, test=test)
        h1 = F.unpooling_2d(h1, ksize=2, stride=2, pad=0, cover_all=False)
        
        h2 = self.b2_1(x2, test=test)
        h2 = self.b2_2(F.concat([h1, h2]))
        h2 = F.unpooling_2d(h2, ksize=2, stride=2, pad=0, cover_all=False)
        del h1,x1
        
        h3 = self.b3_1(x3, test=test)
        h3 = self.b3_2(F.concat([h2, h3]))
        h3 = F.unpooling_2d(h3, ksize=2, stride=2, pad=0, cover_all=False)
        del h2,x2
        
        h4 = self.b4_1(x4, test=test)
        h4 = self.b4_2(F.concat([h3, h4]))
        h4 = F.unpooling_2d(h4, ksize=2, stride=2, pad=0, cover_all=False)
        del h3,x3             
        
        h5 = self.b5_1(x5, test=test)
        h5 = self.b5_2(F.concat([h4, h5]))
        del h4,x4
        
        h5 = F.sigmoid(self.fin_conv(h5))
        
        return h5*255
    def local_patch(self, content, style_patch, style_patch_norm):
        
        xp = cuda.get_array_module(content.data)
        b,ch,h,w = content.data.shape
        correlation = F.convolution_2d(Variable(content.data,volatile=True), W=style_patch_norm.data, stride=1, pad=0)
        indices = xp.argmax(correlation.data, axis=1)
        nearest_style_patch = style_patch.data.take(indices, axis=0).reshape(b,-1,3*3*ch).transpose(1,0,2).reshape(-1,b*9*ch)
        content = F.convolution_2d(content, W=Variable(xp.identity(ch*3*3,dtype=xp.float32).reshape((ch*3*3,ch,3,3))),stride=1,pad=0).transpose(2,3,0,1).reshape(-1,b*3*3*ch)
        c_norm = content/xp.linalg.norm(content.data,axis=1,keepdims=True)
        style_loss = F.mean_squared_error(content, nearest_style_patch)+F.mean_squared_error(xp.identity(content.shape[0],dtype=xp.float32),F.matmul(c_norm,F.transpose(c_norm)))*40000
        
        return style_loss
    """
    def local_patch(self, content, style_patch, style_patch_norm):
        
        xp = cuda.get_array_module(content.data)
        b,ch,h,w = content.data.shape
        correlation = F.convolution_2d(Variable(content.data,volatile=True), W=style_patch_norm.data, stride=1, pad=0)
        indices = xp.argmax(correlation.data, axis=1)
        nearest_style_patch = style_patch.data.take(indices, axis=0).reshape(b,-1)
        content = F.convolution_2d(content, W=Variable(xp.identity(ch*3*3,dtype=xp.float32).reshape((ch*3*3,ch,3,3))),stride=1,pad=0).transpose(0,2,3,1).reshape(b,-1)
        style_loss = F.mean_squared_error(content, nearest_style_patch)
        return style_loss
        """
class FaceSwapNet2(chainer.Chain):
    
    def __init__(self):
        super(FaceSwapNet2, self).__init__(
            c1 = L.Convolution2D(3, 64, 5, stride=1, pad=2),
            c2 = L.Convolution2D(3, 64, 5, stride=1, pad=2),
            c3 = L.Convolution2D(3, 64, 5, stride=1, pad=2),
            c4 = L.Convolution2D(3, 64, 5, stride=1, pad=2),
            c5 = L.Convolution2D(3, 64, 5, stride=1, pad=2),
            c2_2 = L.Convolution2D(128, 64, 5, stride=1, pad=2),
            c3_2 = L.Convolution2D(128, 64, 5, stride=1, pad=2),
            c4_2 = L.Convolution2D(128, 64, 5, stride=1, pad=2),
            c5_2 = L.Convolution2D(128, 64, 5, stride=1, pad=2),
            b1 = Bottleneck(64,64),
            
            b2_1 = Bottleneck(64,32),
            b2_2 = Bottleneck(64,32),
            b2_3 = Bottleneck(64,32),
            
            b3_1 = Bottleneck(64,32),
            b3_2 = Bottleneck(64,32),
            b3_3 = Bottleneck(64,32),
            
            b4_1 = Bottleneck(64,32),
            b4_2 = Bottleneck(64,32),
            b4_3 = Bottleneck(64,32),
            
            b5_1 = Bottleneck(64,32),
            b5_2 = Bottleneck(64,32),
            b5_3 = Bottleneck(64,32),
            
            fin_conv = L.Convolution2D(64, 3, 1, stride=1, pad=0),      
        )
        
    def __call__(self, x1, x2, x3, x4, x5, test=False):
        h = self.c1(x1)
        h1 =  F.leaky_relu(self.b1(h, test=test)+h)
        h1 = F.unpooling_2d(h1, ksize=2, stride=2, pad=0, cover_all=False)
        
        h = self.c2(x2)
        h = F.leaky_relu(self.b2_1(h, test=test)+h)
        h2 = F.leaky_relu(self.b2_2(self.c2_2(F.concat([h1, h])))+h+h1)
        h2 = F.leaky_relu(self.b2_3(h2, test=test)+h2)
        h2 = F.unpooling_2d(h2, ksize=2, stride=2, pad=0, cover_all=False)
        
        h = self.c3(x3)
        h = F.leaky_relu(self.b3_1(h, test=test)+h)
        h3 = F.leaky_relu(self.b3_2(self.c3_2(F.concat([h2, h])))+h+h2)
        h3 = F.leaky_relu(self.b3_3(h3, test=test)+h3)
        h3 = F.unpooling_2d(h3, ksize=2, stride=2, pad=0, cover_all=False)
        
        h = self.c4(x4)
        h = F.leaky_relu(self.b4_1(h, test=test)+h)
        h4 = F.leaky_relu(self.b4_2(self.c4_2(F.concat([h3, h])))+h+h3)
        h4 = F.leaky_relu(self.b4_3(h4, test=test)+h4)
        h4 = F.unpooling_2d(h4, ksize=2, stride=2, pad=0, cover_all=False)
        
        h = self.c5(x5)
        h = F.leaky_relu(self.b5_1(h, test=test)+h)
        h5 = F.leaky_relu(self.b5_2(self.c4_2(F.concat([h4, h])))+h+h4)
        h5 = F.leaky_relu(self.b5_3(h5, test=test)+h5)
        h5 = F.sigmoid(self.fin_conv(h5))
        
        return h5*255

    def local_patch(self, content, style_patch, style_patch_norm):
        
        xp = cuda.get_array_module(content.data)
        b,ch,h,w = content.data.shape
        correlation = F.convolution_2d(Variable(content.data,volatile=True), W=style_patch_norm.data, stride=1, pad=0)
        indices = xp.argmax(correlation.data, axis=1)
        nearest_style_patch = style_patch.data.take(indices, axis=0).reshape(b,-1,3*3*ch).transpose(1,0,2).reshape(-1,b,9*ch)
        content = F.convolution_2d(content, W=Variable(xp.identity(ch*3*3,dtype=xp.float32).reshape((ch*3*3,ch,3,3))),stride=1,pad=0).transpose(2,3,0,1).reshape(-1,b,9*ch)
        c_norm = (content/xp.linalg.norm(content.data,axis=2,keepdims=True)).reshape(-1,b*9*ch)
        style_loss = F.mean_squared_error(content, nearest_style_patch)+F.mean_squared_error(xp.identity(content.shape[0],dtype=xp.float32)*b,F.matmul(c_norm,F.transpose(c_norm)))*200
        
        return style_loss
    
class VGG19(chainer.Chain):

    def __init__(self):
        super(VGG19, self).__init__(
            conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_4 = L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv4_4 = L.Convolution2D(512, 512, 3, stride=1, pad=1),

            #conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
            #conv5_4 = L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )
        self.mean = np.asarray([104, 117, 124], dtype=np.float32)
        
    def vgg_preprocess(X, input_type="trans"):
        if input_type=="trans":
            X -= np.asarray([[[124]],[[117]],[[104]]], dtype=np.float32)
        elif input_type=="RGB":
            X = np.rollaxis(X[:,:,::-1]-np.asarray([104, 117, 124], dtype=np.float32),2)
        return X


    def __call__(self, x):
        layer_names = ['1_1', '1_2', 'pool', '2_1', '2_2', 'pool', '3_1',
                       '3_2', '3_3', '3_4', 'pool', '4_1', '4_2']#, '4_3', '4_4',
                       #'pool', '5_1', '5_2', '5_3', '5_4']
        layers = {}
        h = x
        for layer_name in layer_names:
            if layer_name == 'pool':
                h = F.max_pooling_2d(h, 2, stride=2)
            else:
                h = F.relu(self['conv' + layer_name](h))
                layers[layer_name] = h
        return layers
