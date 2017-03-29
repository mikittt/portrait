import numpy as np
import pickle
import os
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
from PIL import Image
import glob
import time

from model import FaceSwapNet
from model import VGG19

def conv_setup(ORIGINAL_VGG,VGG):
    VGG.conv1_1 = ORIGINAL_VGG.conv1_1
    VGG.conv1_2 = ORIGINAL_VGG.conv1_2
    VGG.conv2_1 = ORIGINAL_VGG.conv2_1
    VGG.conv2_2 = ORIGINAL_VGG.conv2_2
    VGG.conv3_1 = ORIGINAL_VGG.conv3_1
    VGG.conv3_2 = ORIGINAL_VGG.conv3_2
    VGG.conv3_3 = ORIGINAL_VGG.conv3_3
    VGG.conv4_1 = ORIGINAL_VGG.conv4_1
    VGG.conv4_2 = ORIGINAL_VGG.conv4_2
    """
    VGG.conv4_3 = ORIGINAL_VGG.conv4_3
    VGG.conv5_1 = ORIGINAL_VGG.conv5_1
    VGG.conv5_2 = ORIGINAL_VGG.conv5_2
    VGG.conv5_3 = ORIGINAL_VGG.conv5_3
    VGG.fc6=ORIGINAL_VGG.fc6
    VGG.fc7=ORIGINAL_VGG.fc7
    """
    return VGG
    
def load_data(content_path, style_path, target_width):
    X=[]
    for size in [8,16,32,64,128]:
        
        X_tmp=[]
        """
        for path in glob.glob(content_path+"*.jpg"):
            image = Image.open(path).convert('RGB')
            X_tmp.append(np.array(image.resize((size, size), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
        np.save("X_"+str(size)+".npy",np.array(X_tmp))
        """
        
        X_tmp=np.load("X_"+str(size)+".npy")
        X.append(np.array(X_tmp))
        print("size:{} {}loaded".format(size,len(X_tmp)))
        
    style=[]
    for path in glob.glob(style_path+"*"):
        image = Image.open(path).convert('RGB')
        width, height = image.size
        target_height = int(round(float(height * target_width) / width))
        style.append(np.array(image.resize((target_width, target_height), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
    
    style=np.array(style)
    
    return X,style



def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=xp.float32), volatile=x.volatile)
    ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=xp.float32), volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)




vgg=VGG19()
original_vgg19=VGG19()
serializers.load_hdf5("vgg19.model",original_vgg19)

vgg=conv_setup(original_vgg19,vgg)
del original_vgg19

cnn=FaceSwapNet()

X,style=load_data(content_path="data/content/fin_celebA/",style_path="data/style/",target_width=128)
print("succesfully data loaded!")

X_train=[]
X_test=[]
for i in range(len(X)):
    X_train.append(X[i][:-10])
    X_test.append(X[i][-10:])
del X

optimizer=optimizers.Adam(alpha=0.01)
optimizer.setup(cnn)

cnn.to_gpu()
vgg.to_gpu()
print("model to gpu")

xp=cnn.xp

N=len(X_train[0])
batch_size=16
kernel=3
alpha=1.0
beta=0
gamma=1e-5
n_epoch=10000
save_model_interval=1
save_image_interval=400

style_patch=[]
style_patch_norm=[]
"""
style=Variable(xp.array(style,dtype=xp.float32),volatile=True)
style-=xp.array([[[[104]],[[117]],[[124]]]])
style_feature=vgg(style)
for name in ["3_1","4_1"]:
    patch_norm=xp.array([style_feature[name][0,:,j:j+kernel,i:i+kernel].data/xp.linalg.norm(style_feature[name][0,:,j:j+kernel,i:i+kernel].data) for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    patch=xp.array([style_feature[name][0,:,j:j+kernel,i:i+kernel].data for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    np.save("style_patch_norm"+name+".npy",cuda.to_cpu(patch_norm))
    np.save("style_patch"+name+".npy",cuda.to_cpu(patch))

    style_patch.append(patch)
    style_patch_norm.append(patch_norm)
del patch,patch_norm
"""

style_patch_norm=[xp.array(np.load("style_patch_norm"+name+".npy"),xp.float32) for name in ["3_1","4_1"]]
style_patch=[xp.array(np.load("style_patch"+name+".npy"),xp.float32) for name in ["3_1","4_1"]]


for epoch in range(1,n_epoch+1):
    print("epoch",epoch)
    perm=np.random.permutation(N)
    for i in range(0,N,batch_size):
        if beta<0.4:
            beta+=0.0001
        print(i,N,batch_size)
        cnn.zerograds()
        vgg.zerograds()

        x1=xp.array(X_train[0][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        x2=xp.array(X_train[1][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        x3=xp.array(X_train[2][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        x4=xp.array(X_train[3][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        x5=xp.array(X_train[4][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        
        
        swap_X=cnn(x1,x2,x3,x4,x5)
        contents=Variable(xp.array(X_train[-1][perm[i:i+batch_size]],dtype=xp.float32),volatile=True)
        swap_X-=xp.array([[[[104]],[[117]],[[124]]]])
        contents-=xp.array([[[[104]],[[117]],[[124]]]])
        
        swap_feature=vgg(swap_X)
        content_feature=vgg(contents)
        ## content loss
        L_content=F.mean_squared_error(Variable(content_feature["4_2"].data), swap_feature["4_2"])
        ## style loss
        L_style=0
        for s,name in enumerate(["3_1","4_1"]):
            L_style+=cnn.local_patch(swap_feature[name],Variable(style_patch[s],volatile=True),Variable(style_patch_norm[s],volatile=True))
        L_style/=2
        ## total variation loss
        L_tv=total_variation(swap_X)

        L=alpha*L_content+beta*L_style+gamma*L_tv
        L.backward()
        optimizer.update()


        if i%save_image_interval==0:
            for k,X in enumerate(swap_X.data):
                X = xp.transpose(X+xp.array([[[104]],[[117]],[[124]]]), (1,2,0))
                Image.fromarray(cuda.to_cpu(X)[:,:,::-1].astype(np.uint8)).save("out/portrait"+str(epoch)+"_"+str(k)+"_"+str(beta)+".jpg")
        print("content loss={} style loss={} tv loss={}".format(L_content.data/batch_size,L_style.data/batch_size,L_tv.data/batch_size))
    #with open("log.txt","w") as f:
    #    f.write("content loss={} style loss={} tv loss={}".format(sum_lc/N,sum_ls/N,sum_lt/N)+str("\n"))

    if epoch%save_model_interval==0:
        serializers.save_hdf5('PortraitModel_{}.model'.format(str(L.data/N).replace('.','')), cnn)
        
