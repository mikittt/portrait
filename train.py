import numpy as np
import pickle
import os
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as F
from PIL import Image
import glob
import time
from tqdm import tqdm

from model import smallUnet as net
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
    #X_8=[]
    #X_16=[]
    #X_32=[]
    #X_64=[]
    #X_128=[]  
    """
    for path in tqdm(glob.glob(content_path+"*.jpg")):
        image = Image.open(path).convert('RGB')
        w,h = image.size
        X_8.append(np.array(image.resize((8, 8*h//w), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
        X_16.append(np.array(image.resize((16, 2*(8*h//w)), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
        X_32.append(np.array(image.resize((32, 4*(8*h//w)), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
        X_64.append(np.array(image.resize((64, 8*(8*h//w)), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
        X_128.append(np.array(image.resize((128, 16*(8*h//w)), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
    np.save("/data/unagi0/xenogu/various_size_faces2/X_8.npy",np.array(X_8))
    np.save("/data/unagi0/xenogu/various_size_faces2/X_16.npy",np.array(X_16))
    np.save("/data/unagi0/xenogu/various_size_faces2/X_32.npy",np.array(X_32))
    np.save("/data/unagi0/xenogu/various_size_faces2/X_64.npy",np.array(X_64))
    np.save("/data/unagi0/xenogu/various_size_faces2/X_128.npy",np.array(X_128))
      
    X.append(X_8) 
    X.append(X_16)
    X.append(X_32)
    X.append(X_64)
    X.append(X_128)   
    """
    #X_8  =np.load("/data/unagi0/xenogu/various_size_faces2/X_8.npy")
    #X_16 =np.load("/data/unagi0/xenogu/various_size_faces2/X_16.npy")
    #X_32 =np.load("/data/unagi0/xenogu/various_size_faces2/X_32.npy")
    X_64 =np.load("/data/unagi0/xenogu/various_size_faces2/X_64.npy")
    X_128=np.load("/data/unagi0/xenogu/various_size_faces2/X_128.npy")
    
    #X.append(X_8) 
    #X.append(X_16)
    #X.append(X_32)
    X.append(X_64)
    X.append(X_128)   
    
    style = []
    """
    path = glob.glob(style_path+"*.jpg")[0]
    image = Image.open(path).convert('RGB')
    width, height = image.size
    target_height = (int(round(float(height * target_width) / width))//2)*2
    style.append(np.array(image.resize((target_width, target_height), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
    style.append(np.array(image.resize((target_width//2, target_height//2), Image.ANTIALIAS))[:,:,::-1].transpose(2,0,1))
    """
    return X,style




def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=xp.float32), volatile=x.volatile)
    ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=xp.float32), volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)

def augment(x_train, bs):
    size4 = (x_train[-2].shape[2])//3
    size5 = (x_train[-1].shape[2])//3
    inds = np.random.randint(0,len(x_train[-1]),(bs-1)*3)
    ind = np.random.randint(len(x_train[-1]))
    x4_1 = x_train[-2][inds[:(bs-1)], :, :size4, :]
    x4_2 = x_train[-2][inds[(bs-1):(bs-1)*2], :, :size4, :]
    x4_3 = x_train[-2][inds[(bs-1)*2:], :, size4*2:, :]
    x5_1 = x_train[-1][inds[:(bs-1)], :, :size5, :]
    x5_2 = x_train[-1][inds[(bs-1):(bs-1)*2], :, :size5, :]
    x5_3 = x_train[-1][inds[(bs-1)*2:], :, size5*2:, :]
    x4_ = np.concatenate([x4_3, x4_2, x4_1], axis=2)
    x5_ = np.concatenate([x5_3, x5_2, x5_1], axis=2)
    x4 = np.concatenate([x_train[-2][ind][np.newaxis,:], x4_], axis=0)
    x5 = np.concatenate([x_train[-1][ind][np.newaxis,:], x5_], axis=0)
    x4 = xp.array(x4,dtype=xp.float32)
    x5 = xp.array(x5,dtype=xp.float32)
    return x4,x5,x4/127.5-1,x5/127.5-1


vgg=VGG19()
original_vgg19=VGG19()
serializers.load_hdf5("/home/mil/tanaka/seminar/portrait/fast_portrait/vgg19.model",original_vgg19)

vgg=conv_setup(original_vgg19,vgg)
del original_vgg19

cnn=net()
serializers.load_hdf5("18400sunet_2926200769431403.model",cnn)

X,style=load_data(content_path="/data/unagi0/xenogu/celeb160x120/",style_path="/home/mil/tanaka/seminar/portrait/fast_portrait/data/style/",target_width=128)
print("succesfully data loaded!")

#X_train=[]
#X_test=[]
#for i in range(len(X)):
#    X_train.append(X[i][:-10])
#    X_test.append(X[i][-10:])
#del X
X_train=X
optimizer=optimizers.Adam(alpha=0.01)
optimizer.setup(cnn)

cnn.to_gpu()
vgg.to_gpu()
print("model to gpu")

xp=cnn.xp

N=len(X_train[0])
batch_size=9
kernel=3

c_t=3000
#alpha=20000.
gamma=1e-5
n_epoch=10000
save_model_interval=1
save_image_interval=270

"""
style_large=Variable(xp.array(style[0],dtype=xp.float32)[xp.newaxis,:,:,:],volatile=True)
style_small=Variable(xp.array(style[1],dtype=xp.float32)[xp.newaxis,:,:,:],volatile=True)
style_large-=xp.array([[[[104]],[[117]],[[124]]]])
style_small-=xp.array([[[[104]],[[117]],[[124]]]])
style_feature=vgg(style_large)
for name in ["3_1","4_1"]:
    patch_norm=xp.array([cuda.to_cpu(style_feature[name][0,:,j:j+kernel,i:i+kernel].data/xp.linalg.norm(style_feature[name][0,:,j:j+kernel,i:i+kernel].data)) for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    patch=xp.array([cuda.to_cpu(style_feature[name][0,:,j:j+kernel,i:i+kernel].data) for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    np.save("data/style/style_patch_norm_large"+name+".npy",cuda.to_cpu(patch_norm))
    np.save("data/style/style_patch_large"+name+".npy",cuda.to_cpu(patch))
    

style_feature=vgg(style_small)
for name in ["3_1","4_1"]:
    patch_norm=xp.array([cuda.to_cpu(style_feature[name][0,:,j:j+kernel,i:i+kernel].data/xp.linalg.norm(style_feature[name][0,:,j:j+kernel,i:i+kernel].data)) for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    patch=xp.array([cuda.to_cpu(style_feature[name][0,:,j:j+kernel,i:i+kernel].data) for j in range(style_feature[name].shape[2]-kernel+1) for i in range(style_feature[name].shape[3]-kernel+1)],dtype=xp.float32)
    
    np.save("data/style/style_patch_norm_small"+name+".npy",cuda.to_cpu(patch_norm))
    np.save("data/style/style_patch_small"+name+".npy",cuda.to_cpu(patch))
"""

style_patch_norm=[xp.concatenate((xp.array(np.load("data/style/style_patch_norm_large"+name+".npy"),xp.float32),xp.array(np.load("data/style/style_patch_norm_large"+name+".npy"),xp.float32)[:,:,:,::-1]),axis=0) for name in ["3_1","4_1"]]
style_patch=[xp.concatenate((xp.array(np.load("data/style/style_patch_large"+name+".npy"),xp.float32),xp.array(np.load("data/style/style_patch_large"+name+".npy"),xp.float32)),axis=0) for name in ["3_1","4_1"]]

style_patch_norm_small=[xp.concatenate((xp.array(np.load("data/style/style_patch_norm_small"+name+".npy"),xp.float32),xp.array(np.load("data/style/style_patch_norm_small"+name+".npy"),xp.float32)[:,:,:,::-1]),axis=0) for name in ["3_1","4_1"]]
style_patch_small=[xp.concatenate((xp.array(np.load("data/style/style_patch_small"+name+".npy"),xp.float32),xp.array(np.load("data/style/style_patch_small"+name+".npy"),xp.float32)),axis=0) for name in ["3_1","4_1"]]

for epoch in range(1,n_epoch+1):
    print(("epoch",epoch))
    perm=np.random.permutation(N)
    for i in range(0,N,batch_size):
        print((i,N,batch_size))
        cnn.zerograds()
        vgg.zerograds()

        #x2=xp.array(X_train[1][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        #x3=xp.array(X_train[2][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        #x4=xp.array(X_train[-2][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        #x5=xp.array(X_train[-1][perm[i:i+batch_size]],dtype=xp.float32)/127.5-1.
        x4, x5, x4n, x5n = augment(X_train, batch_size)
        
        swap_X,swap_X_small=cnn(x5n)
        #contents=Variable(xp.array(X_train[-1][perm[i:i+batch_size]],dtype=xp.float32),volatile=True)
        #contents_small=Variable(xp.array(X_train[-2][perm[i:i+batch_size]],dtype=xp.float32),volatile=True)
        contents=Variable(xp.array(x5,dtype=xp.float32),volatile=True)
        contents_small=Variable(xp.array(x4,dtype=xp.float32),volatile=True)
        swap_X-=xp.array([[[[104]],[[117]],[[124]]]])
        contents-=xp.array([[[[104]],[[117]],[[124]]]])
        swap_X_small-=xp.array([[[[104]],[[117]],[[124]]]])
        contents_small-=xp.array([[[[104]],[[117]],[[124]]]])
        
        swap_feature=vgg(swap_X)
        swap_feature_small=vgg(swap_X_small)
        content_feature=vgg(contents)
        content_feature_small=vgg(contents_small)
        ## content loss
        L_content=F.mean_squared_error(Variable(content_feature["4_2"].data), swap_feature["4_2"])
        L_content+=F.mean_squared_error(Variable(content_feature_small["4_2"].data), swap_feature_small["4_2"])
        #L_var = F.mean_squared_error(Variable(content_feature["4_2"].data), swap_feature["4_2"])-F.square(F.mean_absolute_error(Variable(content_feature["4_2"].data), swap_feature["4_2"]))
        #alpha = alpha + L_content.data*30
        #print(alpha)
        ## style loss
        L_style=0
        for s,name in enumerate(["3_1","4_1"]):
            L_style+=cnn.local_patch(swap_feature[name],Variable(style_patch[s],volatile=True),Variable(style_patch_norm[s],volatile=True))
        for s,name in enumerate(["3_1","4_1"]):
            L_style+=cnn.local_patch(swap_feature_small[name],Variable(style_patch_small[s],volatile=True),Variable(style_patch_norm_small[s],volatile=True))
        L_style/=2
        ## total variation loss
        L_tv=total_variation(swap_X)
        L=8*L_content+L_style+gamma*L_tv#+10*L_var
        L.backward()
        optimizer.update()


        if i%9999==0 and i>0:
            serializers.save_hdf5('sunet_{}.model'.format(str(L.data/N).replace('.','')), cnn)
            
        if i%save_image_interval==0:
            if batch_size!=9:
                X = xp.transpose(swap_X.data[0]+xp.array([[[104]],[[117]],[[124]]]), (1,2,0))
                Image.fromarray(np.clip(cuda.to_cpu(X)[:,:,::-1],0,255).astype(np.uint8)).save("out/portrait"+str(epoch)+"_"+".jpg")
            else:
                image=[]
                w = swap_X.shape[3]
                for j in range(3):
                    X = np.transpose(cuda.to_cpu(swap_X.data[j*3:j*3+3])+np.array([[[104]],[[117]],[[124]]]), (0,2,3,1)).reshape(-1,w,3)
                    image.append(X)
                    X = np.transpose(cuda.to_cpu(contents.data[j*3:j*3+3])+np.array([[[104]],[[117]],[[124]]]), (0,2,3,1)).reshape(-1,w,3)
                    image.append(X)
                image = np.concatenate(image,axis=1)
                Image.fromarray(np.clip(image[:,:,::-1],0,255).astype(np.uint8)).save("out/portrait"+str(epoch)+"_"+".jpg")
        print(("content loss={} style loss={} tv loss={}".format(L_content.data,L_style.data,L_tv.data)))
        
        
    #with open("log.txt","w") as f:
    #    f.write("content loss={} style loss={} tv loss={}".format(sum_lc/N,sum_ls/N,sum_lt/N)+str("\n"))

    if epoch%save_model_interval==0:
        serializers.save_hdf5('sunet_{}.model'.format(str(L.data/N).replace('.','')), cnn)
        
