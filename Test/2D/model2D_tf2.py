"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

@author: haggarwal
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf
import tensorflow.compat.v1 as tfv
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from os.path import expanduser
tfv.disable_v2_behavior()
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)


# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tfv.real(x),tfv.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tfv.complex(x[...,0],x[...,1])

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    W=tfv.get_variable('W',shape=szW,initializer=tfv.keras.initializers.glorot_uniform())
    x = tfv.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn=tfv.layers.batch_normalization(x,training=trainning,fused=True,name='BN',reuse = tfv.AUTO_REUSE)
    # bn = tfv.keras.layers.BatchNormalization(trainable=trainning,fused=True)
    # xbn = bn(x)

    if not(lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.
    """
    # inp = tfv.Print(inp, [tfv.shape(inp)], "message ")
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,2,64)
    szW[nLay]=(3,3,64,2)

    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with tfv.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw= shortcut+nw['c'+str(nLay)]
    return dw


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self,mask,lam):
        with tf.name_scope('Ainit'):
            s=tf.shape(mask)
            self.nrow,self.ncol=s[0],s[1]
            self.pixels=self.nrow*self.ncol
            self.mask=mask
            #self.csm=csm
            self.SF=tfv.complex(tf.sqrt(tfv.cast(self.pixels,tf.float32) ),0.)
            self.lam=lam
            #self.cgIter=cgIter
            #self.tol=tol
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            #coilImages=self.csm*img
            kspace=  tf.signal.fft2d(img)/self.SF
            temp=kspace*self.mask
            coilImgs =tf.signal.ifft2d(temp)*self.SF
            #coilComb= tf.reduce_sum(coilImgs*tf.conj(self.csm),axis=0)
            coilComb=coilImgs+self.lam*img
        return coilComb

def myCG(A,rhs):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs=r2c(rhs)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,10), rTr>1e-10)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=A.myAtA(p)
            alpha = rTr / tfv.cast(tf.reduce_sum(tfv.conj(p)*Ap),tf.float32)
            alpha=tf.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tfv.cast( tf.reduce_sum(tfv.conj(r)*r),tf.float32)
            beta = rTrNew / rTr
            beta=tfv.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tfv.cast( tf.reduce_sum(tfv.conj(r)*r),tf.float32)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)

def getLambda():
    """
    create a shared variable called lambda.
    """
    with tfv.variable_scope(tfv.get_variable_scope(), reuse=tfv.AUTO_REUSE):
        lam = tf.abs(tfv.get_variable(name='lam1', dtype=tf.float32, initializer=.05)) +0.0001
    return lam

def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch
    """
    G=tf.get_default_graph()
    getnext=G.get_operation_by_name('getNext')
    _,_,csm,mask=getnext.outputs
    l=getLambda()
    l2=tf.complex(l,0.)
    def fn(tmp):
        c,m,r=tmp
        Aobj=Aclass(c,m,l2)
        y=myCG(Aobj,r)
        return y
    inp=(csm,mask,rhs)
    rec=tfv.map_fn(fn,inp,dtype=tf.float32,name='mapFn2' )
    return rec

@tf.custom_gradient
def dcManualGradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y=callCG(x)
    def grad(inp):
        out=callCG(inp)
        return out
    return y,grad



def dc(rhs,mask,lam1):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2=tf.complex(lam1,0.)
    def fn(tmp):
        r,m=tmp
        Aobj=Aclass(m,lam2)
        y=myCG(Aobj,r)
        return y
    inp=(rhs,mask)
    rec=tfv.map_fn(fn,inp,name='mapFn',fn_output_signature = tf.float32)
    return rec

def dc_map(inp,training,nLayers):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def fn_dw(tmp):
        y=dw(tmp,training,nLayers)
        return y
    i=(inp)
    rec=tfv.map_fn(fn_dw,i,name='mapFn_dw',fn_output_signature = tf.float32)
    return rec

def makeModel(atb,mask,mu,sig,training,nLayers,K,gradientMethod):
    """
    This is the main function that creates the model.

    """

    # mask = tf.constant(mask,dtype = tf.complex64,name = 'mask')
    out={}
    atb = atb[0,:,:,:]
    mask = mask[0,:,:,:]
    sig = sig[0,:,:,:]
    mu = mu[0,:,:,:]
    # print(atb)
    out['dc0']=c2r(atb)
    with tf.name_scope('myModel'):
        with tfv.variable_scope('Wts',reuse=tfv.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                tmp = c2r((r2c(out['dc'+str(i-1)]) - mu)/sig)
                # print(tmp)
                # tmp2 = dc_map(tmp,training,nLayers)
                tmp2 = dw(tmp,training,nLayers)
                out['dw'+j]= c2r(r2c(tmp2)*sig + mu)
                lam1=getLambda()
#                rhs=out['dc'+str(i-1)] + lam1*out['dw'+j]
                rhs = out['dc0'] + lam1*out['dw'+j]
                if gradientMethod=='AG':
                    out['dc'+j]=dc(rhs,mask,lam1)
                elif gradientMethod=='MG':
                    if training:
                        out['dc'+j]=dcManualGradient(rhs)
                    else:
                        out['dc'+j]=dc(rhs,csm,mask,lam1)
    return out



def main():
    nLayers = 5
    K = 1
    gradientMethod = "AG"
    training = True
    # mask = np.ones([100,100,100])
    atbP = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='atb')
    orgP = tfv.placeholder(tf.float32,shape=(None,None,None,None),name='org')
    mu = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='mu')
    sig = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='sig')
    mask = tfv.placeholder(tf.complex64,shape=(None,None,None,None),name='mask')

    out = makeModel(atbP,mask,mu,sig,True,nLayers,K,gradientMethod)


if __name__ == '__main__':
    main()
