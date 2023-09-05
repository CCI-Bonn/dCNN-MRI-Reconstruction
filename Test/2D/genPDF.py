import numpy as np
import matplotlib.pyplot as plt
def genPDF(imSize,p,pctg,distType = 2,radius = 0,disp = 0):
    
    minval = 0
    maxval = 1
    val = 0.5
    imSize = np.asarray(imSize)
    if np.size(imSize) ==1:
        imSize = np.append(imSize, 1)
    
    sx = imSize[0]
    sy = imSize[1]
    PCTG = np.floor(pctg*sx*sy)

    if (imSize==1).sum() == 0: # it is a 2D map
        x1 = np.linspace(-1,1,sx)
        y1 = np.linspace(-1,1,sy)
        x,y = np.meshgrid(y1,x1)
    
        if distType == 1:
            r = np.maximum(abs(x),abs(y))
        else :
            r = np.sqrt(x**2 + y**2)
            r = r/np.max(r)
    else:

        r = abs(np.linspace(-1,1,max(sx,sy)))
    idx = np.where(r < radius)
    pdf = (1-r)**p
    pdf[idx] = 1
    
    if np.floor(np.sum(pdf)) > PCTG:
        print("error: infeasible without undersampling dc, increase p")
        return

    while(1) :
        val = minval/2 + maxval/2
        pdf = (1-r)**p + val
        pdf[np.where(pdf>1)] = 1
        pdf[idx] = 1 
        N = np.floor(np.sum(pdf))
        if N > PCTG:
            maxval = val
        elif N < PCTG:
            minval = val
        elif N == PCTG:
            break

    pdf = np.asarray(pdf)    
    if disp:
        plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, 1))
        plt.clf()
        plt.subplot(211)
        plot(pdf)
        plt.axis('off')
        plt.title('pdf')
        if (imSize==1).sum() == 0:
            plt.subplot(212)
            tmp = int(np.shape(pdf)[0]/2)
            plt.plot(pdf[tmp,:])
            plt.ylim(0.4,1)
        else:
            plt.subplot(212)
            plt.plot(pdf)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
        plt.show()

    return pdf


def genSampling(pdf,iter,tol):
    
    pdf[np.where(pdf >1)] = 1
    K = np.sum(pdf)
    minIntr = 1e99
    minIntrVec = np.zeros(pdf.shape)
    stat = []
    for n in range(iter):
        tmp = np.zeros(pdf.shape)
        while abs(np.sum(tmp)-K) > tol:
            tmp = np.random.rand(*pdf.shape) < pdf

        TMP = np.fft.fft2(tmp/pdf)
        if np.max(abs(TMP.flatten('F')[1:])) < minIntr:
            minIntr = np.max(abs(TMP.flatten('F')[1:]))
            minIntrVec = tmp
        stat.append(np.max(abs(TMP.flatten('F')[1:])))

    actpctg = np.sum(minIntrVec)/np.prod(np.shape(minIntrVec));
    minIntrVec = np.fft.ifftshift(minIntrVec)
    return minIntrVec


def give_mask(size,R):
    if R > 12:
        pdf = genPDF([size[0],size[1]],6,1/R,2,0,0);
    else:
        pdf = genPDF([size[0],size[1]],5,1/R,2,0,0);
    # print(pdf.shape)
    mask = genSampling(pdf,2,5)
    mask[0,0] = True
    mask3D = np.repeat(mask[:,:,np.newaxis], size[2], axis=2)
    #mask3D = mask3D.transpose((2,0,1))
    return mask3D

if __name__ == '__main__':
    pdf = genPDF([128,128,2],5,1/5,2,0,0);
    mask = genSampling(pdf,2,5)
    print(mask.shape)
    mask3D = np.repeat(mask[:,:,np.newaxis], 10, axis=2)
    mask3D = mask3D.transpose((2,0,1))
    print('Undersampling rates is %f'% (np.size(mask3D)/np.sum(mask3D)))
    print(mask[0,0])
    print(mask3D.shape)
    plt.subplot(131)
    plt.imshow(mask3D[4,:,:],cmap=plt.cm.gray, clim=(0.0, 1))
    plt.subplot(132)
    plt.imshow(mask3D[2,:,:],cmap=plt.cm.gray, clim=(0.0, 1))
    plt.subplot(133)
    plt.imshow(mask3D[:,:,50],cmap=plt.cm.gray, clim=(0.0, 1))
    plt.show()




