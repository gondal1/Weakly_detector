import skimage.io
import skimage.transform
#import ipdb

import numpy as np
import tensorflow as tf

def load_image( path ):
    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    img /= 255.

    #short_edge = min( img.shape[:2] )
    #yy = int((img.shape[0] - short_edge) / 2)
    #xx = int((img.shape[1] - short_edge) / 2)
    #crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    #resized_img = skimage.transform.resize( crop_img, [224,224] )
    resized_img = skimage.transform.resize( img, [224,224] )
    return resized_img

def crop_image(im):
    
    
    
    '''
    for r in range(im.shape[0]):
        if np.any(im[r,:,:]>1):
            rowStart = r
            break
    for r in range(im.shape[0]-1,0,-1):
        if np.any(im[r,:,:]>1):
            rowStop = r
            break
    for c in range(im.shape[1]):
        if np.any(im[:,c,:]>1):
            colStart = c
            break
    for c in range(im.shape[1]-1,0,-1):
        if np.any(im[:,c,:]>1):
            colStop = c
            break
        
    im_crop = im[rowStart:rowStop+1,colStart:colStop+1,:]
    return scipy.misc.imresize(im_crop, (448,448,3)) 
    '''
def augment(image):

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.05)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.4, upper=1.0)

    # Subtract off the mean and divide by the variance of the pixels.
    #distorted_image = tf.image.per_image_whitening(distorted_image)
    distorted_image = tf.image.central_crop(distorted_image, 0.95)
    
    return distorted_image