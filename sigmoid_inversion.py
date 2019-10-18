from matplotlib.pyplot import imread
from matplotlib.pyplot import imsave
import numpy as np
from scipy.special import expit


def sig_invert(file_name):

    # get input image as float of rgb
    img_in = imread(file_name)
    img_in = img_in.astype(np.float64)
    
    # width of input image
    width = img_in.shape[1]
    
    # inversion image
    img_inv = 255-img_in
    img_inv = img_inv.astype(np.float64)

    # 2d range from left to right
    mod = np.zeros(img_in.shape[:2])
    mod = mod.astype(np.float64)
    mod[:,] = np.arange(width)/width
    
    # extend to 3d
    rgb_mod = np.repeat(mod[:, :, np.newaxis], 3, axis=2)

    # sig center position to right as %
    pos = 0.55
    
    # transition rate
    rate = 12
    
    # apply sigmoid for transition
    sig = expit((rgb_mod-pos)*rate)

    # transition to inversion by sigmoid
    out_float = img_in*(1-sig) + img_inv*sig
    
    # convert back to rgb format
    out_float = out_float.round()
    output_image = out_float.astype(np.uint8)
    
    # save the file
    out_file = 'sig_invert.jpeg'
    imsave(out_file, output_image)

