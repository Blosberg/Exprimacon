from PIL import Image as img
import numpy as np
import matplotlib.pyplot as plt
import math

def vis_arr(dat_in):
    (Nxpix, Nypix, ncols) = dat_in.shape
    conv2ints=np.ndarray(shape=(Nxpix, Nypix, ncols), dtype="uint8")

    conv2ints[:,:,0] = [ [round(x) for x in arr] for arr in dat_in[:,:,0] ]
    conv2ints[:,:,1] = [ [round(x) for x in arr] for arr in dat_in[:,:,1] ]
    conv2ints[:,:,2] = [ [round(x) for x in arr] for arr in dat_in[:,:,2] ]

    arr2im    = img.fromarray(conv2ints)
    arr2im.show()

# #Save the image generated from an array
# arr2im.save("array2Image.jpg")

def get_radial_mask( arr_in, COLSAT=255, min=0):
    Nx =  arr_in.shape[0]
    Ny =  arr_in.shape[1]
    #
    centre_x = Nx/2
    centre_y = Ny/2
    #
    dist_to_min = math.sqrt( sum( [centre_x**2, centre_y**2] ) )
    #
    #    [ [y+x for x in range(4)] for y in range(5) ]
    #    radmask_out = np.array( [ [ int(COLSAT*( 1 - math.sqrt((y-centre_x)**2 + (x-centre_y)**2)/dist_to_min)) for x in range(Nx)] for y in range(Ny) ] )
    radmask_out = np.array( [ [ COLSAT*( 1 - math.sqrt((y-centre_y)**2 + (x-centre_x)**2)/dist_to_min) for x in range(Nx)] for y in range(Ny) ] )
    #
    return( radmask_out )
