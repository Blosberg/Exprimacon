from PIL import Image as img
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import tsne
import umap

from img_manip_funcs import *

# get the raw 100x100 pixel image
#im = img.open("mpi_logo_reduced_100x100.jpg")
# im = img.open("MPI2.png")
im = img.open("imgs/MPI3_400x400.png")

im_rgb = im.convert('RGB')
# test, example:
# r, g, b = im_rgb.getpixel((50, 50))
im_rgbarr = np.array(im_rgb)
Nxpix, Nypix, ncols = im_rgbarr.shape

# catch all pixels that are white, or off-quite:
COLSAT=255
BTHRESH=5

ispixblack = ( ((im_rgbarr[:,:,0] < BTHRESH) & (im_rgbarr[:,:,1] < BTHRESH)) & ( im_rgbarr[:,:,0] < BTHRESH) )

masked_white_arr = im_rgbarr.copy()
masked_white_arr[ispixblack]  = 0
masked_white_arr[~ispixblack] = 1

temp        = np.linspace(0,COLSAT,Nxpix)
grad_vals_x = np.array([ temp[i] for i in range(len(temp)) ] )

# verticl gradient has to be array of arrays:
temp = np.linspace(0,COLSAT,Nypix)
grad_vals_y = [ [ temp[i] ] for i in range(len(temp)) ]


# ==============================

img_horiz_redgrad = np.ndarray(shape=(Nxpix, Nypix, ncols), dtype=float)
img_horiz_redgrad[:,:,1]=0.0
img_horiz_redgrad[:,:,2]=0.0
img_horiz_redgrad[:,:,0]=masked_white_arr[:,:,0]*grad_vals_x

img_vert_greengrad =  np.ndarray(shape=(Nxpix, Nypix, ncols), dtype=float)
img_vert_greengrad[:,:,0]=0
img_vert_greengrad[:,:,2]=0
img_vert_greengrad[:,:,1]=masked_white_arr[:,:,1]*grad_vals_y[::-1]

img_rad_blugrad =  np.ndarray(shape=(Nxpix, Nypix, ncols), dtype=float)
img_rad_blugrad[:,:,0] = 0
img_rad_blugrad[:,:,1] = 0
img_rad_blugrad[:,:,2] = get_radial_mask(img_rad_blugrad.copy()) * masked_white_arr[:,:,2]

Allgrads= np.ndarray(shape=(Nxpix, Nypix, ncols), dtype=float)
Allgrads[:,:,0]=img_horiz_redgrad[:,:,0]
Allgrads[:,:,1]=img_vert_greengrad[:,:,1]
Allgrads[:,:,2]=img_rad_blugrad[:,:,2]

# This is the image to be reconstructed:
# vis_arr(Allgrads)


Allgrads_lin=np.reshape(Allgrads, (Nxpix * Nypix, ncols), order='C')
masked_white_arr_lin = np.reshape(masked_white_arr, (Nxpix * Nypix, ncols), order='C')


realness = [ not all(masked_white_arr_lin[p,:]==0) for p in range(masked_white_arr_lin.shape[0]) ]
realpoints_normed = Allgrads_lin[ realness, :] / np.max(Allgrads_lin)

np.random.shuffle(realpoints_normed)

# Y = tsne.tsne(Allgrads_lin, 2, 3, 20.0)
Y = tsne.tsne(realpoints_normed, 2, 3, 20.0)

np.savetxt("./out/tSNE_outdat.tsv", Y, fmt='%.18e', delimiter='\t')

arr2im = Image.fromarray(masked_white_arr)
umap_embedding = umap.UMAP( n_neighbors=50, min_dist=0.01, metric='correlation').fit_transform(realpoints)

umap_embedding.save("umap_embedding_out.tsv")
np.savetxt("/out/umap_out.tsv", umap_embedding, delimiter ="\t")


# linearized such that for any x,y
# Allgrads[x,y,:] == Allgrads_lin[ x*Nypix + y,:]

# pureblack=np.zeros([Nxpix, Nypix, ncols])
# Eventually convert it back to an image
# arr2im = Image.fromarray(img2arr)
#
# #Display it
# arr2im.show()
#
# #Save the image generated from an array
# arr2im.save("array2Image.jpg")
