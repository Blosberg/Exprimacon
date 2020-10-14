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
im = img.open("imgs/00_MPI3_200x200.png")


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
# channel 0 and 1 are just horizontal and vertical gradients
img_channel_0 = np.ones( (Nxpix, Nypix), dtype = float)
img_channel_0 = img_channel_0*grad_vals_x

img_channel_1 = np.ones( (Nxpix, Nypix), dtype = float)
img_channel_1 = img_channel_1*grad_vals_y[::-1]

img_channel_2 = np.ones( (Nxpix, Nypix), dtype = float)
img_channel_2 = masked_white_arr[:,:,1]*COLSAT


img_channel_3 = get_wave_img( np.zeros( (Nxpix,Nypix) ),  Ncrests = 4,  angle=45 )
img_channel_4 = get_wave_img( np.zeros( (Nxpix,Nypix) ),  Ncrests = 10, angle=135 )

Ncodes = 5
Allgrads= np.ndarray(shape=(Nxpix, Nypix, Ncodes), dtype=float)
Allgrads[:,:,0] = img_channel_0[:,:]
Allgrads[:,:,1] = img_channel_1[:,:]
Allgrads[:,:,2] = img_channel_2[:,:] # <--- the actual image
Allgrads[:,:,3] = img_channel_3[:,:]
Allgrads[:,:,4] = img_channel_4[:,:]

Allgrads_lin  = np.reshape( Allgrads, (Nxpix * Nypix, Ncodes), order='C')
np.savetxt("./out/arr_lin.tsv",     Allgrads_lin, fmt='%.18e', delimiter='\t')

np.random.shuffle(Allgrads_lin)
np.savetxt("./out/arr_lin_shuffled.tsv",     Allgrads_lin, fmt='%.18e', delimiter='\t')

# Y = tsne.tsne(Allgrads_lin, 2, 3, 20.0)
Y = tsne.tsne( Allgrads_lin, 2, 3, 20.0)

dat_out = np.hstack((Y,Allgrads_lin[:,2]))
np.savetxt("./out/tSNE_outdat.tsv", dat_out, fmt='%.8e', delimiter='\t')


arr2im = img.fromarray(masked_white_arr)
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
