import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

# Overlay small image on another image
def AddPictures(image_template, image_add, x_offset, y_offset, scale):
    # resize added image
    SIZE = (int(image_add.shape[1]/scale), int(image_add.shape[0]/scale))
    s_img = cv2.resize(image_add, SIZE, interpolation=cv2.INTER_AREA)
    # Add intensity
    s_img = np.clip(s_img*5,0, 200)
    # stack to 3d array
    bin_stack = np.zeros_like(s_img)
    s_map = np.dstack([s_img,bin_stack,bin_stack])
    # img overlay
    image_template[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_map    
    return image_template

def MultiplePlots(grid, images, titles, savetofile=None):
    nrow = grid[0]
    ncol = grid[1]
    plt.figure(figsize=(ncol+1, nrow+1)) 
    gs = gridspec.GridSpec(nrow, ncol,
             wspace=0.2, hspace=0.2, 
             top=1.-0.2/(nrow+1), bottom=0.2/(nrow+1), 
             left=0.2/(ncol+1), right=1-0.2/(ncol+1))

    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i,j])
            img = images[i][j]
            title = titles[i][j]
            ax.imshow(img)
            ax.set_title(title)
            #xticks = np.arange(0, img.shape[1], 100)
            #yticks = np.arange(0, img.shape[0], 100)
            xticks = []
            yticks = []
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(xticks)
            ax.set_yticklabels(yticks)
            
    # save or plot        
    if savetofile is None:
        plt.show()
    else:
        plt.savefig(savetofile)