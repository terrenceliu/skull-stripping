import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import sys

def loadImage(imageName):
    """
    Loads the npy file of the image to be shifted.
    ---------------------------------------------
    Input:
    imageName   - numpy image

    Output:
    img         - numpy array
    """
    np_img = np.asanyarray(Image.open(imageName).convert('L'))
    return np_img

#
# def loadMaskImage(imageName):
#     """
#     Loads the npy file of the masked image to be shifted.
#     ---------------------------------------------
#     Input:
#     imageName   - numpy image
#
#     Output:
#     img         - numpy array
#     """
#     img = np.load(imageName)
#     return img
#
#
# def loadMask(imageName):
#     """
#     Loads the npy file of the mask to be shifted.
#     ---------------------------------------------
#     Input:
#     imageName   - numpy image
#
#     Output:
#     img         - numpy array
#     """
#     img = np.load(imageName)
#     return img

def shift_right(image, n_pixels):
    """
    Shifts image to right by n_pixels
    --------------------------------------------
    Input:
    image     -  Numpy image array
    n_pixels  -  Number of pixels to move image by

    Output:
    dat       -  Shifted numpy image
    """
    dat = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    dat[:,n_pixels:] = image[:,:image.shape[1]-n_pixels]
    return dat

def shift_left(image, n_pixels):
    """
    Shifts image to left by n_pixels
    --------------------------------------------
    Input:
    image     -  Numpy image array
    n_pixels  -  Number of pixels to move image by

    Output:
    dat       -  Shifted numpy image
    """
    dat = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    dat[:,:dat.shape[1]-n_pixels] = image[:,n_pixels:]
    return dat
 
def shift_up(image, n_pixels):
    """
    Shifts image up by n_pixels
    --------------------------------------------
    Input:
    image     -  Numpy image array
    n_pixels  -  Number of pixels to move image by

    Output:
    dat       -  Shifted numpy image
    """
    dat = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    dat[:dat.shape[0]-n_pixels,:] = image[n_pixels:,:]
    return dat
 
def shift_down(image, n_pixels):
    """
    Shifts image down by n_pixels
    --------------------------------------------
    Input:
    image     -  Numpy image array
    n_pixels  -  Number of pixels to move image by

    Output:
    dat       -  Shifted numpy image
    """
    dat = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    dat[n_pixels:,:] = image[:dat.shape[0]-n_pixels,:]
    return dat


def run(file_name):
    input_dir = './data/train_data'

    out_dir = './data/train_data/transform'

    # Output directory
    outFile1 = out_dir + '/image_' + file_name + '.npy'
    # outFile2 = './data/train_data/roiIm_' + file_name + '.npy'
    outFile3 = out_dir + '/mask_' + file_name + '.npy'

    # Input files
    in_Image = input_dir + '/origin/' + file_name + '.png'
    # in_roi = 'results/t1_3/t1data_' + file_name + '.npy'
    in_mask = input_dir + '/mask/' + file_name + '.tif'

    imgs_out = np.zeros((21, 512, 512), dtype=np.float32)
    # roi_out = np.zeros((21, 256, 256), dtype=np.float32)
    mask_out = np.zeros((21, 512, 512), dtype=np.float32)

    try:
        imgs = loadImage(in_Image)
        # imgs_scale = loadMaskImage(in_roi)
        masks = loadImage(in_mask)

        moves1 = [0, 2, 4, 6, 8, 10, 12]
        moves2 = [2, 4, 6, 8, 10, 12, 14]
        moves3 = [2, 4, 6, 8, 10, 12, 14]
        moves4 = [2, 4, 6, 8, 10, 12]

        imgs_out[0] = imgs
        # roi_out[0] = imgs_scale
        mask_out[0] = masks

        for i in range(1, len(moves1)):
            imgs_out[i] = shift_right(imgs, moves1[i])
            # roi_out[i] = shift_right(imgs_scale,moves1[i])
            mask_out[i] = shift_right(masks, moves1[i])

        for i in range(0, len(moves2)):
            imgs_out[i + 7] = shift_left(imgs, moves2[i])
            # roi_out[i+7] = shift_left(imgs_scale,moves2[i])
            mask_out[i + 7] = shift_left(masks, moves2[i])

        for i in range(0, len(moves3)):
            imgs_out[i + 14] = shift_up(imgs, moves3[i])
            # roi_out[i+14] = shift_up(imgs_scale,moves3[i])
            mask_out[i + 14] = shift_up(masks, moves3[i])

        #    for i in range(0,len(moves4)):
        #        imgs_out[i+19] = shift_down(imgs,moves4[i])
        #        roi_out[i+19] = shift_down(imgs_scale,moves4[i])
        #        mask_out[i+19] = shift_down(masks,moves4[i])

        np.save(outFile1, imgs_out)
        # np.save(outFile2, roi_out)
        np.save(outFile3, mask_out)

        print("Finish " + file_name)
    except:
        print(sys.exc_info()[0])
        return

if __name__ == '__main__':


    workload = []
    for root, dirs, files in os.walk('./data/train_data/origin'):
        for file in files:
            if file.endswith('.png'):
                file_name = file.rstrip(".png")
                workload.append(file_name)

    print("CPU: %d" % cpu_count())

    p = Pool(cpu_count())

    p.map(run, workload)




