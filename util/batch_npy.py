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


def run(file_name):
    input_dir = './data/train_data'
    out_dir = './data/train_data/npy'

    # Output directory
    outFile1 = out_dir + '/' + file_name + '.npy'
    # outFile2 = './data/train_data/roiIm_' + file_name + '.npy'
    outFile3 = out_dir + '/' + file_name + '_mask.npy'

    # Input files
    in_Image = input_dir + '/origin/' + file_name + '.png'
    in_mask = input_dir + '/mask/' + file_name + '.tif'

    imgs_out = np.zeros((512, 512), dtype=np.float32)
    # roi_out = np.zeros((21, 256, 256), dtype=np.float32)
    mask_out = np.zeros((512, 512), dtype=np.float32)

    try:
        imgs = loadImage(in_Image)
        masks = loadImage(in_mask)

        imgs_out = imgs
        mask_out = masks

        np.save(outFile1, imgs_out)
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