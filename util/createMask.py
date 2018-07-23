import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import sys
from matplotlib import pyplot as plt


def read_image(image_path):
    """
    Loads the npy file of the image to be shifted.
    ---------------------------------------------
    Input:
    imageName   - numpy image

    Output:
    img         - numpy array
    """
    np_img = np.asanyarray(Image.open(image_path).convert('L'))
    np_img.flags.writeable = True
    np_img[np_img != 0] = 255
    # plt.imshow(np_img)

    return np_img

# def mask(img):



if __name__ == '__main__':

    in_dir = './data/train_data/mask'

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            img_array = read_image(os.path.join(root, file))
            img = Image.fromarray(img_array)
            img.save(os.path.join(root, file))
            # img.show()

    # plt.show()
