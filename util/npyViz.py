import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # img_array = np.load('imgs_mask_test.npy')

    img_array = np.load('./data/imgs_test.npy')
    print(img_array.shape)

    for i in range(img_array.shape[0]):

        mask = img_array[i, :, :]
        plt.imshow(mask, cmap='gray')
        plt.show()
