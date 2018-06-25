from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io
import skimage.transform as trans

class TrainSequence(Sequence):
    """
    Sequence
    keras.utils.Sequence
    """

    def __init__(self, x_set, y_set, batch_size, img_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        res_x = np.array([resize(imread(file_name, as_gray=True), (self.img_size, self.img_size)) for file_name in batch_x])
        # res_y = np.array([resize(imread(file_name, as_gray=True), (self.img_size, self.img_size)) for file_name in batch_y])

        res_x = res_x[..., np.newaxis]
        # res_y = res_y[..., np.newaxis]

        return res_x, 0


class TestSequence(Sequence):
    """
    Sequence
    keras.utils.Sequence
    """

    def __init__(self, x_set, batch_size, img_size):
        self.x = x_set
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        res_x = np.array(
            [resize(imread(file_name, as_gray=True), (self.img_size, self.img_size)) for file_name in batch_x])

        res_x = res_x[..., np.newaxis]

        return res_x

def adjust_data(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    # Sanity Check: Display Image
    # print(img.shape)
    # io.imshow(img[0][:, :, 0])
    # io.imshow(mask[0][:, :, 0])
    # io.show()
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask)
        yield (img, mask)


# def test_seq_generator(img_path, batch_size=32, img_size=256):
    # x_path = img_path
    #
    # x_set = []
    #
    # for root, dirs, files in os.walk(x_path):
    #     for file in files:
    #         if file.endswith(".png"):
    #             x_set.append(os.path.join(root, file))
    #
    # seq = TestSequence(x_set, batch_size, img_size)
    # return seq

def testGenerator(test_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    file_dict = {}
    for root, dirs, files in os.walk(test_path):
        for file in files:
            file_dict[int(file.rstrip(".png"))] = os.path.join(root, file)

    k_srt = sorted(file_dict.keys())

    for key in k_srt:
        img_path = file_dict[key]
        img = io.imread(img_path, as_gray=as_gray)
        # img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def save_result(out_path, result, postfix):
    for i, item in enumerate(result):
        img = item[:, :, 0]
        io.imsave(os.path.join(out_path, "%d_pred_%s.png" % (i, postfix)), img)
