from model import *
from data import *
import time
import sys
import shutil
import skimage.io as io
from skimage import transform
# import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def train(train_path, img_folder, mask_folder, val_path, val_img_folder, val_mask_folder, weight_path,
          model_num, epochs, steps, postfix):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')


    seq = trainGenerator(2, train_path,
                            img_folder,
                            mask_folder,
                            data_gen_args,
                            save_to_dir=None)

    validata = valGenerator(2, val_path,
                                val_img_folder,
                                val_mask_folder,
                                data_gen_args, save_to_dir=None)

    model = unet() if model_num == 0 else unet2()

    tb_path = "./logs/" + postfix
    print(tb_path)

    if not os.path.exists(tb_path):
        os.mkdir(tb_path)

    tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True)

    model_checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit_generator(seq, validation_data=validata, validation_steps=100, steps_per_epoch=steps, epochs=epochs,
                                  callbacks=[model_checkpoint, tensorboard], workers=8)

    # GC
    del model
    del history


def test(input_path, output_path, weight_path, model_num, postfix):
    seq = testGenerator(input_path)

    model = unet() if model_num == 0 else unet2()
    model.load_weights(weight_path)

    steps = 0

    for root, dirs, files in os.walk(input_path):
        for file in files:
            steps += 1

    result = model.predict_generator(seq, steps, verbose=1)
    save_result(output_path, result, postfix)

def post_process(test_path, out_path):
    # Move every images from test_path to here
    for root, dirs, files in os.walk(test_path):
        for f in files:
            if f.endswith(".png"):
                path = os.path.join(root, f)
                shutil.copy(path, out_path)

    # Read image
    img_map = {}
    mask_map = {}

    for root, dirs, files in os.walk(out_path):
        for file in files:
            if file.endswith(".png"):
                tag = int(file.rstrip(".png").split("_")[0])
                if "pred" in file:
                    mask_map[tag] = (root, file)
                elif "_" not in file:
                    img_map[tag] = (root, file)

    print("Img map: %d" % len(img_map.keys()))
    print("Mask map: %d" % len(mask_map.keys()))

    # Batch Convert
    for k, v in img_map.items():
        img_path = os.path.join(v[0], v[1])
        img = io.imread(img_path, as_gray=True)

        img = transform.resize(img, (256, 256))

        try:
            mask_val = mask_map.get(k)
            mask_path = os.path.join(mask_val[0], mask_val[1])
            mask = io.imread(mask_path, as_gray=True)

            # Process mask
            mask = mask * 1.0 / 255
            img = img * 1.0

            res = img * mask

            outpath = os.path.join(v[0], v[1].rstrip(".png") + "_mask.png")

            plt.imsave(outpath, res, cmap=plt.get_cmap("gray"))
        except:
            print("err. img = %s", img_path)




if __name__ == '__main__':
    """
    File Paths
    """
    train_path = "./data/train_data"
    train_folder = "origin"
    mask_folder = "mask"

    val_path = "./data/val_data"
    val_og_folder = "origin"
    val_mask_folder = "mask"


    """
    Hyper Param
    """
    # epochs = 5
    # steps = 2000
    # model_num = 1
    # epochs_list = [2, 3, 5, 8]
    # steps_list = [500, 1000, 1500, 2000, 3000]

    # param_list = [(5, 1500), (10, 500), (10, 1500), (50, 1000), (100, 300), (100, 500), (100, 1000)]
    # param_list = [(10, 500), (10, 1500), (50, 1000), (100, 300), (100, 500), (100, 1000)]
    param_list = [(50, 1000), (80, 1000), (100, 1000), (100, 2000)]
    for epochs, steps in param_list:

        for model_num in [0]:


            """
            Config
            """

            postfix = "_%d_ep%d_stp%d_td3" % (model_num, epochs, steps)

            print("*" * 30)
            print(postfix)

            weight_path = "./data/weights/unet" + postfix + ".h5"

            # sys.stdout = Logger("./logs/" + postfix + ".txt")


            """
            Train
            """
            train_start_time = time.time()

            train(train_path, train_folder, mask_folder, val_path, val_og_folder, val_mask_folder,
                  weight_path, model_num, epochs, steps, postfix.lstrip("_"))

            train_end_time = time.time()

            print("*" * 30)
            print("Training Time: %.3f" % (train_end_time - train_start_time))


            """
            Test
            """
            """
            File Path
            """
            test_path = "./data/test_data"
            out_path = "./data/output/test" + postfix

            if not os.path.exists(out_path):
                os.mkdir(out_path)
            test_start_time = time.time()

            try:
                test(test_path, out_path, weight_path, model_num, postfix)
            except:
                print(sys.exc_info())

            test_end_time = time.time()
            print("Test Time: %.3f" % (test_end_time - test_start_time))

            sys.stdout.flush()

            """
            Post Process
            """
            # post_process(test_path, out_path)



