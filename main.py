from model import *
from data import *
import time
import sys

def train(trian_path, img_folder, mask_folder, weight_path, model_num, epochs, steps):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    seq = trainGenerator(2, trian_path,
                            img_folder,
                            mask_folder,
                            data_gen_args, save_to_dir=None)

    model = unet() if model_num == 0 else unet2()

    model_checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(histogram_freq=1)
    csv_logger = CSVLogger('%d_ep%d_stp%d.log' % (model_num, epochs, steps))
    model.fit_generator(seq, steps_per_epoch=steps, epochs=epochs, callbacks=[model_checkpoint, tensorboard, csv_logger])


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

if __name__ == '__main__':
    """
    File Paths
    """
    train_path = ".\\data\\train_data"
    train_folder = "origin"
    mask_folder = "mask"



    """
    Hyper Param
    """
    # epochs = 5
    # steps = 2000
    # model_num = 1
    epochs_list = [2, 3, 5, 8]
    steps_list = [500, 1000, 1500, 2000, 3000]
    for epochs in epochs_list:
        for steps in steps_list:
            for model_num in [0, 1]:
                postfix = "_%d_ep%d_stp%d" % (model_num, epochs, steps)

                weight_path = ".\\data\\weights\\unet" + postfix + ".h5"

                """
                Train
                """
                train_start_time = time.time()

                train(train_path, train_folder, mask_folder, weight_path, model_num, epochs, steps)

                train_end_time = time.time()

                print("*" * 30)
                print("Training Time: %.3f"% (train_end_time - train_start_time))


                """
                Test
                """
                """
                File Path
                """
                test_path = ".\\data\\train_data\\test"
                out_path = ".\\data\\train_data\\test" + postfix

                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                test_start_time = time.time()

                try:
                    test(test_path, out_path, weight_path, model_num, postfix)
                except:
                    print(sys.exc_info())

                test_end_time = time.time()
                print("Test Time: %.3f" % (test_end_time - test_start_time))


