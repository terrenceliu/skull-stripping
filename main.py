from model import *
from data import *
import time

def train(trian_path, img_folder, mask_folder, weight_path, epochs, steps):
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

    model = unet()
    model_checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(seq, steps_per_epoch=steps, epochs=epochs, callbacks=[model_checkpoint])

def test(input_path, output_path, weight_path, postfix):
    seq = testGenerator(input_path)

    model = unet()
    model.load_weights(weight_path)

    steps = -1

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

    test_path = ".\\data\\train_data\\test"
    out_path = ".\\data\\train_data\\test"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    """
    Hyper Param
    """
    epochs = 5
    steps = 2000

    postfix = "_ep5_stp2000_bientro"

    weight_path = ".\\data\\weights\\unet_epoch" + postfix + ".h5"

    """
    Train
    """
    train_start_time = time.time()

    # train(train_path, train_folder, mask_folder, weight_path, epochs)
    train_end_time = time.time()

    print("*" * 30)
    print("Training Time: %.3f"% (train_end_time - train_start_time))


    """
    Test
    """
    test_start_time = time.time()


    test(test_path, out_path, weight_path, postfix)

    test_end_time = time.time()
    print("Test Time: %.3f" % (test_end_time - test_start_time))


