from model import *
from data import *

def train(trian_path, img_folder, mask_folder, weight_path, epochs):
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
    model.fit_generator(seq, steps_per_epoch=300, epochs=epochs, callbacks=[model_checkpoint])

def test(input_path, output_path, weight_path):
    seq = test_seq_generator(input_path)

    model = unet()
    model.load_weights(weight_path)

    result = model.predict_generator(seq)
    save_result(output_path, result)

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
    epochs = 5

    weight_path = ".\\data\\weights\\unet_epoch" + str(epochs) + ".h5"

    # train(train_path, train_folder, mask_folder, weight_path, epochs)

    test_path = ".\\data\\train_data\\test"
    out_path = ".\\data\\train_data\\predict\\epoch_" + str(epochs)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    test(test_path, out_path, weight_path)

