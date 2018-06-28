from model import *
from data import *
import time
import sys
import shutil
import skimage.io as io
from skimage import transform
import matplotlib.pyplot as plt


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
    File Path
    """

    postfix = "_0_ep50_stp1000_sfd"

    test_path = ".\\data\\test_data"
    out_path = ".\\data\\output\\test" + postfix

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    weight_path = ".\\data\\weights\\unet" + postfix + ".h5"

    model_num = postfix.split("_")[1]
    model_num = int(model_num)

    print("Model Num: %d" % model_num)

    try:
        test(test_path, out_path, weight_path, model_num, postfix)
    except:
        print(sys.exc_info())

    """
    Post Process
    """
    post_process(test_path, out_path)
