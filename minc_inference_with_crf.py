# -*- coding:utf-8 -*-


"""
Script to perform full scene segmentation as described, and using the
training networks from http://opensurfaces.cs.cornell.edu/publications/minc/#

It basically consists of 2 steps:

1. Perform patch classification via sliding window
2. Upsample output and pass it to a CRF

More details in the paper.
"""


import itertools
import caffe
import pydensecrf.densecrf as dcrf  # CRF
from skimage.color import rgb2lab
import numpy as np
import glob
import os.path
import sys


CATEGORIES = ["brick", "carpet", "ceramic", "fabric", "foliage",
              "food", "glass", "hair", "leather", "metal",
              "mirror", "other", "painted", "paper", "plastic",
              "polishedstone", "skin", "sky", "stone", "tile",
              "wallpaper", "water", "wood"]

COMPATIBILITIES = 2 * np.float32([
    [0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 1. , 3. ],
    [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
     1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0. , 3. ],
    [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
     1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0. ]])


def sliding_window(img, stride, window_shape, zero_pad=True):
    """
    :param zero_pad: If true, windows over the corners will be padded
      with zeros, otherwise won"t be included.
    """
    img_h, img_w, img_ch = img.shape
    dt = img.dtype
    stride_y, stride_x = stride
    win_h, win_w = window_shape
    for y in range(0, img_h, stride_y):
        for x in range(0, img_w, stride_x):
            patch = img[y:y + win_h, x:x + win_w]
            p_h, p_w = patch.shape[:2]
            if (p_h, p_w) == window_shape:
                yield (x, y, patch)
            else:
                if zero_pad:
                    padded = np.zeros((win_h, win_w, img_ch), dtype=dt)
                    padded[:p_h, :p_w] = patch
                    yield (x, y, padded)


def batched_sliding_window(img, stride, window_shape, batch_size,
                           zero_pad=True):
    """
    """
    g = sliding_window(img, stride, window_shape, zero_pad)
    gg = list(itertools.islice(g, batch_size))
    while gg:
        xxx, yyy, windows = zip(*gg)
        batch = np.stack(windows)
        yield (xxx, yyy, batch)
        gg = list(itertools.islice(g, batch_size))


def sliding_window_inference(img, net, stride, window_shape,
                             batch_size, zero_pad=True):
    """
    """
    # Cut image into batches of strided windows and pass to NN
    results = {}
    for (xx, yy, batch) in batched_sliding_window(img, stride, window_shape,
                                                  batch_size, zero_pad):
        print("[processing batch of patches starting at]:", zip(xx, yy))
        z = net.predict(batch)
        for xxx, yyy, zzz in zip(xx, yy, z):
            results[(xxx, yyy)] = zzz
    # lazy way of converting results dict into a numpy array
    some_z = next(iter(results.values()))
    dt = some_z.dtype
    num_classes = len(some_z)
    x_entries, y_entries = set(), set()
    for xe, ye in results.keys():
        x_entries.add(xe)
        y_entries.add(ye)
    out_arr = np.zeros((len(y_entries), len(x_entries), num_classes), dtype=dt)
    # lazy way of filling array with results:
    for x_i, x_e in enumerate(sorted(x_entries)):
        for y_i, y_e in enumerate(sorted(y_entries)):
            try:
                out_arr[y_i, x_i] = results[(x_e, y_e)]
            except KeyError:
                print("missing prediction:", (x_e, y_e))
    return out_arr


# #############################################################################
# # CRF
# #############################################################################
def crf_pass(img, prob_maps, stddev_loc=0.1, stddev_L=10.0, stddev_ab=5.0,
             num_crf_iters=20):
    """
    :param prob_maps: per-pixel probabilities, ``np.float32(h, w, classes)``
      with values between 0 and 1.

    https://github.com/lucasb-eyer/pydensecrf
    https://github.com/keunhong/kitnn/blob/master/kitnn/models/minc.py
    """
    img_h, img_w, img_ch = img.shape
    maps_h, maps_w, num_classes = prob_maps.shape
    assert img_h == maps_h, "prob_maps must have same (h, w) as img!"
    assert img_w == maps_w, "prob_maps must have same (h, w) as img!"
    assert prob_maps.min() >= 0, "prob_maps must be in range [0, 1]!"
    assert prob_maps.max() <= 1, "prob_maps must be in range [0, 1]!"
    #
    crf = dcrf.DenseCRF2D(img_h, img_w, num_classes)
    # add activations as unary energy
    U = np.ones((num_classes, img_h * img_w), dtype=np.float32)
    U[:] = prob_maps.reshape((img_h * img_w, num_classes)).T
    crf.setUnaryEnergy(-np.log(U))
    # add pairwise features: 2 for x/y coords, 3 for image channels
    NUM_FEATURES = 5
    input_features = np.zeros((NUM_FEATURES, img_h, img_w), dtype=np.float32)
    #
    local_span = float(min(img_h, img_w))
    input_features[:2] = np.mgrid[0:img_h, 0:img_w] / (local_span * stddev_loc)
    input_features[2:5] = rgb2lab(img).transpose(2, 0, 1)
    input_features[2] /= stddev_L
    input_features[3] /= stddev_ab
    input_features[4] /= stddev_ab
    #
    crf.addPairwiseEnergy(input_features.reshape(NUM_FEATURES, -1),
                          compat=COMPATIBILITIES)
    crf_out = crf.inference(num_crf_iters)
    crf_out = np.array(crf_out).reshape(
        num_classes, img_h, img_w).transpose(1, 2, 0)
    return crf_out



# #############################################################################
# # MAIN ROUTINE
# #############################################################################
def run_nn(img_path="data/example.jpg"):
    """
    Runs the neural net via sliding window
    """
    #globals
    WINDOW_SHAPE = (150, 150)  # y, x
    # STRIDE = (50, 50)  # y, x
    STRIDE = (150, 150)  # y, x
    BATCH_SIZE = 25
    ZERO_PAD = True
    ARCH = "googlenet"  # "alexnet" "vgg16"
    #
    print("[loading image]")
    img = (caffe.io.load_image(img_path)*255).astype(np.uint8)
    print("[loading NN]")
    net = caffe.Classifier("models/deploy-{}.prototxt".format(ARCH),
                           "models/minc-{}.caffemodel".format(ARCH),
                           channel_swap=(2, 1, 0),
                           mean=np.array([104, 117, 124]))
    print("[performing sliding window inference]")
    out_arr = sliding_window_inference(img, net, STRIDE, WINDOW_SHAPE,
                                       BATCH_SIZE, ZERO_PAD)
    np.savez(img_path + "_features", categories=CATEGORIES,
             activations=out_arr)
    print("[saved predictions]")
    import matplotlib.pyplot as plt
    plt.clf(); plt.imshow(out_arr[:, :, 1]); plt.show()
    breakpoint()

def load_features_and_run_crf(img_path="data/example.jpg",
                              nn_out_path="data/example.jpg_features.npz"):
    """
    Loads the output of the NN, passes it through a CRF and argmaxes to
    obtain segmentation.
    """
    # The bigger, the more sensitive to that
    STDDEV_LOC = 0.1  # 0.1
    STDDEV_L = 1.0  # 10.0
    STDDEV_AB = 10.0  # 5.0
    NUM_CRF_ITERS = 20
    print("[loading image]")
    img = (caffe.io.load_image(img_path)*255).astype(np.uint8)
    print("[loading predictions]")
    activations = np.load(nn_out_path)["activations"]  # float32(h, w, ch)
    resized_activations = caffe.io.resize_image(activations, (660, 990),
                                                interp_order=2).clip(1e-10, 1)
    print("[CRF pass]")
    refined_activations = crf_pass(img, resized_activations, STDDEV_LOC,
                                   STDDEV_L, STDDEV_AB, NUM_CRF_ITERS)
    print("[argmax for segmentation]")
    segmentation = refined_activations.argmax(axis=-1)
    np.savez(img_path + "_segmentation", categories=CATEGORIES,
             segmentation=segmentation)
    print("[saved segmentation]")
    import matplotlib.pyplot as plt
    plt.clf(); plt.imshow(segmentation); plt.show()
    # plt.clf(); plt.imshow(resized_activations[:, :, 1]); plt.show()
    # plt.clf(); plt.imshow(refined_activations[:, :, 1]); plt.show()
    breakpoint()




if __name__ == "__main__":
    run_nn(img_path="data/example.jpg")
    load_features_and_run_crf()

