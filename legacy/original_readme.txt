=== Description ===

These are Caffe models finetuned on the MINC dataset. A model predicts 23 categories of materials. We found GoogleNet to perform the best.

The classifier predicts for the point at the center of the input patch. The remainder of the input patch may be more of the same material or may be contextual information. The classifier was trained to predict with an input patch which is 23.3% of the smaller image dimension when the image is a full photographc composition. See MINC-2500 or the included example image for a visualization of an appropriate patch size.

Further details appear in

@article{bell15minc,
	author = "Sean Bell and Paul Upchurch and Noah Snavely and Kavita Bala",
	title = "Material Recognition in the Wild with the Materials in Context Database",
	journal = "Computer Vision and Pattern Recognition (CVPR)",
	year = "2015",
}

Please cite this paper if you use these models in a publication.

Contact: sbell@cs.cornell.edu, paulu@cs.cornell.edu

=== Contents ===

categories.txt - An ordered list of categories (zero-indexed labels).

test.py - A classification example.

example.jpg - An example of an appropriate patch size.

deploy-googlenet.prototxt - BVLC GoogLeNet model.
minc-googlenet.caffemodel - Finetuned model weights.

deploy-vgg16.prototxt - VGG 16 layer model.
minc-vgg16.caffemodel - Finetuned model weights.

deploy-alexnet.prototxt - BVLC AlexNet model.
minc-alexnet.caffemodel - Finetuned model weights.

=== Model inputs ===

Inputs to the network should be scaled to [0,255], presented in BGR channel order and have a mean of B:104, G:117, R:124 subtracted. See the included test script that uses Caffe's Python interface (2015 June 18) for an example.

Our tests were conducted with the C++ extract_features tool and the testing procedure described below.


=== MINC testing procedure ===

We tested on MINC with 15x oversampling. First, consider a square region centered at each test location. This region has size

    p = 256 / 1100 * d

where d is the smaller image dimension. Then we consider 3 scaled squares: p*sqrt(2), p, p/sqrt(2). Each scaled square is extracted from the original image then resized to 256x256. Five 224x224 crops (center and corners) are then passed through the network. The final prediction is the average of 15 crop predictions.

