# 4.2 ImageNet
With close to 1.3 million images distributed over a 1000 classes, the 
ImageNet dataset is a good stress test for image classification models. 

To be able to run the provided scripts, a small modification needs to be made
to the Keras preprocessing package. In line with most literature on ImageNet
we used a custom rescaling of the images, this rescaling option is not 
available through the standard Keras Library. 

To be exact, the modification needs to be made in 
`keras_preprocessing.image.utils`. Here the function `load_img` need to be 
replaced with the one in `keras_preprocessing_image_utils_load_img.py` 
provided in this repository.

## Dataset
For our experiments we used the ImageNet dataset used during the ILSVRC 
competition organized in 2012. The dataset can be found 
[here](http://www.image-net.org/challenges/LSVRC/2012).

To run the script provided here, the location of the train and test folder 
needs to be specified using command line parameters. 

## Parameters
The following additional parameter settings are available.

|parameter|type|purpose|note|
|---|---|---|---|
|train_path|str|Path to the folder containing all training images, divided into different folders by class||
|test_path|str|Path to the folder containing all test images, divided into different folders by class||

