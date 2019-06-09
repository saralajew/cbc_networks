# 4.1.2 Interpretation of the reasoning
In Sec. 4.1.2 we explored the interpretation of CBC networks. For this purpose 
we used two CBCs, one with trainable pixel probabilities (alpha-CBC) and one 
with non-trainable pixel probabilities (diameter-CBC). 

Both models are available through the same script. By default the 
`patch_CBC_mnist.py` uses non-trainable pixel probabilities. By specifying the 
parameter `use_pixel_probabilities` trainable pixel probabilities are used. 

For both CBCs a trained weights file is available.

## Adversaries
In Sec. 4.1.2 we look at how the inherent interpretablility of the CBCs can 
be used to explain the success of an adversarial attack. To repeat this 
evaluation a dataset of adversaries is supplied. The dataset can be loaded 
and used for the evaluation by specifying the path to its *.npy file through 
the `eval_data` parameter.

## Dataset
The MNIST dataset is available by default in Keras.

## Parameters
The following additional parameter settings are available.

|parameter|type|purpose|note|
|---|---|---|---|
|eval_data|str|Provide a path to images which should be evaluated. Otherwise the test dataset is used. The method expects a *.npy file and and assumes that the images are preprocessed equivalent to the train/test data.||
|use_pixel_probabilities|flag|Train with pixel probabilities. Set this argument for the alpha-CBC. Otherwise it is equivalent to the diameter-CBC.||
