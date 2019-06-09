# 4.1.1 Negative reasoning: beyond the Best-Matching-Prototype principle
In Sec. 4.1.1 we use two models, one with 9 components and one with 10, to 
evaluate the negative reasoning and draw a comparison between CBCs and the 
Best-Matching-Prototype principle.

Which model to use can be specified using the `n_components` command line 
parameter. For both models there is also a pretrained weights file available. 
Additionally, we provide one for the experiment with 8 components discussed 
in the supplementary.

## Dataset
The MNIST dataset is available by default in Keras.

## Parameters
The following additional parameter settings are available.

|parameter|type|purpose|note|
|---|---|---|---|
|n_components|int|Number of components used by the model.|Defaults to 9|
