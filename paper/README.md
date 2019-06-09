# Experiments scripts
This folder contains all scripts required to replicate the results presented 
in the paper. Each section is represented in a different folder. 

While most parameters for the scripts are hard coded in line with the settings 
used in the paper, some settings are available as command line parameters. 
These parameters are dependent on the architecture on which the script is ran 
and therefore impossible to specify before hand. However, when these 
parameters are not specified when the script is called, the default values 
will represent the setting used in the paper.

## Parameters
The following standard parameter settings are available.

|parameter|type|purpose|note|
|---|---|---|---|
|weights|str|Load *.h5 model trained weights||
|save_dir|str|Output directory|Defaults to './output'|
|epochs|int|Number of epochs the model is trained for|Handle with care for the CBCs for CIFAR-10 and GTSRB as they require custom loss schedulers.|
|lr|float|Initial learning rate of the optimizer||
|batch_size|int|Batch size used to train the model||
|gpu|int|Index of the GPU used during training|For the experiments on ImageNet, two gpus need to be specified seperated by a comma.|
|eval|flag|Skips training and instead only evaluates the model||
