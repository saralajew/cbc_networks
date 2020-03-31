# E.3 GTSRB
The GTSRB dataset contains images of 43 different traffic signs. Intuitively, 
the GTSRB classes should be easy to represent by a single class specific 
prototype.

We supply weights file for the CBC trained on the dataset and for the 
baseline CNN.

## Dataset
The GTSRB dataset can be downloaded 
[here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). 
We recommend using the official train and test set. 

The supplied script contains additional parameters to dictate where the test 
and train set can be found on the local file system. Specifying these paths 
is required. 

## Parameters
The following additional parameter settings are available.

|parameter|type|purpose|note|
|---|---|---|---|
|train_path|str|Path to the folder containing all training images. The training images are expected to be stored using a different folder for each class.||
|test_images_path|str|Path to the folder containing all test images in a single directory.||
|test_csv_path|str|Path to the csv file describing the mapping from file name to class for test images.||

