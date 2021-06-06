# Age-Gender-Classification
This repo is created to describe the project 'age-gender classification' conducted from May 30 to Jun 4. Also, it is used for finding and predicting gender and age of human from a video. 
My goal is 93 % acc for gender and MAE for 5.2 for age can be achieved after 50 epochs of training.

## Requirement
* `pip3 install --upgrade opencv-python, imutils, skimage`
* `pip3 install retina` for face detection

## Usage
#### *Training*
1. Firstly, `cd process_imdb-wiki`, run `IMDB-WIKI_dataset.ipynb` for filtered dataset IMDB-WIKI
2. Run `train_test_split.py` for create train.csv and test.csv file. Also, Using AFAD Dataset for finetune on AsianFace. Combine both of the datasets, get `train_combinedataset.csv` and `test_combinedataset.csv` for training and evaluation

3. Run `train.py`

#### *Real-time Prediction*
Run `inference.py`, put you video in this file and get the result.

## Details:
* Train a model base on ResNet18,
    - The latest conv feature is put into 2 discriminative branches: gender(2 neuron male-female) , age(10 neuron for bin-age normalized (0-9), each bin(10 neuron represent (0-9)))
    - Modify model can be found on `model.py`
* Test :
    - Using default retina (external python library) for face detection, I would recommend you train another model for face detection to get better result and inference.



