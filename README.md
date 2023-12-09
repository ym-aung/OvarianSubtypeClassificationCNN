# CS184A CNN Model Final

## Setup Instructions
1. Clone this repository.
  - There should be 2 files besides the ReadMe: **data_preprocessing.py** and **model.py**.
  - **data_preprocessing.py** contains the code used to process the imageset from Kaggle.
  - **model.py** contains the code for training and visualizing results of the CNN model.
2. We used the train_thumbnails folder from https://www.kaggle.com/competitions/UBC-OCEAN/data as the dataset for the project, but this has not been included here because of Kaggle's rules.
  - To run this model, download this folder from the Kaggle website in the same directory as model.py and data_preprocessing.py.
  - Then to prepare the processed images folder, run the data_preprocessing.py file. You may need to run the terminal command **pip install Pillow** for this file to run properly.

## Running the Model
1. To run the model, first follow the setup instructions and then run model.py.
  - NOTE: To run the model, you may need to pip install the following modules with these commands:
    - **pip install -U scikit-learn**
    - **pip install torch torchvision**
    - **pip install -U albumentations**
