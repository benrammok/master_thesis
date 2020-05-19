import os

"""
This python scripts prepare the folder structure needed for generation of
TFRecords which are used to train Tensorflow models
"""

for folder in ['train_labels', 'test_labels', 'images', 'pretrained_model', 'training']:
    if not os.path.isdir(os.path.join(os.getcwd(), folder)):
        os.mkdir(folder)