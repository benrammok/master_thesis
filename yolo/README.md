# YOLO Training Pipeline
This folder contains files for preparing folder structure, convert Pascal VOC xml annotations to YOLO format.
Darknet is required to train and deploy a YOLO model. The orignal Darknet API can be found on https://github.com/pjreddie/darknet and 
a improved version which also features the new YOLOv4 can be found at https://github.com/AlexeyAB/darknet.
Weight files must also be provided externaly, these can be found on https://pjreddie.com/darknet/yolo/ or on AlexeyABs Darknet repo given above. 

# Preparing the folder structure
To convert Pascal VOC XML to YOLO format the annotation files and images should be placed in three spesific folders; /train_labels, /test_labes for annotations and /images.
These folders can be generated in the current working directory by calling the "prepare_folder_structure.py" file.

# Preparing the Dataset
To use a custom dataset, the only requirement is that the annotation files for the object detector uses the Pascal VOC XML format and that the labels have been placed in train and test folders.
Training and test images can be placed in the same folder. The generated YOLO format annotation is generated for each image and placed in the /image folder.

The xml_to_yolo.py have two primary options -i, --input which is the input path to the folders containing the Pascal VOC XML annotations.
-o, --output which is the output path to folder containing images. A classes.name and config.data file are generated which contain information about the classes contained in the dataset and paths to backup during training, path to test and train labels.

# Configuring YOLO config file
The "update_config_file.py" takes a YOLO configuration file as an input. Configuration files are availiable in the Darknet repo under the folder cfg.
The program takes a config file and classes.names file as input and modifes the filter values of the required layers in YOLO and changes the number of classes to the number of classes stated in the classes.names file.


# Training and Deploying a Custom Darknet Detector
To train a YOLO detector one can use the Darknet API.
Training is initiated using ./darknet detector train <path_to_data_file> <path_to_yolo_config_file> <path_to_yolo_weights_file>. If you are using pretrained weights as a starting point, one need to add -clear 0 to allow training to start,
if training is initiated from the pretrained weights it tend to stop after a single iteration.

# Evalutating the YOLO models using COCO
To evaluate the trained YOLOv3 detectors, one can apply the program developed by ydixion: https://github.com/ydixon/mAP_eval
The program uses the filepath of each test sample to create id for each detection, this is done by using regular expression to fetch the number in the filename.
In custom datasets the filename might not be with numbers which can be used as ID.

This can be fixed by changing two lines under the get_image_id_from_path function.
To fix this replace:
m = re.search(r'\d+$', image_path)
return int(m.group())
with:
m = re.search(r'\w+$', image_path)
return m.group() 
This uses the entire filename as an identifier and allows for custom datasets.
This must be changed in both pred_yolo2json.py and gt_yolo2json.py.
The final change must be done to gt_yolo2json.py where the line under create_annotations_dict must be changed from:
label_path_list = [img_path.replace('jpg', 'txt').replace('images', 'labels') for img_path in img_path_list]
to:
label_path_list = [img_path.replace('jpg', 'txt').replace('images', 'images') for img_path in img_path_list]
to search for labels in the image directory.