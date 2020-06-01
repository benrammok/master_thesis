import pandas as pd
import xml.etree.ElementTree as ElementTree
import glob
import os
import argparse
from tqdm import tqdm



""" 
A python program for converting a set of XML files in VOC format to a single CSV file
"""
def _remove_double_entry(a_list):
    """
    Parameters
    ----------
    a_list : list
        List containing multiple entries to be removed
    
    Returns
    -------
        List with the multiple entries supressed
    """
    return list(set(a_list))

"""
Helper function for converting VOC bounding boxes using x1, y1, x2 and y2 coordinates
to designate left upper box and right lower box edges to darknet/yolo format which takes x and y as box center
and width and height of box

Returns:
    object_data[tuple] -- A new tuple containing class and the converted coordinates
"""
def _convert_to_darknet_format(img_width, img_height, object_data):
    # Scaler is used to normalize coordinate in respect to the image size
    scaler_x = 1.0 / img_width
    scaler_y = 1.0 / img_height

    x = scaler_x * (object_data[1] + object_data[3]) / 2.0
    y = scaler_y * (object_data[2] + object_data[4]) / 2.0
    w = scaler_x * (object_data[3] - object_data[1])
    h = scaler_y * (object_data[4] - object_data[2])

    return (object_data[0], x, y, w, h)

def xml_to_yolo(path):
    """ 
    Parameters
    ---------
    path : str
        Path to folder containing VOC XML files
    
    Returns
    -------
        xml_yolo: dictionary
            A dictionary with filename as key and objects as data
        class_labels : list
            A list containing the class labels found in both the test and train dataset
    """
    print(f"[INFO] Processing XML files in {path}")
    class_labels = []
    xml_yolo = {}
    for xml_file in tqdm(glob.glob(path + "/*.xml")):

        tree = ElementTree.parse(xml_file)
        root_node = tree.getroot()
        filename = root_node.find('filename').text
        image_width = 0
        image_height = 0
        obj = []
        for elements in root_node.findall('object'):
            # Create a tuple containg: filename, image width and height, class name, bounding box parameters
            # Using find to find sub element and then the element within, makes the code more flexible
            # if data is moved or rearraged in the XML files which can happen using find prevents sampling of wrong data

            image_width = int(root_node.find('size').find('width').text)
            image_height = int(root_node.find('size').find('height').text)

            # For Yolo we only need to know the class, and the bounding box coordinates
            object_data = (
                        elements.find('name').text,
                        float(elements.find('bndbox').find('xmin').text),
                        float(elements.find('bndbox').find('ymin').text),
                        float(elements.find('bndbox').find('xmax').text),
                        float(elements.find('bndbox').find('ymax').text))

            # Convert bounding box cooridnates from x1, y1, x2, y2 format to x, y, w, h format
            object_data = _convert_to_darknet_format(image_width, image_height, object_data)
            obj.append(object_data)

            class_labels.append(elements.find('name').text)
        
        if filename in xml_yolo:
            continue
        else:
            xml_yolo[filename] = obj

    # Clean up the class labels by creating a list, this removes dual entries
    class_labels = _remove_double_entry(class_labels)
    class_labels.sort()
    return xml_yolo, class_labels

def create_pbtxt_content(class_labels, verbose):
    """ 
    Parameters
    ---------
    class_labels : list
        List containing class labels for the dataset
    
    Returns
    -------
    pb_text_content : str
        String containg each class and corresponding mapping
    """
    pb_text_content = ""
    for i, cl in enumerate(class_labels):
        if verbose: print(f"[INFO] Found object class: {cl}")
        pb_text_content += "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, cl)
    pb_text_content.strip()
    return pb_text_content

def _return_class_id(class_labels, name):
    for i, cl in enumerate(class_labels):
        if cl == name:
            return i
    return None

def main():
    
    parser = argparse.ArgumentParser('XML to CSV Formatter', 'A Python Program for converting a set of XML files in VOC format to CSV format')
    parser.add_argument('-i', '--input', type=str, default='', help='Input path to folders containing VOC XML files (Folders must have the names of \'train_labels\' and \'test_labels\')')
    parser.add_argument('-o', '--output', type=str, default='', help='Output path to image folder storing the formatted output (Defaults to image folder in current working directory)')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Displays more information during the conversion process')
    args = parser.parse_args()

    # Create Path to image directory
    # This directory will hold the yolo txt files and the image files
    if args.output == "":
        image_dir_path = os.path.join(os.getcwd(), 'images')
    else:
        image_dir_path = os.path.join(args.output)

    
    # We need to find the path of each folder containing the train and test labels and we need to generate a .names file
    # Combine any classes found in the train and test labels.
    # If by any chance that there exist a class in the train set which is not in the test set,
    # or wise versa this ensures that every class is accounted for
    combined_class_labels = []

    formated_files = []

    # Find path to the xml label files
    for label in ['train_labels', 'test_labels']:
        # Create Path for the input file
        if args.input == "":
            label_path = os.path.join(os.getcwd(), label)
        else:
            label_path = os.path.join(args.input, label)


  

        # Create dictonary containing every object extracted from the dataset labels
        # the xml_yolo_format is a dictionary with filename as key and detection information as a list
        xml_yolo_format, class_labels = xml_to_yolo(label_path)
        formated_files.append((label, xml_yolo_format))
        print(f"[Info] Number of files {len(xml_yolo_format.keys())}")
        combined_class_labels += (item for item in class_labels)

       
               
    # Remove double entries
    combined_class_labels = _remove_double_entry(combined_class_labels)
    combined_class_labels.sort()
    # Create Path for train and test txt files
    if args.output == "":
        output_path = os.getcwd()
    else:
        output_path = os.path.join(args.output)

    # For each dataset train and test, loop trough the files and objects and create a txt file for each and create a test and train txt containting image paths
    for label_set in formated_files:
        images_in_set = ""
        for filename, object_list in tqdm(label_set[1].items()):
            with open(os.path.join(image_dir_path, filename.split(".")[0] + ".txt"), 'w') as f:
                for obj in object_list:
                    f.write(f"{_return_class_id(combined_class_labels, obj[0])} {obj[1]} {obj[2]} {obj[3]} {obj[4]} \n")

            images_in_set += os.path.join(image_dir_path, filename) + "\n"
        with open(os.path.join(output_path, label_set[0] + ".txt"), 'w') as f:
            f.write(images_in_set)
        print(f"[INFO] Finished writing {label_set[0]}.txt to {output_path}\n")
        
    # Generate classes.names file which contains each of the classes the model is to be trained on.
    with open(os.path.join(output_path, "classes.names"), "w") as f:
        print(combined_class_labels)
        for labels in combined_class_labels:
            f.write(labels + '\n')
    print(f"[INFO] Finished writing classes.names to {output_path}\n")

    # Create a .data file which is needed for training YOLOv3
    with open(os.path.join(output_path, "config.data"), "w") as f:
            config_text = ""
            config_text += f"classes={len(combined_class_labels)}\n"  
            config_text += f"train = {os.path.join(output_path, 'train_labels.txt')}\n"
            config_text += f"valid = {os.path.join(output_path, 'test_labels.txt')}\n"
            config_text += f"names = {os.path.join(output_path, 'classes.names')}\n"
            config_text += f"backup = backup/\n"      
            f.write(config_text)
    print(f"[INFO] Finished writing config.data to {output_path}\n")

if __name__ == "__main__":
    main()
