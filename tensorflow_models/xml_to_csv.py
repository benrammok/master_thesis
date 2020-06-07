# Code is written from the ground up with help and inspiration from:
# from: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d
# Added support for multiclass and reworked xml_to_csv function
# Added custom handling for argparser.
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

def xml_to_csv(path):
    """ 
    Parameters
    ---------
    path : str
        Path to folder containing VOC XML files
    
    Returns
    -------
        xml_pd: pandas.Dataframe
            A Pandas Dataframe containing the data extracted from the VOC XML
        class_labels : list
            A list containing the class labels found in both the test and train dataset
    """
    print(f"[INFO] Processing XML files in {path}")
    class_labels = []
    xml_files = []
    for xml_file in tqdm(glob.glob(path + "/*.xml")):
        tree = ElementTree.parse(xml_file)
        root_node = tree.getroot()
        for elements in root_node.findall('object'):
            # Create a tuple containg: filename, image width and height, class name, bounding box parameters
            # Using find to find sub element and then the element within, makes the code more flexible
            # if data is moved or rearraged in the XML files which can happen using find prevents sampling of wrong data
            obj = (root_node.find('filename').text,
                   int(root_node.find('size').find('width').text),
                   int(root_node.find('size').find('height').text),
                   elements.find('name').text,
                   int(float(elements.find('bndbox').find('xmin').text)),
                   int(float(elements.find('bndbox').find('ymin').text)),
                   int(float(elements.find('bndbox').find('xmax').text)),
                   int(float(elements.find('bndbox').find('ymax').text)))
            class_labels.append(elements.find('name').text)
            xml_files.append(obj)

    column_label = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] 
    # Use pandas to create a dataframe with the column_label as column description
    xml_pd = pd.DataFrame(xml_files, columns=column_label)
    # Clean up the class labels by creating a list, this removes dual entries
    class_labels = _remove_double_entry(class_labels)
    class_labels.sort()
    return xml_pd, class_labels

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

def main():
    
    parser = argparse.ArgumentParser('XML to CSV Formatter', 'A Python Program for converting a set of XML files in VOC format to CSV format')
    parser.add_argument('-i', '--input', type=str, default='', help='Input path to folders containing VOC XML files (Folders must have the names of \'train_labels\' and \'test_labels\')')
    parser.add_argument('-o', '--output', type=str, default='', help='Output path for storing the formatted output (Defaults to current working directory)')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Displays more information during the conversion process')
    args = parser.parse_args()
    
    # We need to find the path of each folder containing the train and test labels and we need to generate a pbtext file
    combined_class_labels = []
    for label in ['train_labels', 'test_labels']:
        # Create Path for the input file
        if args.input == "":
            label_path = os.path.join(os.getcwd(), label)
        else:
            label_path = os.path.join(args.input, label)
        # Create Dataframe from XML files and store it as CSV using pandas .to_csv
        xml_dataframe, class_labels = xml_to_csv(label_path)
        xml_dataframe.to_csv(f'{label_path}.csv', index=None)
        combined_class_labels += (item for item in class_labels)
        print(f"[INFO] Finished writing converted file to {label_path}\n")
    # Remove double entries
    combined_class_labels = _remove_double_entry(combined_class_labels)
    combined_class_labels.sort()
    
    # Create a Path for the pbtxt which contains the label mapping
    pbtxt_content = create_pbtxt_content(class_labels, args.verbose)
   
    # Create Path for the label_map file
    if args.output == "":
        label_map_path = os.path.join(os.getcwd(), "label_map_path.pbtxt")
    else:
        label_map_path = os.path.join(args.output, "label_map_path.pbtxt")

    # Save PB Text to pbtxt file
    with open(label_map_path, 'w') as file:
        file.write(pbtxt_content)
        print(f"[INFO] Finished writing pbtxt file to {label_map_path}")

if __name__ == "__main__":
    main()