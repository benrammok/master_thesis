# Inspired and modified from: https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d
import tensorflow as tf
from object_detection.utils import dataset_util
import pandas as pd
from PIL import Image
from collections import namedtuple
from tqdm import tqdm
import argparse
import os
import io
from object_detection.utils import label_map_util

# Default Values when generated by xml_to_csv.py

DEFAULT_LABEL_MAP_NAME = "label_map_path.pbtxt"


# This allows for conversion between class given in text format and the actual id specified in the label map
def label_to_int(label, label_dict):
    if label in label_dict:
      return label_dict[label]
    else:
      return None

# Helper function to split up the data

def split_data(data, group):
    data_tp = namedtuple('data', ['filename', 'object'])
    data_group = data.groupby(group)
    return [data_tp(filename, data_group.get_group(i)) for filename, i in zip(data_group.groups.keys(), data_group.groups)]

def generate_tfrecord(group, path, label_dict):
  # Open image file
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as gf:
        encoded_jpg = gf.read()
    
    jpg_io = io.BytesIO(encoded_jpg)
    img = Image.open(jpg_io)

    image_width, image_height = img.size
    filename = group.filename.encode('utf8')
    img_format = b'jpg'

    # For the data in regards to the image
    # we need to store the minimum and maximum x and y positions of the bounding boxes
    bx_min = []
    bx_max = []
    by_min = []
    by_max = []

    classes = []
    classes_texts = []

    # Itterate over every row of data in our given data in the dataset
    for i, row in group.object.iterrows():
      bx_min.append(row['xmin'] / image_width)
      bx_max.append(row['xmax'] / image_width)
      by_min.append(row['ymin'] / image_height)
      by_max.append(row['ymax'] / image_height)
      # Add Class text and the converted labels to
      classes_texts.append(row['class'].encode('utf8'))
      #print(row['class'], " Id from label_to_int: ", label_to_int(row['class'], label_dict))
      classes.append(label_to_int(row['class'], label_dict))




    # TF record example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'img/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(bx_min),
        'image/object/bbox/xmax': dataset_util.float_list_feature(bx_max),
        'image/object/bbox/ymin': dataset_util.float_list_feature(by_min),
        'image/object/bbox/ymax': dataset_util.float_list_feature(by_max),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_texts),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    })
    )
    return tf_example



def main():
    
    parser = argparse.ArgumentParser('Generate TFRECORDS', 'A Python Program for generating TFRECORDS from CSV Files')
    parser.add_argument('-i', '--input', type=str, default='', help='Input path to Image Directory')
    parser.add_argument('-l', '--label', type=str, default='', help='Label Map class encodings (Defaults to current working directory)')
    parser.add_argument('-c', '--csv', type=str, default='', help='Input path to CSV file directory (Leave blank to assign the current working directory)')
    parser.add_argument('-o', '--output', type=str, default='', help='Output path for storing the TFRecord files')
    args = parser.parse_args()
    
    if args.label == '':
          label_map_path = os.path.join(os.getcwd(), DEFAULT_LABEL_MAP_NAME)
    else:
          label_map_path = os.path.join(args.label)

    class_dict = {}

    # Open our Label File using Object Detection Util
    lb_proto = label_map_util.load_labelmap(label_map_path)
    label_dict = label_map_util.get_label_map_dict(lb_proto)

    print(label_dict)

    for csv in ['train_labels', 'test_labels']:
      if args.csv == '':
        csv_file = pd.read_csv(os.path.join(os.getcwd(), csv + '.csv'))
      else:
        csv_file = pd.read_csv(os.path.join(args.csv, csv + '.csv'))   

      # If no path is given, assume that the images is located under current
      # working directory and under the subfolder images, this can potentially be changed by adding another argument
      if args.input == '':
        path = os.path.join(os.getcwd(), 'images')
      else:
        path = args.input

      # Create a TF Record Writer
      tf_writer = tf.io.TFRecordWriter(os.path.join(os.getcwd(), csv + '.record'))

      split_files = split_data(csv_file, 'filename')
    
      for group in tqdm(split_files):
          tf_example = generate_tfrecord(group, path, label_dict)
          tf_writer.write(tf_example.SerializeToString())
      tf_writer.close()
      
      if args.output == '':
        out_path = os.path.join(os.getcwd(), csv + '.record')
      else:
        out_path = os.path.join(args.output, csv + '.record')  
      print('Finished writing TF Record at: {}'.format(out_path))


if __name__ == "__main__":
    main()