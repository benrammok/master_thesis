import argparse
import re


def main():
    
    parser = argparse.ArgumentParser('XML to CSV Formatter', 'A Python Program for converting a set of XML files in VOC format to CSV format')
    parser.add_argument('-i', '--input', type=str, default='', help='Input path to YOLOv3 Config file')
    parser.add_argument('-n', '--names_file', type=str, default='', help='Input path to names file containing the individual classes')
    parser.add_argument('-t', '--tiny', default=False, action='store_true', help='Configure the config file for YOLOv3-Tiny instead of YOLOv3')
    args = parser.parse_args()
	
    num_classes = 0
    num_filters = 0

    def YOLOv3_filter_size(n_classes):
        return (n_classes + 5) * 3

    with open(args.names_file, mode='r') as f:
        print("Opened file")
        for line in f:
          num_classes += 1	
    num_filters = YOLOv3_filter_size(num_classes)
    # These are standard 
    classes_lines = []

    #print(f"Found {num_classes} classes")
    filter_lines = []

    with open(args.input, mode='r') as f:
         lines = f.readlines()
         
    for index, line in enumerate(lines):
        if "linear" in line:
           # Filter is usually always directly above activation=linear
           # This needs improvement since it is hardcoded
           filter_lines.append(index - 1)
        if "classes" in line:
           classes_lines.append(index)
    
    for index in filter_lines:
        lines[index] = f"filters={num_filters}\n"

    for index in classes_lines:
        lines[index] = f"classes={num_classes}\n"
    
    with open(args.input, mode='w') as f:
        f.writelines(lines)              

if __name__ == "__main__":
    main()
