import argparse
import subprocess
import sys
import os

def main():
    argument_parser = argparse.ArgumentParser(description="A Python Program for doing tranfer training on neural network models")
    argument_parser.add_argument('-d', '--det_dir', type=str, default="", help="Path to tensorflow models object detection directory (contains object_detection api, etc)")
    argument_parser.add_argument('-t', '--train_dir', type=str, default="", help="Input path to training directory")
    argument_parser.add_argument('-c', '--config', type=str, default="", help="Path to config file")
    argument_parser.add_argument('-l', '--logtostderr', help="Turn on logging to stderr")
    args = argument_parser.parse_args()

    if args.logtostderr:
        should_log = "--logtostderr"
    else:
        should_log = ""

    if args.train_dir == "":
        train_path = os.path.join(os.getcwd(), 'training')
    else:
        train_path = args.train_dir

    if args.config == "":
        config_path = os.path.join(os.getcwd(), 'pretrained_model/pipeline.config')
    else:
        config_path = args.config

    if args.det_dir == "":
        det_path = os.path.join(os.getcwd(), 'models/research/object_detection', 'model_main.py')
    else:
        det_path = args.train_dir


    # Create a subprocess using model_main.py in object detection folder to train a neural net model
	
    print(det_path, train_path, os.getcwd())

    # Running subprocess.run does not work in Python 2, this should force people to use Python 3
    subprocess.run(
        ["python3",
        det_path,
        "--pipeline_config_path=" + config_path,
	"--model_dir=" + train_path,
        should_log
        ])

if __name__ == "__main__":
   main()
