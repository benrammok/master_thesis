# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified Detection Example for recording power consumption during continous processing"""

## MODIFIED FROM ORIGINAL: Changed Draw_Boxes to Draw_Box which draws only one box at a time
## Added extra print statements which indicates the number of the current image, a FPS counter, average FPS.
## This program takes a set of images given in an image path and constantly loads the CPU / TPU which can be used
## in conjunction with a USB meter to get approximations on the power consumed by the neural networks.

import argparse
import time
import csv

import os

from PIL import Image, ImageDraw, ImageFont

import detect
import tflite_runtime.interpreter as tflite
import platform

from glob import glob

import psutil

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def draw_object(draw, obj, labels):
  """Draws the bounding box and label for one object."""
  bbox = obj.bbox
  draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
  draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')              


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=False,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input_folder', required=True,
                      help='File path of image folder')
  parser.add_argument('-o', '--output', required=False,
                      help='File path for the resultant images with annotations')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  

  image_path = os.path.join(os.path.expanduser(args.input_folder), "*.jpg")
  
  # Run Indefinitly to load the TPU or CPU constantly
  while True:
    current_run = 0
    acumulated_fps = 0
    for image_file in glob(image_path):
      # Open Image using Context Manager to avoid problems with memory
      with Image.open(image_file) as image:
        # Images MUST be converted to RGB, any other mode causes a bug in the supplied detect.py
        converted_img = image.convert('RGB')
        
        # Scale the Image
        scale = detect.set_input(interpreter, converted_img.size,
                                lambda size: converted_img.resize(size, Image.ANTIALIAS))
        # Call the Interpreter and run the inference
        start_time = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start_time

        objs = detect.get_output(interpreter, args.threshold, scale)
      
        current_run += 1
        acumulated_fps += (1.0 / inference_time)
        print("Done with Image {}, Number of Objects {}, Current CPU Utilization: {}, FPS: {}, Running Avg FPS: {}\n".format(current_run, len(objs), psutil.cpu_percent(), 1.0 / inference_time, acumulated_fps/current_run))

if __name__ == '__main__':
  main()
