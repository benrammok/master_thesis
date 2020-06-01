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
"""Example using TF Lite to detect objects in a given image."""

## MODIFIED FROM ORIGINAL: Changed Draw_Boxes to Draw_Box which draws only one box at a time
## Added extra print statements which indicates the number of detections, in the case of people detection, the number of people detected.


import argparse
import time
import csv

from os import listdir
from os.path import isfile, join

from PIL import Image, ImageDraw, ImageFont

import detect
import tflite_runtime.interpreter as tflite
import platform


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
                   outline='red', width=2)
  draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')              

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image folder')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  # Open Image from input argument
  image = Image.open(args.input)
  image = image.convert('RGB')
  #Scale Each Image
  scale = detect.set_input(interpreter, image.size,
                            lambda size: image.resize(size, Image.ANTIALIAS))
  # Call the Interpreter and run the inference
  interpreter.invoke()
  objs = detect.get_output(interpreter, args.threshold, scale)
    
  # Loop over every detected object
  for obj in objs:
    draw_object(ImageDraw.Draw(image), obj, labels)
      
  image.save("prediction.jpg")
  print("Detected {} people in the image".format(len(objs)))


if __name__ == '__main__':
  main()
