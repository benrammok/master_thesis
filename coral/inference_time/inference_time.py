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
"""Modified Detection Example for recording inference time during continous processing of dataset"""

# This is a modified sample of the original detect_image.py supplied in the Coral Github Repo
# It has been repurposed to generate a csv file containing recorded inference time when running a neural network.
# Since this is a repurposed program the original license is given above.


import argparse
import time
import csv

import os

from PIL import Image, ImageDraw, ImageFont

import detect
import tflite_runtime.interpreter as tflite
import platform

from glob import glob


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

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=False,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input_folder', required=True,
                      help='File path of image folder')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects.')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  
  running_average_time = 0
  image_path = os.path.join(os.path.expanduser(args.input_folder), "*.jpg")
  print("Number of images " +  str(len(glob(image_path))))
  # Run Once over the entire dataset
  with open('inference_result' + '.csv', mode='w') as accuracy_f:
    csv_writer = csv.writer(accuracy_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["Current Run: ", "Inference Time: "])
    current_run = 0
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
        running_average_time += inference_time
        csv_writer.writerow([current_run, inference_time])
        print("Done with Image {}\n".format(current_run))
        # Run to 400 images to be consistent with testing on Yolo
        if current_run == 400:
              break
    # Add the average inference time at the end of the csv file
    if current_run != 0:
      csv_writer.writerow(["Average Inference Time: ", running_average_time / current_run])

if __name__ == '__main__':
  main()
