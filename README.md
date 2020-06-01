# master_thesis
This repository contains code used during the completion of my master thesis at NTNU in Spring 2020.

The project explored the use of Tensorflow- and Darknet-based models to do transfer learning for detection of people.
This has since been evaluated by using Desktop GPU and CPU, in addition the networks have been tested on the Google Coral Dev Board.

Currently only Quantized Tensorflow Models can run on the Google Corals Edge TPU. So for the other implementations they were tested using the Dev Boards CPU.

# Google Coral
To run the code on Google Coral, one need to install the tflite_runtime, the version for your operating system can be found at: https://www.tensorflow.org/lite/guide/python
