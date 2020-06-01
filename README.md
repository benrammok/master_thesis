# master_thesis
This repository contains code used during the completion of my master thesis at NTNU in Spring 2020.

The project explored the use of Tensorflow- and Darknet-based models to do transfer learning for detection of people.
This has since been evaluated by using Desktop GPU and CPU, in addition the networks have been tested on the Google Coral Dev Board.

Currently only Quantized Tensorflow Models can run on the Google Corals Edge TPU. So for the other implementations they were tested using the Dev Boards CPU.

The requirements.txt file can be used to install the required modules. THIS SHOULD NOT BE EXECUTED ON THE GOOGLE CORAL, for the Coral or other devices such as Raspberry, the training pipeline should not be used.

# Google Coral
To run the code on Google Coral, one need to install the tflite_runtime, the version for your operating system can be found at: https://www.tensorflow.org/lite/guide/python
