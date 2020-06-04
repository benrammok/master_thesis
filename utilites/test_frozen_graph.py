import tensorflow as tf
import glob
import os
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse


class Object_Detection_Model:
    def __init__(self, model_path):
        self.graph = self.load_graph(graph_def_path=model_path)
        self.input_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    @staticmethod
    def load_graph(graph_def_path):
        temp_graph = tf.Graph()
        session = tf.InteractiveSession(graph=temp_graph)
        with temp_graph.as_default():
            with tf.io.gfile.GFile(name=graph_def_path, mode='rb') as serialized_graph:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(serialized_graph.read())
                tf.import_graph_def(graph_def, name='')
        return temp_graph

    def run_detection(self, image_tensor):
        with self.graph.as_default():
            with tf.compat.v1.Session(graph=self.graph) as session:
                (boxes, scores, classes, num_detections) = session.run([self.detection_boxes,
                                                                        self.detection_scores,
                                                                        self.detection_classes,
                                                                        self.num_detections],
                                                                       feed_dict={self.input_tensor: image_tensor})
                return boxes, scores, classes, num_detections

def main():
    # Gets pb file in current directory, chooses the first found instance if multiple are present

    argparser = argparse.ArgumentParser(description="A Python Program for reading in a frozen \
                                                    tensorflow object detection graph and test on an image. \
                                                    (This only works for Frozen Graphs and not TFLite Format)")

    argparser.add_argument('-i', '--image', help="Path pointing to an image file", required=True)
    argparser.add_argument('-m', '--model', help="Path pointing to a Tensorflow Frozen PB Graph File", required=True)
    argparser.add_argument('-l', '--label', help="Path pointing to label file", required=True)
    args = argparser.parse_args()

    # Define an Object Detection Model Class
    object_detection_graph = Object_Detection_Model(args.model)

    # Read in an Image using CV2, Pillow is another alternative
    img = cv2.imread('person.jpg')
    img_np = np.expand_dims(img, axis=0)

    # Read Label map containing the class labels for the detector
    label_map = label_map_util.load_labelmap('label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Call the run_detection method on the img_np image and store the returned predictions
    (boxes, scores, classes, num_detections) = object_detection_graph.run_detection(img_np)
    # Use the vis_util to create the final image with the annotations
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.4
    )

    # Display output
    cv2.imshow('People Detection', cv2.resize(img, (1200, 800)))
    cv2.imwrite('prediction.jpg', img)
    if cv2.waitKey(10000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
