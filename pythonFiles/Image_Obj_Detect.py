import numpy as np
import re
import os
import sys
import tensorflow as tf
sys.path.append("D:\\Tensorflow_Files\\models\\research\\")
#sys.path.append("C:\\Tensorflow_Files\\models\\research\\object_detection\\utils")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


def valid_path(path):
    return bool(re.search(r"(http(s?):)|([/|.|\w|\s])*\.(?:jpg|gif|png)", path))


while True:
    file_path = input('Please provide a full html file path: ')
    isExists = os.path.exists(file_path)
    if isExists:
        img = cv2.imread(file_path)

        # What model to download.
        MODEL_NAME = "D:/Tensorflow_Files/models/research/object_detection/inference_graph"

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join("D:\\Tensorflow_Files\\models\\research\\object_detection\\training", "labelmap.pbtxt")

        NUM_CLASSES = 3

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)


        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)


        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)

        # In[10]:

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    #      ret, image_np = cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(img, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        img,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4)

#                    cv2.imshow('object detection', cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))
                    cv2.imshow('Object Detection', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break
        break
    else:
        print('Incorrect file path name!!!')


