import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import label_map_util as lm
from load_model import detect_fn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import requests
import warnings

# Loading the label_map
category_index = lm.create_category_index_from_labelmap(
    "label_map.pbtxt", use_display_name=True)

detection_dir = "images/saved_detections/"
test_dir = "images/test_img/"


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def get_img_path(img_url):

    image_path = test_dir + "test_" + \
        str(len(os.listdir(test_dir))) + ".png"  # creating image path

    if os.path.isfile(image_url):  # if image exists locally
        f = Image.open(image_url)
        f.save(image_path)

    else:  # if using url to download image
        # getting image data from url
        url_data = requests.get(image_url).content
        f = open(image_path, 'wb')
        f.write(url_data)
        f.close()

    return image_path


warnings.filterwarnings('ignore')

while True:
    image_url = input("Enter image url/path for detection or 'no' to exit: ")

    if image_url in ["no", "No", "NO"]:
        break

    image_path = get_img_path(image_url)

    print('Running inference for {}... '.format(image_path), end='')

    # there was a problem with one of the images
    try:
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
    except ValueError:
        print("There was a problem with the image")
        continue

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        line_thickness=8,
        min_score_thresh=.4,
        agnostic_mode=False)

    img_name = detection_dir + "detection_" + \
        str(len(os.listdir(detection_dir))) + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np_with_detections)
    plt.axis('off')
    plt.savefig(img_name)
    detect = Image.open(img_name)
    detect.show()
    print('Done')
