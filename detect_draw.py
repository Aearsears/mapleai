# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:57:25 2020

"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import csv
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import threading
from random import choice

from grab_screen import grab_screen,set_window
from Vision import getHP, getMP,getEXP

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import character_functions
import keyboard

lock = threading.Lock()

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list


def capture_screen():
    """
    Main function loop that captures the screen information
    """
    #countdown before the function starts
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        last_time = time.time()
    
    #code to initialize to model. From TF2 object api tutorial!
    pipeline_config = 'path'
    model_dir = 'path'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
      model_config=model_config, is_training=False)

    # Restore model from the checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    detect_fn = get_model_detection_function(detection_model)

    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

   # infowindow = create_info_screen()
    
   # infowindow.update()
    # sets the maplestory window in focus and in 0,0 position on screen
    set_window("Maplestory")

    randommovement = [keyboard.moveLeft,keyboard.moveRight,keyboard.jumpoffladder]
    randomtele = [keyboard.teleright,keyboard.teleleft]

    lock.acquire()
    while True:
        lock.acquire()
        choice(randomtele)()
        time.sleep(1)
        screen = grab_screen(process="MapleStory")
        #screen = grab_screen(region=(3,25,803,625))
        #newscreen= screen[30:,:,:]
        #screen = grab_process("Maplestory")

        #convert numpy array to tensor
        input_tensor = tf.convert_to_tensor(
        np.expand_dims(screen, 0), dtype=tf.float32)
        #predict it
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        #filter out
        boxes,scores,classes,isTherePlayer = character_functions.filter(detections)

        #if can't find the character then skip the frame and go next
        if not isTherePlayer:
          choice(randommovement)()
          time.sleep(0.5)
          continue
        

        label_id_offset = 1
        image_np_with_detections = screen.copy()
        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
          keypoints = detections['detection_keypoints'][0].numpy()
          keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
        #draw it with the filtered boxes that the computer sees
        viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          boxes,
          classes,
          scores,
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,
          min_score_thresh=0.3,
          agnostic_mode=False,
          keypoints=keypoints,
          keypoint_scores=keypoint_scores,
          keypoint_edges=get_keypoint_tuples(configs['eval_config']))

        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        cv2.imshow('window', image_np_with_detections)
        #automatic input into BGR, so need to convert to RGB
        
        #feed boxes and prediction into function that will move character
        character_functions.attack_mob(boxes,classes)
        #wait one seconds before getting next frame and predicting
        lock.release()
        time.sleep(3)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def get_model_detection_function(model):
  """Get a tf.function for detection. From tf2 object api!"""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn



def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.From tf2 object api!

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def predict_oneimage():
    pipeline_config = 'path'
    model_dir = 'path'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
      model_config=model_config, is_training=False)

    # Restore model from the checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    detect_fn = get_model_detection_function(detection_model)

    image_dir = 'ss/'
    image_path = os.path.join(image_dir, 'ss_63.jpg')
    image_np = load_image_into_numpy_array(image_path)
    # image_np = cv2.imread(image_path)

    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
      keypoints = detections['detection_keypoints'][0].numpy()
      keypoint_scores = detections['detection_keypoint_scores'][0].numpy()


    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,
          min_score_thresh=0.3,
          agnostic_mode=False,
          keypoints=keypoints,
          keypoint_scores=keypoint_scores,
          keypoint_edges=get_keypoint_tuples(configs['eval_config']))

    matplotlib.image.imsave('newmodel.png',image_np_with_detections)
    # cv2.imwrite('newmodel.png',image_np_with_detections)

    # plt.figure(figsize=(12,16))
    # plt.imshow(image_np_with_detections)
    # plt.show()



def buffthr():
  lock.acquire()
  print("Buffing!")
  keyboard.buff()
  lock.release()
  threading.Timer(65,buffthr).start()

def ccthr():
  lock.acquire()
  print("CC'ing!")
  keyboard.cc()
  time.sleep(1)
  keyboard.pressg()
  lock.release()
  threading.Timer(90,ccthr).start()
  

#main()
if __name__ == "__main__":
  #main loop detect screen thread
  x=threading.Thread(target=capture_screen)
  x.start()
  #autocc thread
  # kill= threading.Event()
  autobuffthread = threading.Thread(target=buffthr)
  autoccthread = threading.Thread(target=ccthr)
  autobuffthread.start()
  autoccthread.start()

  # capture_screen()
  #predict_oneimage()