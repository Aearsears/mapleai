"""
from tf2 object api tutorial!
"""

import os
import pathlib
import cv2
if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')

import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
import contextlib2
import csv

def create_tf_example(image,csvline):
    # TODO(user): Populate the following variables from your example.
    im=cv2.imread(os.path.join("./eval3/",image))
    height = int(im.shape[0]) # Image height
    width = int(im.shape[1]) # Image width
    fp = open(os.path.join("./eval3/",image),'rb')
    # str = tf.image.encode_png(im)
    filename =  image.encode('utf-8') # Filename of the image. Empty if image is not from file
    # encoded_image_data = tf.io.encode_png(tf.convert_to_tensor(im,dtype=tf.uint8)) # Encoded image bytes
    encoded_image_data = fp.read() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'
    xmins = [] # List of normalized left x coordinates in boundingbox (1 per box)
    xmaxs = [] # List of normalized right x coordinates in boundingbox
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box(1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in boundingbox
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1per box)
    classes = [] # List of integer class id of bounding box (1 perbox)
    for line in csvline:
        xmins.append(int(line[1])/width) # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs.append(int(line[3])/width) # List of normalized right x coordinates in bounding box
                    # (1 per box)
        ymins.append(int(line[2])/height) # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs.append(int(line[4])/height) # List of normalized bottom y coordinates in bounding box
                    # (1 per box)
        classes_text.append(line[5].encode('utf-8')) # List of string class name of bounding box (1 per box)

        if line[5]=="mob":
            classes.append(1)
        else: #else a player
            classes.append(2)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_example



def main():
    IMAGES_PATH ='path'
    num_shards=10
    OUTPUT_PATH='path'
    CSV_PATH='path'


    with contextlib2.ExitStack() as tf_record_close_stack, open(CSV_PATH,'r',newline='') as f:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, OUTPUT_PATH, num_shards)
        reader = csv.reader(f)  
        #iterate through entire csv file of the bounding boxses
        labels=[]

        for index,row in enumerate(reader,start=0):
            #new picture, so add the row
            if len(labels)==0:
                labels.append(row)
            else:
                #if the first labels file name is the same as the one already in the list, then add it
                if(labels[0][0]==row[0]):
                    labels.append(row)
                #if not, then we ahve reached the end of the bounding boxes for an image, and then we create the example
                else:
                    tf_example = create_tf_example(labels[0][0].split("\\")[1],labels)
                    output_shard_index = index % num_shards
                    print("seralizing image "+str(labels[0][0].split("\\")[1]))
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                    labels.clear()
                    labels.append(row)

            

if __name__ == '__main__':
    main()



