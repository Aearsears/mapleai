"""
Based on https://github.com/EricThomson/tfrecord-view 's script
"""

import cv2
import numpy as np
import tensorflow as tf

def cv_bbox(image, bbox, color = (255, 255, 255), line_width = 2):
    """
    use opencv to add bbox to an image
    assumes bbox is in standard form x1 y1 x2 y2
    """

    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
    return

def view_records(file_path,class_labels):
    dataset = tf.data.TFRecordDataset([file_path])
    

if __name__ == '__main__':
    #path to the file
    file_path ='path'
    dataset = tf.data.TFRecordDataset([file_path])

    #  tf_example = tf.train.Example(features=tf.train.Features(feature={
    #         'image/height': dataset_util.int64_feature(height),
    #         'image/width': dataset_util.int64_feature(width),
    #         'image/filename': dataset_util.bytes_feature(filename),
    #         'image/source_id': dataset_util.bytes_feature(filename),
    #         'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    #         'image/format': dataset_util.bytes_feature(image_format),
    #         'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    #         'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    #         'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    #         'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    #         'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    #         'image/object/class/label': dataset_util.int64_list_feature(classes),
    #     }))
    #%%
    feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }

    def print_box(boxes):
        for b in boxes:
            print()

    def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)

    # height = parsed_dataset['image/height']
    num_records = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    class_labels =  {"mob" : 1, "player": 2 }

    for parsed_record in parsed_dataset:
        # print(tf.sparse.to_dense(parsed_record['image/object/bbox/xmin'],default_value=-1))

        encoded_image = parsed_record['image/encoded']
        image_np = tf.image.decode_image(encoded_image, channels=3).numpy()

        h = parsed_record['image/height'].numpy()
        w = parsed_record['image/width'].numpy()
        # print(h,w)

        labels =  tf.sparse.to_dense(parsed_record['image/object/class/label'], default_value=0).numpy()
        x1norm =  tf.sparse.to_dense( parsed_record['image/object/bbox/xmin'], default_value=0).numpy()
        x2norm =  tf.sparse.to_dense( parsed_record['image/object/bbox/xmax'], default_value=0).numpy()
        y1norm =  tf.sparse.to_dense( parsed_record['image/object/bbox/ymin'], default_value=0).numpy()
        y2norm =  tf.sparse.to_dense( parsed_record['image/object/bbox/ymax'], default_value=0).numpy()

        num_bboxes = len(labels)

        #% Process and display image
        image_copy = image_np.copy()
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        if num_bboxes == 0:
            print("no bounding boxes!")

        elif num_bboxes > 0:
            x1 = np.int64(x1norm*w)
            x2 = np.int64(x2norm*w)
            y1 = np.int64(y1norm*h)
            y2 = np.int64(y2norm*h)
            for bbox_ind in range(num_bboxes):
                    bbox = (x1[bbox_ind], y1[bbox_ind], x2[bbox_ind], y2[bbox_ind])
                    label_name = list(class_labels.keys())[list(class_labels.values()).index(labels[bbox_ind])]
                    label_position = (bbox[0] + 5, bbox[1] + 20)
                    cv_bbox(image_rgb, bbox, color = (250, 250, 150), line_width = 2)
                    cv2.putText(image_rgb,
                                label_name,
                                label_position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (10, 10, 255), 2); #scale, color, thickness
        cv2.imshow("bb data", image_rgb)
        k = cv2.waitKey()
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        elif k == ord('n'):
            cv2.destroyAllWindows()
            continue
    