import tensorflow as tf
import csv
import pandas as pd
import io
import os
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


flags = tf.compat.v1.app.flags

classes_text_map = ['jade'.encode('utf8'), 'buddy'.encode('utf8')]

image_path = '../dataset/cats/'

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    with tf.gfile.GFile(os.path.join(image_path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(classes_text_map[2-int(row['class_id'])])
        classes.append(int(row['class_id']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    # generate train record
    flags.DEFINE_string('output_path_train', '../dataset/tf/catbot_tf_train.record', 'output path to tensor')
    FLAGS = flags.FLAGS
    writer = tf.io.TFRecordWriter(FLAGS.output_path_train)
    filename = image_path + '/training.csv'
    csvRows = pd.read_csv(filename)

    grouped = split(csvRows, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())

    writer.close()

    # generate test record
    flags.DEFINE_string('output_path_test', '../dataset/tf/catbot_tf_test.record', 'output path to tensor')
    FLAGS = flags.FLAGS
    writer = tf.io.TFRecordWriter(FLAGS.output_path_test)
    filename = image_path + '/validation.csv'
    csvRows = pd.read_csv(filename)

    grouped = split(csvRows, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
