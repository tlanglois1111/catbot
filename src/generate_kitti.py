import tensorflow as tf
import pandas as pd
import io
import os
from PIL import Image
from collections import namedtuple

classes_text_map = ['jade', 'buddy']
image_path = '../dataset/cats/'


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group):
    with tf.gfile.GFile(os.path.join(image_path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()

    kittis = []

    for index, row in group.object.iterrows():
        kitti_line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".\
            format(classes_text_map[2-int(row['class_id'])],
                   0.0,
                   0,
                   0.0,
                   "{0:.2f}".format(row['xmin']),
                   "{0:.2f}".format(row['ymin']),
                   "{0:.2f}".format(row['xmax']),
                   "{0:.2f}".format(row['ymax']),
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0)
        kittis.append(kitti_line)
    return encoded_jpg, kittis


def main(_):

    # generate train record
    filename = image_path + '/training.csv'
    csv_rows = pd.read_csv(filename)

    kitti_file = []
    grouped = split(csv_rows, 'filename')
    for group in grouped:
        image, kitti_lines = create_tf_example(group)
        for kitti_line in kitti_lines:
            kitti_file.append({'image': image, 'line': kitti_line})

    for i, kitti_map in enumerate(kitti_file):
        with open('../dataset/kitti/train/labels/{}.txt'.format(i), mode='wt', encoding='utf-8') as myfile:
            for kitti_line in kitti_map['line']:
                myfile.write(kitti_line)
            myfile.close()
        with tf.gfile.GFile(os.path.join('../dataset/kitti/train/images/', '{}.jpg'.format(i)), 'wb') as fid:
            fid.write(kitti_map['image'])
            fid.close()

    # generate test record
    filename = image_path + '/validation.csv'
    csv_rows = pd.read_csv(filename)

    kitti_file = []
    grouped = split(csv_rows, 'filename')
    for group in grouped:
        image, kitti_lines = create_tf_example(group)
        for kitti_line in kitti_lines:
            kitti_file.append({'image': image, 'line': kitti_line})

    for i, kitti_map in enumerate(kitti_file):
        with open('../dataset/kitti/val/labels/{}.txt'.format(i), mode='wt', encoding='utf-8') as myfile:
            for kitti_line in kitti_map['line']:
                myfile.write(kitti_line)
            myfile.close()
        with tf.gfile.GFile(os.path.join('../dataset/kitti/val/images/', '{}.jpg'.format(i)), 'wb') as fid:
            fid.write(kitti_map['image'])
            fid.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
