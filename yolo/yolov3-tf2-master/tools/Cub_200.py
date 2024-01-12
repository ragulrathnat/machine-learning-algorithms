import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm
import re


flags.DEFINE_string('data_dir', './data/voc2012_raw/VOCdevkit/VOC2012/',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val spit')
flags.DEFINE_string('output_file', './data/voc2012_train.tfrecord', 'outpot dataset')
flags.DEFINE_string('classes', './data/voc2012.names', 'classes file')
flags.DEFINE_string('source','./data/voc2012.names','sources file')
flags.DEFINE_string('file','./data/voc2012.names','image_path')
# flags.DEFINE_string('ann','bounding_box path')

# path = r'C:\Users\Ragul Rathna\Desktop\CUB_200_2011\image_class_labels.txt'

# ann= r'C:\Users\Ragul Rathna\Desktop\CUB_200_2011\bounding_boxes.txt'

# file = r'C:\Users\Ragul Rathna\Desktop\CUB_200_2011\images.txt'

# source = r'C:\Users\Ragul Rathna\Desktop\CUB_200_2011'

def build_example(annotation, class_map,img_path):
    img_path = os.path.join(
        FLAGS.source, 'images', img_path)
    img_raw = open(img_path, 'rb').read()
    # print(img_raw)
    # key = hashlib.sha256(img_raw).hexdigest()

    # width = int(annotation['size']['width'])
    # height = int(annotation['size']['height'])
    width = 416
    height = 416
    annotation = annotation.split()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    obj = annotation
        # difficult = bool(int(obj['difficult']))
    # difficult_obj.append(int(difficult))
    # print(obj[0])
    xmin.append(tf.expand_dims(float(obj[0]) / width))
    ymin.append(tf.expand_dims(float(obj[1]) / height))
    xmax.append(tf.expand_dims(float(obj[2]) / width))
    ymax.append(tf.expand_dims(float(obj[3]) / height))
    classes_text.append(class_map.encode('utf8'))
    # print(class_map.encode('utf8'))
    # classes.append(class_map[])
    # truncated.append(int(obj['truncated']))
    # views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        # 'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[22,22])),
        # 'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[22,22])),
        # 'image/filename': tf.train.Feature(bytes_list=tf.train.Int64List(value=[22,22])),
        # 'image/source_id': tf.train.Feature(bytes_list=tf.train.Int64List(value=[22,22])),
        # 'image/key/sha256': tf.train.Feature(bytes_list=tf.train.Int64List(value=[22,22])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        # 'image/format': tf.train.Feature(bytes_list=tf.train.Int64List(value=[22,22])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        # 'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[22,22])),
        # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=[22,22])),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=[22,22])),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.Int64List(value=[22,22])),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    class_map =open(FLAGS.classes).read().splitlines()
    # logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    # image_list = open(os.path.join(
    #     FLAGS.data_dir, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
    # logging.info("Image list loaded: %d", len(image_list))

    image_list = open(FLAGS.data_dir).read().splitlines()
    # i=0
    img_path = open(FLAGS.file).read().splitlines()
    
    for i in range(len(image_list)):
        # annotation_xml = os.path.join(
        #     FLAGS.data_dir, 'Annotations', name + '.xml')
        # annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        # annotation = parse_xml(annotation_xml)['annotation']
        # print(name)
        # print(i)
        parts = img_path[i].split(' ', 1)
        result = ' '.join(parts[1:])
        tf_example = build_example(image_list[i], class_map[i],result)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
