# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A demo script to show to train a segmentation model."""
from absl import app
from absl import logging
from absl import flags
import tensorflow as tf

import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import hparams_config
from tf2 import efficientdet_keras

from pathlib import Path
import json

FLAGS = flags.FLAGS

def define_flags():
  """Define the flags."""
  flags.DEFINE_string('model_name', 'efficientdet-d2', 'Efficientdet Model to use.')
  flags.DEFINE_string('dataset_split_json', default=None, help='Path to xBD dataset split json')
  flags.DEFINE_string('dataset_dir', default=None, help='Path to xBD dataset')
  flags.DEFINE_integer('initial_epoch', default=0, help='0 for fresh training, other values will load the model and continue training')
  flags.DEFINE_integer('max_train_images', default=None, help='Maximum number of train images to use from the dataset')
  flags.DEFINE_integer('batch_size', default=1, help='Batch size for training')

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def get_prepared_data(split_json_file, dataset_dir, split='train', max_data_points=None):

    with tf.io.gfile.GFile(split_json_file, 'rb') as f:
      split_info = json.load(f)
    data_points = []
    for disaster in split_info:
      data_points += split_info[disaster][split]

    num_data_points = len(data_points)
    if max_data_points is None:
      max_data_points = num_data_points
    else:
      max_data_points = min(max_data_points, num_data_points)

    img_paths = list()
    label_paths = list()
    for i in range(max_data_points):
      img_path = data_points[i].replace('labels', 'images') + '_pre_disaster.png'
      label_path = data_points[i].replace('labels', 'targets') + '_pre_disaster_target.png'
      img_paths.append(str(Path(dataset_dir)/img_path))
      label_paths.append(str(Path(dataset_dir)/label_path))

    logging.info(f'Using {max_data_points} {split} images')

    return img_paths, label_paths

def get_loaded_image(img_path, mask_path):
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (768, 768))
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_jpeg(mask, channels=3)
  mask = tf.image.resize(mask, (192, 192), method='nearest')
  mask = mask[:,:,:1]
  return img, mask

def get_randomly_transformed_image(img, mask):
  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
  return img, mask

def get_normalized(img):
  return tf.cast(img, tf.float32) / 255.0

def load_image_train(img_path, mask_path):
  img, mask = get_loaded_image(img_path, mask_path)
  img, mask = get_randomly_transformed_image(img, mask)
  img = get_normalized(img)
  return img, mask

def load_image_val(img_path, mask_path):
  img, mask = get_loaded_image(img_path, mask_path)
  img = get_normalized(img)
  return img, mask

def main(_):

  max_val_images = FLAGS.max_train_images//8 if FLAGS.max_train_images else None

  dataset = dict()
  dataset['train'] = tf.data.Dataset.from_tensor_slices(
      (get_prepared_data(FLAGS.dataset_split_json, FLAGS.dataset_dir, split='train', max_data_points=FLAGS.max_train_images)))
  dataset['val'] = tf.data.Dataset.from_tensor_slices(
      (get_prepared_data(FLAGS.dataset_split_json, FLAGS.dataset_dir, split='val', max_data_points=max_val_images)))

  train_examples = len(dataset['train'])
  steps_per_epoch = train_examples // FLAGS.batch_size

  train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val = dataset['val'].map(load_image_val)

  train_dataset = train.shuffle(10).batch(FLAGS.batch_size).repeat()
  train_dataset = train_dataset.prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE)
  val_dataset = val.batch(FLAGS.batch_size)

  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.heads = ['segmentation']
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((1, config.image_size, config.image_size, 3))
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  if FLAGS.initial_epoch != 0:
    try:
      model.load_weights('dataset/xBD/checkpoints/building_ckpt')
      logging.info('Loaded previously learned weights before resuming the training')
    except Exception as e:
      logging.info('Encoutered exception while loading weights: ', e)

  val_steps = len(dataset['val']) // FLAGS.batch_size

  try:
    logging.info('Training starts')
    model.fit(
      train_dataset,
      epochs=10,
      steps_per_epoch=steps_per_epoch,
      validation_steps=val_steps,
      validation_data=val_dataset,
      callbacks=[])
    logging.info('Training ended successfully')
  except Exception as e:
    logging.info('Encountered exception while training', e)

  try:
    logging.info('Saving the model...')
    model.save_weights('dataset/xBD/checkpoints/building_ckpt')
    logging.info('Saved the model successfully')
  except Exception as e:
    logging.info('Encountered exception while training:', e)


if __name__ == '__main__':
  define_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
