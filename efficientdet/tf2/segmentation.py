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

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def get_prepared_data(split_json_file, dataset_dir, split='train'):
    with tf.io.gfile.GFile(split_json_file, 'rb') as f:
        split_info = json.load(f)
    data_points = []
    for disaster in split_info:
        data_points += split_info[disaster][split]

    count = 0
    img_paths = list()
    label_paths = list()
    for data_point in data_points:
      img_path = data_point.replace('labels', 'images') + '_pre_disaster.png'
      label_path = data_point.replace('labels', 'targets') + '_pre_disaster_target.png'
      img_paths.append(str(Path(dataset_dir)/img_path))
      label_paths.append(str(Path(dataset_dir)/label_path))
      count += 1

    print('#########################', 'Found',count,'images' )
    return img_paths[:500], label_paths[:500]

def custom_load_image_train(img_path, mask_path):
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (768, 768))
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_jpeg(mask, channels=3)
  mask = tf.image.resize(mask, (192, 192), method='nearest')
  mask = mask[:,:,:1]

  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)

  img = tf.cast(img, tf.float32) / 255.0

  return img, mask

def custom_load_image_val(img_path, mask_path):
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (768, 768))
  mask = tf.io.read_file(mask_path)
  mask = tf.image.decode_jpeg(mask, channels=3)
  mask = tf.image.resize(mask, (192, 192), method='nearest')
  mask = mask[:,:,:1]
  img = tf.cast(img, tf.float32) / 255.0
  return img, mask

def main(_):
  DATASET_SPLIT_JSON = 'dataset/xBD/custom_all_disaster_splits.json'
  DATASET_DIR = '../../building-damage-assessment/dataset'
  initial_epoch = 0

  dataset = dict()
  dataset['train'] = tf.data.Dataset.from_tensor_slices(
      (get_prepared_data(DATASET_SPLIT_JSON, DATASET_DIR, split='train')))
  dataset['val'] = tf.data.Dataset.from_tensor_slices(
      (get_prepared_data(DATASET_SPLIT_JSON, DATASET_DIR, split='val')))

  train_examples = len(dataset['train'])
  batch_size = 1
  steps_per_epoch = train_examples // batch_size

  train = dataset['train'].map(custom_load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val = dataset['val'].map(custom_load_image_val)

  train_dataset = train.shuffle(10).batch(batch_size).repeat()
  train_dataset = train_dataset.prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE)
  val_dataset = val.batch(batch_size)

  config = hparams_config.get_efficientdet_config('efficientdet-d2')
  config.heads = ['segmentation']
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((1, 768, 768, 3))
  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  if initial_epoch != 0:
    try:
      model.load_weights('dataset/xBD/checkpoints/building_ckpt')
      print('Loaded previously learned weights before resuming the training')
    except Exception as e:
      print('Encoutered exception while loading weights: ', e)

  val_steps = len(dataset['val']) // batch_size

  try:
    print('Training starts')
    model.fit(
      train_dataset,
      epochs=10,
      steps_per_epoch=steps_per_epoch,
      validation_steps=val_steps,
      validation_data=val_dataset,
      callbacks=[])
    print('Training ended successfully')
  except Exception as e:
    print('Encountered exception while training', e)

  try:
    print('Saving the model...')
    model.save_weights('dataset/xBD/checkpoints/building_ckpt')
    print('Saved the model successfully')
  except Exception as e:
    print('Encountered exception while training:', e)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  app.run(main)
