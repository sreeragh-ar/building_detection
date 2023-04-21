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
import json
from pathlib import Path


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

DATASET_SPLIT_JSON = 'dataset/xBD/custom_all_disaster_splits.json'
DATASET_DIR = '../../building-damage-assessment/dataset'
OUTPUT_DIR = 'dataset/xBD/predictions'
MODEL_INPUT_SIZE = 768

DATASET_DIR = Path(DATASET_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main(_):
  config = hparams_config.get_efficientdet_config('efficientdet-d2')
  config.heads = ['segmentation']
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3))

  try:
    print('Loading the model weights...')
    model.load_weights('dataset/xBD/checkpoints/building_ckpt')
    print('Loaded the model weights successfully')
  except Exception as e:
    print('Encountered exception while loading the model:', e)
    sys.exit(0)

  with tf.io.gfile.GFile(DATASET_SPLIT_JSON, 'rb') as f:
    split_info = json.load(f)
    data_points = list()
    for disaster in split_info:
        data_points += split_info[disaster]['val']

  count = 0
  n_data_points = len(data_points)

  for i,data_point in enumerate(data_points):
      try:
        count += 1
        if count > 15:
          break
        print(f'Processing image {count}/{n_data_points}')
        img_path = data_point.replace('labels', 'images') + '_pre_disaster.png'
        img_path = str(DATASET_DIR/img_path)

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        original_height, original_width, _ = img.shape

        img = tf.image.resize(img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.convert_to_tensor([img])
        pred = create_mask(model(img, False))[0]

        pred = tf.image.resize(pred, (original_height, original_width), method='nearest')
        pred_file = f'{Path(data_point).name}.png'
        pred_path = OUTPUT_DIR/pred_file
        tf.keras.preprocessing.image.save_img(pred_path, pred,  scale=False)
      except Exception as e:
        print(f'Error while processing {data_point}')
        print(e)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  app.run(main)