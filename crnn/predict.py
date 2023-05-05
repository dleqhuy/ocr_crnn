import argparse
from pathlib import Path

import glob
import yaml
import tensorflow as tf

from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from dataset_factory import DatasetBuilder
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--weight', type=str, required=True, default='',
                    help='The saved weight path.')
parser.add_argument('--post', type=str, help='Post processing.')
parser.add_argument('--images', type=str, required=True, 
                    help='Image file or folder path.')

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

dataset_builder = DatasetBuilder(**config)

model = build_model(dataset_builder.num_classes,
                    weight=config.get('weight'),
                    img_width=config['img_width'],
                    img_height=config['img_height'],
                    channel=config['channel']
                   )

if args.post == 'greedy':
    postprocess = CTCGreedyDecoder(config['vocab_path'])
elif args.post == 'beam_search':
    postprocess = CTCBeamSearchDecoder(config['vocab_path'])
else:
    postprocess = None

model.load_weights(args.weight)

def read_img_and_resize(path, img_width, img_height, channel):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=channel)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])

    return img

img_paths = glob.glob(f'{args.images}/*.jpg')

for img_path in img_paths:
    img = read_img_and_resize(str(img_path), config['img_width'],  config['img_height'], config['channel'])
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    
    y_pred, probability = postprocess.call(outputs)
    
    print(f'Path: {img_path}, y_pred: {y_pred.numpy().astype(str)}, '
          f'probability: {probability.numpy()}')