import argparse
import pprint
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tensorflow import keras

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True,
                    help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True,
                    help='The path to save the models, logs, etc.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

args.save_dir.mkdir(exist_ok=True)

shutil.copy(args.config, args.save_dir / args.config.name)

batch_size = config['batch_size_per_replica']

dataset_builder = DatasetBuilder(**config['dataset_builder'])

model = build_model(dataset_builder.num_classes,
                    #weight=config.get('weight'),
                    img_width=config['dataset_builder']['img_width'],
                    img_height=config['dataset_builder']['img_height'],
                    channel=config['dataset_builder']['channel']
                   )

model.compile(optimizer=keras.optimizers.Adam(),
              loss=CTCLoss(),
              metrics=[SequenceAccuracy()])

model.summary()

train_df = pd.read_csv(config['train_csv_path']).astype(str)
val_df = pd.read_csv(config['val_csv_path']).astype(str)
test_df = pd.read_csv(config['test_csv_path']).astype(str)

train_ds = dataset_builder(train_df, batch_size, shuffle=True)
val_ds = dataset_builder(val_df, batch_size, cache=True)
test_ds = dataset_builder(test_df, batch_size)

# model_prefix = '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
# model_path = f'{args.save_dir}/{model_prefix}.h5'
model_path = f'{args.save_dir}/best'

callbacks = [
    # keras.callbacks.ModelCheckpoint(model_path,
    #                                 save_weights_only=True),

    keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs',
                                **config['tensorboard']),
    keras.callbacks.EarlyStopping(monitor="val_loss",**config['earlystopping']),

]

history = model.fit(train_ds, epochs=config['epochs'],
                    callbacks=callbacks,
                    verbose=config['fit_verbose'],
                    validation_data=val_ds)

print('Accuracy on test set:')
model.evaluate(test_ds)
model.save(model_path)
