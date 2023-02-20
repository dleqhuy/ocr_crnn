import argparse
import yaml
import os
import glob
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True,
                    help='The config file path.')

parser.add_argument('--dir', type=str, required=True,
                    help='Folder dir')

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)

images  = glob.glob( f'{args.dir}/*.jpg')
labels = [img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0] for img in images]

characters = set(char for label in labels for char in label)
characters = sorted(list(characters))


with open(config['dataset_builder']['vocab_path'], 'w') as f:
    f.write('\n'.join(list(characters)))
    
df = pd.DataFrame( 
    {'file_path': images,
     'label': labels
    })

df['length']  = df['label'].str.len()
df = df[df.length < 50]

df.to_csv(config['train']['train_csv_path'],index=False)
