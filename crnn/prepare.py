import argparse
import yaml
import os
import glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True,
                    help='The config file path.')

parser.add_argument('--dir', type=str, required=True,
                    help='Folder dir')

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)

#Get image and label
images  = glob.glob( f'{args.dir}/*/*.jpg')
labels = [img.split(os.path.sep)[-1].split(".jpg")[0].split("_")[0] for img in images]

#Build Vocab
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

#Write file vocab
with open(config['dataset_builder']['vocab_path'], 'w') as f:
    f.write('\n'.join(list(characters)))

#Create DF image and label
df = pd.DataFrame( 
    {'file_path': images,
     'label': labels
    })

#Set len label
df['length']  = df['label'].str.len()
df = df[df.length < 50]

#Spit train, val
df_train, df_val, = train_test_split(df, test_size=0.05, random_state=42)
#Write file
df_train.to_csv(config['train']['train_csv_path'],index=False)
df_val.to_csv(config['train']['val_csv_path'],index=False)

