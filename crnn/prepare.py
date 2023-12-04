import argparse
import yaml
import numpy as np
import os
import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True, help="The config file path.")

parser.add_argument("--dir", type=str, required=True, help="Folder dir")

args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)

# Dataset splitting

base_path = args.dir
words_list = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
        words_list.append(line)

np.random.shuffle(words_list)

# We will split the dataset into three subsets with a 90:5:5 ratio (train:validation:test).
split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")


base_image_path = os.path.join(base_path, "words")


def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for i, file_line in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

# we prepare the ground-truth labels.

# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))

print("Maximum length: ", max_len)
print("Vocab size: ", len(characters))

# Check some label samples.
print("Some label samples: ", train_labels_cleaned[:10])

# Now we clean the validation and the test labels as well.


def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)


# Write file vocab
with open(config["dataset_builder"]["vocab_path"], "w") as f:
    f.write("\n".join(list(characters)))

# Create DF image and label
df_train = pd.DataFrame({"file_path": train_img_paths, "label": train_labels_cleaned})
df_val = pd.DataFrame(
    {"file_path": validation_img_paths, "label": validation_labels_cleaned}
)
df_test = pd.DataFrame({"file_path": test_img_paths, "label": test_labels_cleaned})

# Write file
df_train.to_csv(config["train"]["train_csv_path"], index=False)
df_val.to_csv(config["train"]["val_csv_path"], index=False)
df_test.to_csv(config["train"]["test_csv_path"], index=False)
