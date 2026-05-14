from datasets import load_dataset
import os

dataset = load_dataset(
    "Samarth0710/bharatanatyam-mudra-dataset"
)

output_dir = "bharatanatyam_dataset"

os.makedirs(output_dir, exist_ok=True)

for split in dataset.keys():

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, item in enumerate(dataset[split]):

        image = item['image']
        label = item['label']

        label_dir = os.path.join(split_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        image.save(f"{label_dir}/{i}.jpg")

print("Dataset saved successfully!")