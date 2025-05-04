import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = Path("animals")
test_dir = Path("animal_test")

# Create test directory if not exists
test_dir.mkdir(exist_ok=True)

# Set the test split ratio
test_ratio = 0.2

# Loop through each animal folder
for animal_class in os.listdir(source_dir):
    class_path = source_dir / animal_class

    if not class_path.is_dir():
        continue

    images = list(class_path.glob("*"))
    random.shuffle(images)

    # Split
    num_test = int(len(images) * test_ratio)
    test_images = images[:num_test]

    # Create corresponding directory in test folder
    target_class_path = test_dir / animal_class
    target_class_path.mkdir(parents=True, exist_ok=True)

    # Move/copy images to test folder
    for img_path in test_images:
        shutil.copy(img_path, target_class_path / img_path.name)

print("✅ Test dataset created in 'animal_test'")
