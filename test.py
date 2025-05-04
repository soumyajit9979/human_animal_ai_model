import os
from PIL import Image, ImageDraw, ImageFont
import shutil
import torch
import argparse
from PIL import Image
import clip
from pathlib import Path
from siglip_finetune.animal_classification import pred_img
from transformers import AutoImageProcessor, SiglipForImageClassification
import json

model_name_siglip = "Soumyajit9979/animal-siglip-classification"
model_siglip = SiglipForImageClassification.from_pretrained(model_name_siglip)
processor_siglip = AutoImageProcessor.from_pretrained(model_name_siglip)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class_labels = ["a photo of a human", "a photo of a animal"]
text_inputs = clip.tokenize(class_labels).to(device)

# Modify this function to return detected_class and confidence
def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        logits_per_image, _ = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    detected_class = class_labels[probs.argmax()]
    confidence = probs.max()

    # Your custom sub-classifier
    res = pred_img(img, model_siglip, processor_siglip)
    
    if confidence > 0.8:
        if detected_class == "a photo of a human":
            detected_class = "human"
            print(detected_class, confidence, res)
        if detected_class == "a photo of a animal":
            detected_class = "animal"
        else:
            detected_class = "unknown"
            print("No significant detection.")
    
    return detected_class, confidence, res, img


def overlay_text_on_image(img, label_text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((20, 20), label_text, fill="red", font=font)
    return img


def process_and_save_images(test_dir="animal_test", output_dir="animal_output", json_path="animal_output/results.json"):
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = []

    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = Path(subdir) / file
            rel_path = image_path.relative_to(test_dir)
            ground_truth_class = rel_path.parts[0]  # First folder = label
            output_path = output_dir / rel_path.parent
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                detected_class, confidence, res, img = classify_image(image_path)

                if detected_class == "human":
                    text = f"{detected_class}: {confidence:.2f}"
                    img_with_text = overlay_text_on_image(img, text)
                    img_with_text.save(output_path / file)

                if detected_class == "animal" and confidence > 0.8:
                    label = res['top_rank']['label']
                    conf = res['top_rank']['confidence']
                    text = f"{label}: {conf:.2f}"
                    img_with_text = overlay_text_on_image(img, text)
                    img_with_text.save(output_path / file)

                # Record results
                results.append({
                    "filename": str(rel_path),
                    "ground_truth": ground_truth_class,
                    "predicted_class": detected_class,
                    "clip_confidence": float(confidence),
                    "siglip_label": res['top_rank']['label'] if detected_class == "animal" else None,
                    "siglip_confidence": float(res['top_rank']['confidence']) if detected_class == "animal" else None,
                    "correct": ground_truth_class == rel_path.parts[0]  # Compare with the first folder name
                })
                print(rel_path.parts[0])

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Save results to JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ All images processed and saved in '{output_dir}'. Results saved in '{json_path}'.")



# Call this function
process_and_save_images()
