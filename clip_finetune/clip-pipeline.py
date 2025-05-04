import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

candidate_labels = ['photo of a human','photo of a animal']

threshold = 0.8  

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  
        probs = logits_per_image.softmax(dim=1)  

    for prob, label in zip(probs[0], candidate_labels):
        confidence = prob.item()
        if confidence >= threshold:
            if label == 'photo of a human':
                label = 'human'
            elif label == 'photo of a animal':
                label = 'animal'
            print(f"Detected '{label}' with confidence {confidence:.2%}")
    
    return probs[0].tolist()


if __name__ == "__main__":
    image_path = "squirrel.jpg"  # Replace with your image path
    print(classify_image(image_path))