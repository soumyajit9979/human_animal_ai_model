import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define your candidate labels
candidate_labels = ['photo of a human','photo of a animal']

# Set the confidence threshold
threshold = 0.8  # 30%

# Initialize the webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    # Convert the captured frame to RGB and then to PIL Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Preprocess the inputs
    inputs = processor(text=candidate_labels, images=pil_image, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # convert to probabilities

    # Process and display results
    for prob, label in zip(probs[0], candidate_labels):
        confidence = prob.item()
        if confidence >= threshold:
            if label == 'photo of a human':
                label = 'human'
            elif label == 'photo of a animal':
                label = 'animal'
            print(f"Detected '{label}' with confidence {confidence:.2%}")

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
