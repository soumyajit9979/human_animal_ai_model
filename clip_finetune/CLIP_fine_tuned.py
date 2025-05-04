import torch
import clip
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load your fine-tuned weights
model.load_state_dict(torch.load("clip_finetuned_human92.pth", map_location=device), strict=False)

model.eval()

def predict(image):
    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Define your custom labels
    labels  = [
    "antelope", "badger", "bear", "bat", "bee", "beetle", "bison", "boar", "butterfly",
    "cat", "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow",
    "deer", "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant",
    "flamingo", "fly", "fox", "goat", "goldfish", "goose", "gorilla", "grasshopper",
    "hamster", "hare", "hedgehog", "hippopotamus", "hornbill", "horse","human", "hummingbird",
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs", "leopard", "lion", "lizard",
    "lobster", "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "otter",
    "owl", "ox", "oyster", "panda", "parrot", "pelecaniformes", "penguin", "pig",
    "pigeon", "porcupine", "possum", "raccoon", "rat", "reindeer", "rhinoceros",
    "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", "sparrow", "squid",
    "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", "wolf",
    "wombat", "woodpecker", "zebra"
] # Example labels
    text_inputs = clip.tokenize(labels).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Compute similarity
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Get the label with highest probability
    predicted_label = labels[probs.argmax()]
    return predicted_label

# Define your class labels (replace with your actual labels)
import gradio as gr

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(label="Predicted Label"),
    title="CLIP Image Classification",
    description="Upload an image to get the predicted label using the fine-tuned CLIP model."
)

interface.launch()
