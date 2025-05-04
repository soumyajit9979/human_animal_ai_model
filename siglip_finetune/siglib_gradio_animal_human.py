import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "Soumyajit9979/siglip-finetuned-animal-human"
model = SiglipForImageClassification.from_pretrained(model_name,revision="new_branch")
processor = AutoImageProcessor.from_pretrained(model_name,revision="new_branch")

def age_classification(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
  "0": "antelope",
  "1": "badger",
  "2": "bat",
  "3": "bear",
  "4": "bee",
  "5": "beetle",
  "6": "bison",
  "7": "boar",
  "8": "butterfly",
  "9": "cat",
  "10": "caterpillar",
  "11": "chimpanzee",
  "12": "cockroach",
  "13": "cow",
  "14": "coyote",
  "15": "crab",
  "16": "crow",
  "17": "deer",
  "18": "dog",
  "19": "dolphin",
  "20": "donkey",
  "21": "dragonfly",
  "22": "duck",
  "23": "eagle",
  "24": "elephant",
  "25": "flamingo",
  "26": "fly",
  "27": "fox",
  "28": "goat",
  "29": "goldfish",
  "30": "goose",
  "31": "gorilla",
  "32": "grasshopper",
  "33": "hamster",
  "34": "hare",
  "35": "hedgehog",
  "36": "hippopotamus",
  "37": "hornbill",
  "38": "horse",
  "39": "human",
  "40": "hummingbird",
  "41": "hyena",
  "42": "jellyfish",
  "43": "kangaroo",
  "44": "koala",
  "45": "ladybugs",
  "46": "leopard",
  "47": "lion",
  "48": "lizard",
  "49": "lobster",
  "50": "mosquito",
  "51": "moth",
  "52": "mouse",
  "53": "octopus",
  "54": "okapi",
  "55": "orangutan",
  "56": "otter",
  "57": "owl",
  "58": "ox",
  "59": "oyster",
  "60": "panda",
  "61": "parrot",
  "62": "pelecaniformes",
  "63": "penguin",
  "64": "pig",
  "65": "pigeon",
  "66": "porcupine",
  "67": "possum",
  "68": "raccoon",
  "69": "rat",
  "70": "reindeer",
  "71": "rhinoceros",
  "72": "sandpiper",
  "73": "seahorse",
  "74": "seal",
  "75": "shark",
  "76": "sheep",
  "77": "snake",
  "78": "sparrow",
  "79": "squid",
  "80": "squirrel",
  "81": "starfish",
  "82": "swan",
  "83": "tiger",
  "84": "turkey",
  "85": "turtle",
  "86": "whale",
  "87": "wolf",
  "88": "wombat",
  "89": "woodpecker",
  "90": "zebra"
}

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=age_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Animal-Human Classification",
    description="Upload an image to predict the class."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
