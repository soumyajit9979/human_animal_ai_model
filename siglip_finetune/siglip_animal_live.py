import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

# Load model and processor
model_name = "Soumyajit9979/animal-siglip-classification"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Define class labels
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
  "39": "hummingbird",
  "40": "hyena",
  "41": "jellyfish",
  "42": "kangaroo",
  "43": "koala",
  "44": "ladybugs",
  "45": "leopard",
  "46": "lion",
  "47": "lizard",
  "48": "lobster",
  "49": "mosquito",
  "50": "moth",
  "51": "mouse",
  "52": "octopus",
  "53": "okapi",
  "54": "orangutan",
  "55": "otter",
  "56": "owl",
  "57": "ox",
  "58": "oyster",
  "59": "panda",
  "60": "parrot",
  "61": "pelecaniformes",
  "62": "penguin",
  "63": "pig",
  "64": "pigeon",
  "65": "porcupine",
  "66": "possum",
  "67": "raccoon",
  "68": "rat",
  "69": "reindeer",
  "70": "rhinoceros",
  "71": "sandpiper",
  "72": "seahorse",
  "73": "seal",
  "74": "shark",
  "75": "sheep",
  "76": "snake",
  "77": "sparrow",
  "78": "squid",
  "79": "squirrel",
  "80": "starfish",
  "81": "swan",
  "82": "tiger",
  "83": "turkey",
  "84": "turtle",
  "85": "whale",
  "86": "wolf",
  "87": "wombat",
  "88": "woodpecker",
  "89": "zebra"
}

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB and PIL Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()


    y_offset = 30
    for i, prob in enumerate(probs):
        if prob > 0.6:
            label = labels[str(i)]
            text = f"{label}: {prob:.2f}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30

    cv2.imshow("Live Classification", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
