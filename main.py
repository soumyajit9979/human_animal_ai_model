import argparse
import torch
import clip
from PIL import Image
import cv2
import os
from siglip_finetune.animal_classification import pred_img
from transformers import AutoImageProcessor, SiglipForImageClassification

model_name_siglip = "Soumyajit9979/animal-siglip-classification"
model_siglip = SiglipForImageClassification.from_pretrained(model_name_siglip)
processor_siglip = AutoImageProcessor.from_pretrained(model_name_siglip)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


#labels
class_labels = ["a photo of a human", "a photo of a animal"]
text_inputs = clip.tokenize(class_labels).to(device)

def classify_image(image_path):
    img=Image.open(image_path)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        logits_per_image, _ = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    detected_class = class_labels[probs.argmax()]
    confidence = probs.max()
    res=pred_img(img, model_siglip, processor_siglip)
    if confidence > 0.8:
        if detected_class == "a photo of a human":
            detected_class = "human"
            print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}")

        if detected_class == "a photo of a animal":
            detected_class = "animal"
            print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}, animal: {res['top_rank']['label']}, confidence: {res['top_rank']['confidence']}")
    else:
        print("No significant detection.")

def classify_video(video_path, stride=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Show the current frame
        cv2.imshow("Video Feed", frame)

        if frame_count % stride == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, _ = model(image, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            detected_class = class_labels[probs.argmax()]
            confidence = probs.max()
            res=pred_img(frame, model_siglip, processor_siglip)
            if confidence > 0.8:
                if detected_class == "a photo of a human":
                    detected_class = "human"
                    print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}")

                if detected_class == "a photo of a animal":
                    detected_class = "animal"
                    print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}, animal: {res['top_rank']['label']}, confidence: {res['top_rank']['confidence']}")
        
        frame_count += 1

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import cv2
import torch
from PIL import Image
import datetime

def classify_video_test(video_path, stride=10, output_path="output_labeled_animal_video.mp4", log_file="animal_video_detection_log.txt"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    with open(log_file, "w") as log:
        log.write("Timestamp, Frame, Label, Confidence\n")  # CSV header

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            label_to_show = ""
            conf_to_show = 0.0
            animal_to_show = ""

            if frame_count % stride == 0:
                # Calculate timestamp
                time_in_sec = frame_count / fps
                video_time = str(datetime.timedelta(seconds=int(time_in_sec)))

                # Image preprocessing and prediction
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_per_image, _ = model(image, text_inputs)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                detected_class = class_labels[probs.argmax()]
                confidence = probs.max()

                if confidence > 0.8:
                    if detected_class == "a photo of a human":
                        detected_class = "human"
                    elif detected_class == "a photo of a animal":
                        detected_class = "animal"

                    label_to_show = detected_class
                    conf_to_show = confidence

                    print(f"[{video_time}] Detected {label_to_show} with confidence {conf_to_show:.2f}")
                    log.write(f"{video_time}, {frame_count}, {label_to_show}, {conf_to_show:.2f}\n")

            label_text = f"{label_to_show} ({conf_to_show:.2f})"
            if animal_to_show:
                label_text += f", animal: {animal_to_show}"

            if label_to_show:
                cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Video Feed", frame)
            out.write(frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")
    print(f"Log saved to: {log_file}")


def classify_live_feed(stride=10):
    cap = cv2.VideoCapture(0)
    frame_count = 0
    print("Starting live camera feed. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % stride == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, _ = model(image, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            detected_class = class_labels[probs.argmax()]
            confidence = probs.max()
            res=pred_img(frame, model_siglip, processor_siglip)
            if confidence > 0.8:
                if detected_class == "a photo of a human":
                    detected_class = "human"
                    print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}")

                if detected_class == "a photo of a animal":
                    detected_class = "animal"
                    print(f"Alert: Detected {detected_class} with confidence {confidence:.2f}, animal: {res['top_rank']['label']}, confidence: {res['top_rank']['confidence']}")
        frame_count += 1

        # Display the frame
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human and Animal Detector using CLIP")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--video", type=str, help="Path to the video file")
    parser.add_argument("--video_test", type=str, help="Path to the video file for testing")
    parser.add_argument("--live", action="store_true", help="Use live camera feed")
    args = parser.parse_args()

    if args.image:
        classify_image(args.image)
    elif args.video:
        classify_video(args.video)
    elif args.video_test:
        classify_video_test(args.video_test)
    elif args.live:
        classify_live_feed()
    else:
        print("Please provide either an image path, a video path, or use --live for camera feed.")
