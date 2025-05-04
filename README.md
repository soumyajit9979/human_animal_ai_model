# Human and Animal Detector

This project uses CLIP (Contrastive Language-Image Pre-Training) and a fine-tuned SigLIP model to detect humans and animals in images and videos, with additional classification for animal types.

## Features

- Human and animal detection in images
- Human and animal detection in videos (with frame-by-frame analysis)
- Real-time detection using webcam feed
- Logging of detections with timestamps
- Output video generation with detection labels

## Requirements

```
torch
clip
Pillow
opencv-python
transformers
```

You'll also need the fine-tuned SigLIP model for animal classification.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/soumyajit9979/human_animal_ai_model.git
   cd human_animal_ai_model
   ```

2. Install dependencies:
   ```
   pip install torch clip Pillow opencv-python transformers
   pip install -r requirements.txt
   ```


## Usage

The script provides four main functionalities:

### 1. Image Classification

Analyze a single image for human or animal presence:

```
python main.py --image path/to/image.jpg
```

### 2. Video Classification

Analyze a video file for human or animal presence:

```
python main.py --video path/to/video.mp4
```

### 3. Video Classification with Output

Analyze a video and create a labeled output video with detection logs:

```
python main.py --video_test path/to/video.mp4
```

This will generate:
- An output video with detection labels: `output_labeled_animal_video.mp4`
- A log file with timestamps and detections: `animal_video_detection_log.txt`

### 4. Live Camera Feed

Use your webcam for real-time human and animal detection:

```
python main.py --live
```

Press 'q' to quit the live feed.

## How It Works

1. **First-level detection**: Uses OpenAI's CLIP model to determine if an image contains a human or an animal with a confidence threshold of 0.8
2. **Second-level classification**: If an animal is detected, a fine-tuned SigLIP model (`Soumyajit9979/animal-siglip-classification`) classifies the specific animal type

## Sample Outputs

Sample image detection outputs can be found at the following link:
[Animal Detection Images (Google Drive)](gdrive link)

Sample video detection outputs and inputs with log files are available here:
[Video Detection Examples (Google Drive)](gdrive link)




### Gradio Interfaces

#### Animals-only Classification Interface

Run the Gradio interface for the SigLIP model fine-tuned on animals:

```
cd siglip_finetune
python siglip-gradio.py
```

#### Combined Human and Animal Classification Interface

Run the Gradio interface for the SigLIP model fine-tuned on both humans and animals:

```
cd siglip_finetune
python siglib_gradio_animal_human.py
```


Two fine-tuned SigLIP models are available for use:

1. **Animal-only Classification Model**:
   - HuggingFace Model: [Soumyajit9979/animal-siglip-classification](https://huggingface.co/Soumyajit9979/animal-siglip-classification)
   - Trained on animal dataset

2. **Combined Human and Animal Classification Model**:
   - HuggingFace Model: [Soumyajit9979/siglip-finetuned-animal-human](https://huggingface.co/Soumyajit9979/siglip-finetuned-animal-human)
   - Trained on combined human and animal dataset

## Training Datasets

The following datasets were used for training the fine-tuned models:

1. **Animals-only Dataset**:
   - HuggingFace Dataset: [Soumyajit9979/animals-dataset](https://huggingface.co/datasets/Soumyajit9979/animals-dataset)

2. **Combined Human and Animal Dataset**:
   - HuggingFace Dataset: [Soumyajit9979/animals-humans](https://huggingface.co/datasets/Soumyajit9979/animals-humans)

## Notes

- The application uses GPU acceleration if available, otherwise falls back to CPU
- For video processing, frames are analyzed at regular intervals (stride) to improve performance
- Detection confidence values are displayed both in the console and on the processed video frames

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]
