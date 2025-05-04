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
imbalanced-learn
scikit-learn==1.4.2
pandas
evaluate
datasets
accelerate
git+https://github.com/huggingface/transformers.git
huggingface_hub
matplotlib
numpy
pillow==11.0.0
torchvision
opencv-python
gradio
openai-clip
datasets
tqdm
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
   pip install -r requirements.txt
   ```

## Fine-tuning Code

### CLIP Fine-tuning

The CLIP model was fine-tuned using the following notebook:
```
clip_finetune/CLIP_finetune_kaggle.ipynb
```

This notebook contains the complete fine-tuning pipeline for adapting the CLIP model to better detect humans and animals.

### SigLIP Fine-tuning

The SigLIP model was fine-tuned using the following notebook:
```
siglip_finetune/siglip-finetune-kaggle.ipynb
```

This notebook contains the full process for fine-tuning the SigLIP model on animal and human classification tasks.

## Model Weights

### CLIP Fine-tuned Model

The weights for the fine-tuned CLIP model can be found at:

   -[CLIP weights](https://drive.google.com/drive/folders/1ByVl3Aw52sktAikegyTpO2eTi_4ww26-?usp=sharing)

To test the fine-tuned CLIP model, use:
```
python clip_finetune/CLIP_finetuned_pipeline.py
```

### SigLIP Fine-tuned Models

Two fine-tuned SigLIP models are available on HuggingFace:

1. **Animal-only Classification Model**:
   - HuggingFace Model: [Soumyajit9979/animal-siglip-classification](https://huggingface.co/Soumyajit9979/animal-siglip-classification)

2. **Combined Human and Animal Classification Model**:
   - HuggingFace Model: [Soumyajit9979/siglip-finetuned-animal-human](https://huggingface.co/Soumyajit9979/siglip-finetuned-animal-human)

## Performance Metrics

| Model | Accuracy |
|-------|----------|
| CLIP (Fine-tuned) | 92.89% |
| SigLIP (Fine-tuned) | 92.96% |

The fine-tuned SigLIP model achieves marginally better accuracy than the fine-tuned CLIP model, with both models achieving excellent performance for the human and animal detection tasks.

## Integration

The main application (`main.py`) uses the CLIP model for initial human/animal detection and then leverages the fine-tuned SigLIP model for more detailed animal classification. This two-stage approach provides both broad categorization and specific animal identification.


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

## [Sample Output Link](https://drive.google.com/drive/folders/1MssJ4cPRYcB_4JkixFSX7XUgAsHzki7a?usp=sharing)



## Sample Outputs

Sample image detection outputs can be found at the following link:
[Animal Detection Images (Google Drive)](https://drive.google.com/drive/folders/11vd-E64NHqzXWL3SyU--AdhxZV4laQYj?usp=sharing)

Sample video detection outputs and inputs with log files are available here:
[Video Detection Examples (Google Drive)](https://drive.google.com/drive/folders/1bsEurz2tHF7dk_BJOW6DXyaRPJgdSs9Y?usp=sharing)




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

