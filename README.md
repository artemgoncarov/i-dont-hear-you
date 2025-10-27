# Audio Classification Solution - Whisper-based Approach
# AKA Blending and Private Sharing Tutorial

thx artineon for the memes

## Overview

This project implements a multi-approach solution for audio classification using Whisper models. The task involves detecting specific audio patterns in speech recordings, specifically identifying cases where phrases like "не слышу" (can't hear) are present.

## Problem Statement

The goal is to classify audio files into two categories:
- **Class 0**: Audio where the phrase is not present
- **Class 1**: Audio where the phrase "не слышу" (can't hear) is present

## Solution Architecture

### Approach 1: Direct Whisper Inference with Rule-Based Classification

**Description**: Use Whisper model for transcription and apply simple rule-based classification.

**Method**:
1. Load and transcribe audio files using `openai/whisper-large-v3-turbo`
2. Apply rule: `1 if 'не слыш' in text else 0`
3. Process audio in chunks (4000s chunk length)

**Pros**: Fast, no training required  
**Cons**: Low accuracy, too simplistic

---

### Approach 2: CatBoost Text Classification

![Image alt](https://github.com/artemgoncarov/i-dont-hear-you/raw/main/images/catboost.jpeg)

**Description**: Improve upon Approach 1 by training a CatBoost classifier on transcribed texts.

**Method**:
1. Transcribe all training and test audio using Whisper
2. Load ground truth labels from `word_bounds.json`
3. Train CatBoost classifier with text features
4. Predict on test set

**Configuration**:
```python
CatBoostClassifier(
    iterations=10000,
    text_features=['text'],
    eval_metric='Accuracy',
    verbose=1000
)
```

**Pros**: Better accuracy than rule-based approach  
**Cons**: Still relies on transcription quality

---

### Approach 3: Fine-tuning Whisper for Classification

![Image alt](https://github.com/artemgoncarov/i-dont-hear-you/raw/main/images/agi.jpeg)

**Description**: Fine-tune Whisper encoder with a classification head directly on audio features.

**Architecture**:
```
Whisper Encoder → Global Average Pooling → Dropout(0.2) → Linear(hidden_size, 2)
```

**Training Configuration**:
- **Model**: `openai/whisper-medium`
- **Batch Size**: 4
- **Cross-Validation**: 5-fold StratifiedKFold
- **Epochs**: 1
- **Optimizer**: Adam
  - Encoder learning rate: 1e-5
  - Head learning rate: 3e-4
- **Loss**: CrossEntropyLoss
- **Metric**: F1-score (macro)

**Data Processing**:
- Sampling rate: 16,000 Hz
- Audio resampling using torchaudio
- Feature extraction using AutoProcessor

**Key Components**:

1. **Dataset Class** (`AudioClassificationDataset`):
   - Loads audio files using torchaudio
   - Resamples to 16kHz if needed
   - Processes using Whisper processor
   - Returns input features and labels

2. **Model Architecture** (`AudioClassificationModel`):
   - Uses Whisper encoder (frozen initially, then fine-tuned)
   - Global average pooling over time dimension
   - Dropout layer for regularization
   - Linear classification head

3. **Training Loop**:
   - Train/validation split with stratification
   - Progress tracking with tqdm
   - Loss and F1-score monitoring
   - Model checkpointing

**Pros**: Direct audio-to-label mapping, no transcription needed  
**Cons**: Requires training time and GPU resources

---

### Approach 4: Ensemble of Multiple Models

![Image alt](https://github.com/artemgoncarov/i-dont-hear-you/raw/main/images/danger.jpeg)
![Image alt](https://github.com/artemgoncarov/i-dont-hear-you/raw/main/images/agi1.png)

**Description**: Train multiple Whisper variants and blend predictions using majority voting.

**Models Used**:
1. `openai/whisper-medium`
2. `openai/whisper-large-v3-turbo`
3. `openai/whisper-small-v2`

**Ensemble Method**: Majority Voting
- Each model provides a prediction (0 or 1)
- Final prediction is the most common vote among the three models

**Implementation**:
```python
votes = [model1_pred, model2_pred, model3_pred]
final_label = max(set(votes), key=votes.count)
```

**Pros**: Improved robustness and accuracy through diversity  
**Cons**: Increased computational cost and training time

![Image alt](https://github.com/artemgoncarov/i-dont-hear-you/raw/main/images/money.jpeg)

---

## Results Summary

| Approach | Method | Complexity | Performance |
|----------|--------|------------|-------------|
| 1 | Rule-based | Low | Baseline |
| 2 | CatBoost on text | Medium | Better |
| 3 | Fine-tuned Whisper | High | Best (single model) |
| 4 | Ensemble | Very High | Best (overall) |

---

## Installation & Requirements

### Dependencies

```bash
pip install -U transformers huggingface_hub
pip install torch torchaudio
pip install librosa
pip install pandas numpy
pip install scikit-learn
pip install catboost
pip install tqdm
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (training takes significant time on CPU)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for models and data

---

## Usage

### 1. Prepare Data

Ensure your data structure is:
```
data/
├── wav_train/          # Training audio files
├── wav_test/           # Test audio files
└── word_bounds.json    # Training labels
```

### 2. Run Training

The notebook is organized sequentially. Run cells in order:

1. Install dependencies
2. Choose your approach (1-4)
3. Run training cells
4. Generate predictions

### 3. Generate Submissions

Each approach generates a CSV file:
- `approach_1.csv`
- `approach_2.csv`
- `approach_3_<model_id>.csv`
- `approach_4_blended.csv`

---

## Key Code Snippets

### Audio Loading
```python
def load_audio(path: str, target_sr: int = 16000):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, target_sr
```

### Model Training Loop
```python
for epoch in range(1, num_epochs + 1):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_features, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
```

### Inference
```python
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(input_features)
        preds = outputs['logits'].argmax(dim=1)
```

---

## Performance Considerations

1. **Chunk Length**: Using 4000s chunks for Approach 1-2 balances memory and quality
2. **Batch Size**: Adjust based on GPU memory (4 works well for most GPUs)
3. **Cross-Validation**: 5-fold CV provides robust validation
4. **Learning Rates**: Different rates for encoder (1e-5) and head (3e-4) prevent catastrophic forgetting

---

## Future Improvements

1. **Data Augmentation**: Add noise, time stretching, pitch shifting
2. **Longer Training**: Increase epochs for better convergence
3. **Advanced Ensembling**: Use weighted voting or stacking
4. **Model Distillation**: Train smaller models using ensemble knowledge
5. **Attention Mechanisms**: Add attention layers for better feature extraction

---

## Author Notes

This solution demonstrates progressive improvement through multiple approaches, from simple rule-based methods to sophisticated ensemble learning. The key insight is that direct audio feature learning (Approach 3-4) outperforms text-based methods by avoiding transcription errors.

---

## License

This project is for educational and competition purposes.
