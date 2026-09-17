# Image Captioning — Case Study & Model Training

A from-scratch walkthrough of building an image captioning model using transfer learning (ResNet50) and a custom-trained LSTM decoder, trained on the Flickr8k dataset. This notebook produced the `model_9.h5` model powering the deployed captioning app.

**Live App:** https://image-caption-generator-frontend.vercel.app/

## Related Repositories

| Repo | Description |
|---|---|
| [image-caption-generator-backend](https://github.com/AryanSehgal/image-caption-generator-backend) | Flask API that serves the model trained in this notebook |
| [image-caption-generator-frontend](https://github.com/AryanSehgal/image-caption-generator-frontend) | React + TypeScript UI for the deployed app |

## Overview

This repository documents the full pipeline for training an image captioning model from scratch: preparing the Flickr8k dataset, extracting image features via transfer learning, building a vocabulary, preparing GloVe word embeddings, and training an encoder-decoder neural network to generate captions.

## Dataset: Flickr8k

The model is trained on **Flickr8k** — approximately **8,000 images**, each paired with **5 human-written captions** (40,460 captions total). The dataset is dominated by photos of **people and dogs in everyday, outdoor settings** (playing, running, climbing, at the park, etc.).

This composition directly shapes what the trained model can and can't do:

> **The deployed app performs best on photos of people, dogs, and everyday outdoor scenes — and can produce confidently incorrect captions for anything well outside that domain (e.g. close-ups of unfamiliar objects, animals, or scenes not represented in Flickr8k).** This is a direct consequence of the training data, not a bug in the model or the deployment pipeline.

## Pipeline

### 1. Data Cleaning
Captions are lowercased, stripped of punctuation and non-alphabetic characters, and single-letter tokens are removed.

### 2. Vocabulary Construction
Words appearing **10 or fewer times** across the entire caption corpus are discarded. This reduces the vocabulary from 8,424 unique words down to **1,845 words** (1,848 including `startseq`/`endseq`/padding tokens). The trained model can only ever produce words from this fixed vocabulary — it is structurally incapable of outputting anything outside of it.

### 3. Image Feature Extraction (Transfer Learning)
Every image is passed through **ResNet50** (pretrained on ImageNet — this part is *not* trained by us) with its final classification layer removed, producing a **2048-dimensional feature vector** per image. This step is purely a fixed, generic "what does this image look like numerically" encoder.

### 4. Word Embeddings
Rather than learning word representations from scratch, captions use **pretrained GloVe embeddings** (50-dimensional). These are frozen (`trainable = False`) during training — the model learns how to sequence words, not what they mean.

### 5. Model Architecture
A relatively small encoder-decoder network (~1.47M trainable parameters):
- Image feature vector (2048-d) → Dropout → Dense(256, relu)
- Caption sequence → Embedding (frozen GloVe) → Dropout → LSTM(256)
- The two branches are merged (added), passed through Dense(256, relu), then a final Dense(vocab_size, softmax) layer predicts the next word

### 6. Training
Trained for **20 epochs**, using a custom batch generator (captions are split into partial sequences for next-word prediction, a standard technique for sequence generation). A checkpoint is saved after every epoch (`model_0.h5` through `model_19.h5`).

### 7. Inference (Greedy Decoding)
Captions are generated one word at a time: starting from `startseq`, the model predicts the most probable next word, appends it, and repeats until it predicts `endseq` or hits a max length of 35 words. This is **greedy decoding** — the model never reconsiders an earlier word choice, which is a known contributor to occasionally awkward or generic-sounding captions (a common characteristic of simple encoder-decoder captioning models, not unique to this one).

## Model Checkpoint Used in Production

The deployed backend currently uses **`model_9.h5`** — the checkpoint from epoch 9 of 20. This was the checkpoint available at deployment time; later epochs (`model_10.h5`–`model_19.h5`) may offer improved caption quality if available, since the notebook itself does not include a validation-based "best epoch" selection step.

## Requirements to Reproduce Training

- Flickr8k images and caption files (`Flickr8k.token.txt`, `Flickr_8k.trainImages.txt`, `Flickr_8k.testImages.txt`) — publicly available, not included in this repo due to size
- GloVe embeddings (`glove.6B.50d.txt`, ~800MB) — [available here](https://nlp.stanford.edu/projects/glove/)
- Python environment with TensorFlow/Keras, numpy, pandas, matplotlib, nltk
  - **Note for Apple Silicon (M1/M2/M3/M4) users:** standard `pip install tensorflow` may fail to find a compatible build. Use `conda`/Miniforge with Apple's official TensorFlow packages (`tensorflow-deps`, `tensorflow-macos`, `tensorflow-metal`) instead.

```bash
pip install tensorflow pandas numpy matplotlib nltk pillow
```

## Deployment

The `model_9.h5` checkpoint produced by this notebook, along with the generated `word_to_idx.pkl` / `idx_to_word.pkl` vocabulary mappings, are consumed directly by the [backend API](https://github.com/AryanSehgal/image-caption-generator-backend), which serves them via a Dockerized Flask application on Render. See that repo for the full deployment pipeline and how these artifacts are used in production inference.

## Credits

Original model architecture and training notebook by **Apoorv Garg**. Deployment pipeline and companion web application by **Aryan Sehgal**.
