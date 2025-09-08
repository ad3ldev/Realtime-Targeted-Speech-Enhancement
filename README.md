# Real-Time Targeted Speech Enhancement

This repository contains the implementation of two novel models, the **Personalized Fast FullSubNet (PFFSN)** and the **Personalized Denoiser (PDenoiser)**, developed for real-time, targeted speech enhancement. This work builds upon existing deep learning models in speech enhancement with a strong focus on personalization for specific speakers and optimization for real-time performance.

## Abstract

Speech Enhancement (SE) has undergone significant advancements in recent years, moving from traditional algorithmic methods to more advanced deep learning-based models. These deep learning models have greatly improved speech quality and intelligibility. A major development in SE is personalized speech enhancement (PSE), where models are trained to enhance the speech of specific speakers. This project addresses the additional challenge of real-time constraints, which impose limitations on the model's architecture and size.

We propose two models: the **Personalized Fast FullSubNet (PFFSN)** and the **Personalized Denoiser (PDenoiser)**. Both models build on established SE architectures, extended with personalization support. PFFSN, an adaptation of FFSN, uses a subband approach to classify and amplify the subbands containing the target speaker's voice. PDenoiser, an adaptation of the Denoiser architecture, suppresses both stationary and non-stationary noises as well as non-primary speakers.

Our research also includes the creation of a novel Arabic dataset specifically designed for personalized speech enhancement. The project culminates in a real-time desktop application with a user-friendly graphical interface that seamlessly integrates the trained models and supports on-device real-time audio processing. It achieves an impressive low real-time factor (RTF) of 0.21.

## Key Features

* **Personalized Speech Enhancement:** The models are trained to recognize and enhance the voice of a specific, targeted speaker.
* **Real-Time Performance:** Architectures are optimized for low latency, with the application achieving an RTF of 0.21.
* **Two Novel Models:**
    * **Personalized Fast FullSubNet (PFFSN):** A subband-based model that efficiently isolates and enhances target speech by intelligently amplifying relevant subbands.
    * **Personalized Denoiser (PDenoiser):** An adaptation of the Denoiser architecture that suppresses both stationary and non-stationary noises as well as non-primary speakers in the time domain.
* **Novel Arabic Dataset:** A meticulously collected dataset of 30 speakers (73% male, 27% female) was created for this project, comprising over 83 hours of recorded speech for the training-validation set and 3.3 hours for the test set.
* **Desktop Application:** The project includes a real-time desktop application with a GUI for on-device audio processing, featuring a streaming pipeline with adjustable frame size and multi-threading.

## Model Architectures

### Personalized Fast FullSubNet (PFFSN)
The PFFSN model consists of two primary modules:
1.  **Speaker Embedding Module:** This module extracts speaker-specific features from the input signal. It uses the encoder and bottleneck layers of the FFSN architecture, followed by a feedforward network to classify if a subband contains the target speaker's voice.
2.  **Denoising Module:** This module enhances the speech signal using the features from the speaker embedding module.

### Personalized Denoiser (PDenoiser)
The PDenoiser model also features two main components:
1.  **Speaker Embedding Module:** This module utilizes the TitaNet architecture to extract 192-dimensional speaker embeddings, which are then enhanced to a dimensionality of 768 using linear, normalization, and dropout layers.
2.  **Denoising Module:** This module employs a standard Denoiser model. The output of the encoder is reweighted by the embedding module's output, and a weighted residual connection is added for better gradient flow.

## Datasets and Evaluation

This project utilized several publicly available datasets and introduced a new one.

* **Datasets Used:** Valentini, DNS 2020/2022/2023, LibriSpeech, and a new custom-collected Arabic dataset.
* **Evaluation Metrics:** Performance was assessed using a variety of objective metrics, including:
    * Perceptual Evaluation of Speech Quality (PESQ)
    * Short-Time Objective Intelligibility (STOI)
    * Deep Noise Suppression Mean Opinion Score (DNSMOS)
    * Real-time Factor (RTF)
    * Word Error Rate (WER)
    * Signal-to-Noise Ratio (SNR)

## Getting Started

The project is built on the following technology stack:

* **PyTorch**: The core deep learning framework.
* **PyTorch Lightning**: For streamlined training and evaluation.
* **Hydra**: For configuration management.
* **Weights and Biases / TensorBoard**: For experiment tracking and visualization.
* **Tkinter / CustomTkinter**: For the graphical user interface.
* **SQLite**: For database management within the application.

*Detailed instructions for setting up the environment, installing dependencies, and running the models will be provided here*

## Note

For more details and a deeper dive into the research, please refer to the full PDF file: **[Link to PDF](https://github.com/ad3ldev/Realtime-Targeted-Speech-Enhancement/blob/main/Real-Time%20Targeted%20Speech%20Enhancement.pdf)**

## License

This project is part of academic research at Alexandria University and is licensed under GPL. Please contact the authors for usage permissions.
