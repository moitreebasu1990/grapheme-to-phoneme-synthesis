# Grapheme-to-Phoneme Conversion with LSTM in PyTorch

This project implements a Grapheme-to-Phoneme (G2P) conversion model using an LSTM-based sequence-to-sequence architecture with attention mechanism in PyTorch. The model converts written characters (graphemes) into their corresponding sounds (phonemes).

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Testing and Visualization](#testing-and-visualization)

## Introduction

Grapheme-to-Phoneme conversion is an essential task in speech synthesis and recognition systems. This project demonstrates how to build and train a G2P model using PyTorch, leveraging the CMU Pronouncing Dictionary for training data.

## Installation

To run this project, you need to install the following dependencies:

```bash
pip install torch nltk seaborn matplotlib numpy tqdm
```

Additionally, download the required NLTK data:

```python
import nltk
nltk.download('cmudict')
```

## Usage

The project is structured as a Jupyter notebook. To use it:

1. Open the notebook in a Jupyter environment.
2. Run the cells sequentially to load data, build the model, train it, and test it.

## Model Architecture

The G2P model consists of three main components:

1. **Encoder**: An LSTM that processes input graphemes.
2. **Attention Mechanism**: Calculates attention weights between encoder and decoder states.
3. **Decoder**: An LSTM that generates output phonemes, utilizing the attention mechanism.

## Training

The model is trained using the following hyperparameters:

- Embedding size: 128
- Hidden size: 256
- Number of layers: 2
- Dropout: 0.2
- Learning rate: 0.001
- Number of epochs: 30

The training loop iterates over the dataset, calculates the loss for each batch, and updates the model parameters using the Adam optimizer.

## Testing and Visualization

After training, you can test the model on custom words using the `g2p_convert` function. The `plot_attention` function allows you to visualize the attention weights, providing insights into how the model focuses on different parts of the input sequence when generating the output.

Example usage:

```python
test_word = "archivist"
phonemes, attentions = g2p_convert(model, test_word, grapheme2idx, idx2phoneme, phoneme2idx, device)
print(f"Input word: {test_word}")
print(f"Phonemes: {phonemes}")
plot_attention(test_word, phonemes, attentions)
```

This will output the predicted phonemes for the input word and display a heatmap of the attention weights.