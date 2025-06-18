# 🧠 GPT From Scratch: Transformer Decoder Model

This project builds a GPT-style language model from scratch using PyTorch. It walks through every stage: from data loading, vocabulary encoding, and batching, to implementing self-attention, multi-head attention, positional encoding, transformer decoder blocks, and full autoregressive text generation.

The model is trained on the **Tiny Shakespeare** dataset and can generate coherent character-level text samples.

---

## 📁 Project Structure

```bash
transformer-gpt-scratch/
├── main.py            # All code: data prep, model, training, generation
├── input.txt          # Shakespeare corpus (auto-downloaded)
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation (this file)
```

---

## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the project**

```bash
python main.py
```

This will:

* Download the Tiny Shakespeare dataset
* Train a custom GPT model on character-level data
* Plot training and validation loss curves
* Generate a sample Shakespearean-style text

---

## 🔍 Key Features

### 🧠 GPT-style Transformer

* **Positional Encoding**: Add temporal information to character embeddings
* **Masked Self-Attention**: Prevent future token peeking during training
* **Multi-Head Attention**: Improve learning with parallel attention heads
* **Layer Normalization** and **Residual Connections**
* **Feedforward Layers** between attention modules
* **Stacked Decoder Layers** for richer context modeling

### 📚 Dataset

* Tiny Shakespeare (110,000+ characters of text)
* Downloaded automatically if not present

---

## 🔬 Training Setup

* Model trained for 10 epochs
* Optimizer: Adam
* Loss Function: CrossEntropy
* Hyperparameters:

  * `d_model = 128`
  * `num_heads = 8`
  * `context_len = 100`
  * `num_layers = 2`
  * `batch_size = 64`

---

## 📈 Outputs

### ✅ Loss Plot

Shows convergence of training and validation loss over epochs.

### 📝 Generated Text Sample

```text
Generated Sample:
From thine eyes to the soul that sleeps in night,
Yet would I weep no more for joy than fright.
```

---

## 🧰 Requirements

```txt
torch
numpy
matplotlib
requests
```

---

## 💡 Learning Objectives

* Understand transformer decoders and masked self-attention
* Implement positional encoding and multi-head attention from scratch
* Gain experience in training and evaluating autoregressive models
* Generate creative text using learned character patterns

---

## 👥 Contributors

- **Julisa Delfin** – MS Data Science, DePaul University
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/julisadelfin/) 
