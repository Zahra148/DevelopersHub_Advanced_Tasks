# 📰 News Topic Classifier using BERT

This project implements a News Topic Classification system using BERT
(Bidirectional Encoder Representations from Transformers) fine-tuned on
the AG News dataset. The model classifies news articles into one of four
categories using deep learning and Hugging Face Transformers.

------------------------------------------------------------------------

## 🚀 Project Overview

The goal of this project is to build a high-performance text
classification model using a pre-trained BERT model and fine-tune it for
multi-class news topic classification.

### 📌 Categories

The model predicts one of the following categories:

-   World\
-   Sports\
-   Business\
-   Sci/Tech

### ✅ System Includes

-   Dataset loading and preprocessing\
-   Tokenization using BERT tokenizer\
-   Model fine-tuning\
-   Evaluation with accuracy and F1-score\
-   Model saving\
-   Deployment using Gradio

------------------------------------------------------------------------

## 🗂 Dataset

We use the AG News dataset from Hugging Face Datasets.

-   120,000 training samples\
-   7,600 test samples\
-   4 news categories

Each sample contains:

-   `text` (news article)\
-   `label` (category)

### 📥 Dataset Loading

``` python
from datasets import load_dataset

dataset = load_dataset("ag_news")
```

------------------------------------------------------------------------

## 🧠 Model Architecture

-   **Pretrained Model:** `bert-base-uncased`\
-   **Task:** Sequence Classification\
-   **Number of Labels:** 4\
-   **Max Sequence Length:** 128 tokens\
-   **Optimizer:** AdamW (default via Trainer)\
-   **Epochs:** 3\
-   **Learning Rate:** 2e-5\
-   **Batch Size:** 16

### 🔧 Model Initialization

``` python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)
```

------------------------------------------------------------------------

## ⚙️ Installation

Install required dependencies:

``` bash
pip install transformers==5.0.0 datasets torch scikit-learn gradio accelerate
```

------------------------------------------------------------------------

## 🔄 Training Pipeline

### 1️⃣ Tokenization

-   Uses `BertTokenizerFast`\
-   Truncation enabled\
-   Dynamic padding via `DataCollatorWithPadding`

### 2️⃣ Training (Hugging Face Trainer API)

``` python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
```

------------------------------------------------------------------------

## 📊 Evaluation Metrics

The model is evaluated using:

-   Accuracy\
-   Weighted F1-score

``` python
from sklearn.metrics import accuracy_score, f1_score

accuracy_score(labels, predictions)
f1_score(labels, predictions, average="weighted")
```

### Example Output

``` json
{
  "eval_loss": 0.25,
  "eval_accuracy": 0.94,
  "eval_f1": 0.94
}
```

------------------------------------------------------------------------

## 💾 Saving the Model

After training, the model and tokenizer are saved locally:

``` python
trainer.save_model("bert-ag-news-model")
tokenizer.save_pretrained("bert-ag-news-model")
```

You can zip the model directory for sharing or deployment.

------------------------------------------------------------------------

## 🌐 Deployment with Gradio

The project includes a simple Gradio interface to test the classifier
interactively.

### ✨ Features

-   Text input box\
-   Real-time prediction\
-   User-friendly interface

### 🧪 Example Usage

``` python
import gradio as gr

interface = gr.Interface(
    fn=predict_function,
    inputs="textbox",
    outputs="label"
)

interface.launch()
```

This allows users to input news text and receive predicted topic
instantly.

------------------------------------------------------------------------

## 📁 Hugging Face Space Structure

``` bash
.
├── Task1_News_Topic_Classifier_Using_BERT.ipynb
├── bert-ag-news-model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_cofig.json
├── app.py
├── Task 1 Hugging Face Link.txt
└── requirements.txt

```

------------------------------------------------------------------------

## 🧪 How to Run

### Option 1: Run Notebook

-   Open the notebook in Jupyter or Google Colab\
-   Run cells step-by-step\
-   Train and evaluate model

### Option 2: Use Saved Model

-   Load model from `bert-ag-news-model`\
-   Run Gradio interface\
-   Test custom news inputs

------------------------------------------------------------------------

## 📈 Expected Performance

BERT fine-tuned on AG News typically achieves:

-   Accuracy: \~93--95%\
-   F1-score: \~93--95%

> Exact results may vary depending on training environment.

------------------------------------------------------------------------

## 🛠 Technologies Used

-   Python\
-   PyTorch\
-   Hugging Face Transformers\
-   Hugging Face Datasets\
-   Scikit-learn\
-   Gradio

------------------------------------------------------------------------

## 📌 Key Learning Outcomes

-   Fine-tuning transformer models\
-   Text preprocessing and tokenization\
-   Using Hugging Face Trainer API\
-   Model evaluation techniques\
-   Deploying NLP models with Gradio

------------------------------------------------------------------------

## 👤 Author: Nayyab Zahra

Developed as part of a Deep Learning / NLP task for news classification
using BERT.
