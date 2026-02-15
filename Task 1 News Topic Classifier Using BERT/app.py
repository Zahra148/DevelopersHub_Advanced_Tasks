import torch
import gradio as gr
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_PATH = "./bert-ag-news-model"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

examples = [
    ["United Nations holds emergency meeting amid rising geopolitical tensions"],  # World
    ["Lakers defeat Warriors in dramatic overtime NBA playoff game"],               # Sports
    ["Global stock markets surge after central bank announces interest rate cut"],  # Business
    ["Researchers develop breakthrough quantum computing processor"],               # Sci/Tech
]

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline"),
    outputs="text",
    title="📰 AG News Topic Classifier (BERT)",
    description="Classifies news headlines into World, Sports, Business, or Sci/Tech",
    examples=examples
)

demo.launch()
