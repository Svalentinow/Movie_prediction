import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import transformers
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel(BertConfig())
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))


@st.cache_resource
def load_modernbert():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_bert_classifier():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertClassifier(n_classes=10)
    model_path = hf_hub_download(
        repo_id="Svalent/bert-classifier-movie-genre",
        filename="bert_classifier_model.pth",
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_prediction_artifacts():
    with open("best_logistic_model.pkl", "rb") as file:
        logistic_model = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    return logistic_model, label_encoder


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def predict_single_text(text, model, tokenizer, max_len=400):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    return torch.argmax(outputs, dim=1).item()

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
txt = st.text_area('movie plot summary:',
                       placeholder="...", height=140)

if st.button('Submit'):
    st.write('You entered:')
    tokenizer, modernbert = load_modernbert()
    bert_tokenizer, bert_model = load_bert_classifier()
    logistic_model, label_encoder = load_prediction_artifacts()

    new_embeddings = get_embedding(txt, tokenizer, modernbert)
    X_new = new_embeddings.reshape(1, -1)
    predicted_label = logistic_model.predict(X_new)
    predicted_genre = label_encoder.inverse_transform(predicted_label)
    st.write(f"Predicted Genre: {predicted_genre[0]}")

    bert_label = predict_single_text(txt, bert_model, bert_tokenizer)
    bert_genre = label_encoder.inverse_transform([bert_label])[0]
    st.write(f"BERT predicted genre: {bert_genre}")


