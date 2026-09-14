import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import transformers
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel

st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #6B214F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
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
def load_bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource
def load_bert_classifier():
    tokenizer = load_bert_tokenizer()
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
def load_prediction_models():
    with open("best_logistic_model.pkl", "rb") as file:
        logistic_model = pickle.load(file)
    with open("best_forest_classifier.pkl", "rb") as file:
        random_forest_model = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    return logistic_model, random_forest_model, label_encoder


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

class BertCommercialSuccessClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size + 1, n_classes)

    def forward(self, input_ids, attention_mask, budget):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = self.drop(outputs.pooler_output)
        budget_feature = budget.float().view(-1, 1)
        combined = torch.cat((pooled_output, budget_feature), dim=1)
        return self.out(combined)


@st.cache_resource
def load_bert_commercial_success_classifier():
    tokenizer = load_bert_tokenizer()
    model = BertCommercialSuccessClassifier(n_classes=2)
    model_path = hf_hub_download(
        repo_id="Svalent/bert_commercial_success_classifier",
        filename="bert_commercial_success_classifier.pth",
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    budget_scaler = StandardScaler()
    budget_scaler.mean_ = checkpoint["budget_scaler_mean"]
    budget_scaler.scale_ = checkpoint["budget_scaler_scale"]
    budget_scaler.var_ = budget_scaler.scale_ ** 2
    budget_scaler.n_features_in_ = budget_scaler.mean_.shape[0]
    class_names = checkpoint["class_names"]

    model.eval()
    return tokenizer, model, budget_scaler, class_names
def predict_commercial_success(text, budget, model, tokenizer, budget_scaler, max_len=400):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    scaled_budget = budget_scaler.transform(
        np.asarray([[budget]], dtype=np.float32)
    )
    budget_tensor = torch.tensor(scaled_budget, dtype=torch.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            budget=budget_tensor.to(device),
        )
    return torch.argmax(outputs, dim=1).item()

st.title("🎈 Movie Genre Predictor")

genre_tab, commercial_tab = st.tabs(["Genre prediction", "Commercial success"])

with genre_tab:
    genre_text = st.text_area(
        "Paste the movie plot summary to get a genre prediction:",
        placeholder="...",
        height=140,
        key="genre_text",
    )

    if st.button("Predict genre", key="genre_button"):
        tokenizer, modernbert = load_modernbert()
        bert_tokenizer, bert_model = load_bert_classifier()
        logistic_model, random_forest_model, label_encoder = load_prediction_models()

        new_embeddings = get_embedding(genre_text, tokenizer, modernbert)
        X_new = new_embeddings.reshape(1, -1)
        logistic_label = logistic_model.predict(X_new)
        logistic_genre = label_encoder.inverse_transform(logistic_label)
        st.write(f"Predicted Logistic Regression Genre: {logistic_genre[0]}")

        random_forest_label = random_forest_model.predict(X_new)
        random_forest_genre = label_encoder.inverse_transform(random_forest_label)
        st.write(f"Predicted Random Forest Genre: {random_forest_genre[0]}")

        bert_label = predict_single_text(genre_text, bert_model, bert_tokenizer)
        bert_genre = label_encoder.inverse_transform([bert_label])[0]
        st.write(f"BERT predicted genre: {bert_genre}")

with commercial_tab:
    
    commercial_budget = st.number_input(
        "Movie budget",
        min_value=0.0,
        value=0.0,
        step=1_000_000.0,
        format="%.0f",
        help="Enter the movie budget using USD currency.",
    )

    commercial_text = st.text_area(
            "Paste the movie plot summary and budget to predict commercial success:",
            placeholder="...",
            height=140,
            key="commercial_text",
        )
    if st.button("Predict commercial success", key="commercial_button"):
        bert_tokenizer, commercial_model, budget_scaler, class_names = (
            load_bert_commercial_success_classifier()
        )
        commercial_label = predict_commercial_success(
            commercial_text,
            commercial_budget,
            commercial_model,
            bert_tokenizer,
            budget_scaler,
        )
        st.write(
            f"Commercial success prediction: "
            f"{class_names[commercial_label]}"
        )


