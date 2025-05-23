import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model
import joblib
import streamlit as st

# ---------- Caminhos ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLB_PATH = os.path.join(BASE_DIR, "data", "mlb_597.pkl")

# ---------- Parâmetros ----------
TOP_N = 10
CHUNK_PB = 512
CHUNK_ESM = 1024

# ---------- Cache dos modelos HuggingFace ----------
@st.cache_resource
def load_hf_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=False)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tokenizer, model

# ---------- Função para gerar embedding por chunk ----------
def embed_sequence(model_name, seq, chunk_size):
    tokenizer, model = load_hf_model(model_name)

    def format_seq(s):
        return " ".join(list(s))

    chunks = [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]
    embeddings = []

    for chunk in chunks:
        formatted = format_seq(chunk)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls)

    return np.mean(embeddings, axis=0, keepdims=True)

# ---------- Carregar modelos ----------
mlp_pb  = load_model(os.path.join(MODELS_DIR, "mlp_protbert.h5"), compile=False)
mlp_bfd = load_model(os.path.join(MODELS_DIR, "mlp_protbertbfd.h5"), compile=False)
mlp_esm = load_model(os.path.join(MODELS_DIR, "mlp_esm2.h5"), compile=False)
stacking = load_model(os.path.join(MODELS_DIR, "modelo_ensemble_stack.h5"), compile=False)

# ---------- Carregar MultiLabelBinarizer ----------
mlb = joblib.load(MLB_PATH)
go_terms = mlb.classes_

# ---------- Interface Streamlit ----------
st.title("Predição de Funções de Proteínas")

seq = st.text_area("Insere a sequência FASTA:", height=200)

# Limpar sequência: remover cabeçalhos (">") e espaços/quebras
if seq:
    seq = "\n".join([line for line in seq.splitlines() if not line.startswith(">")])
    seq = seq.replace(" ", "").replace("\n", "").strip()

if st.button("Prever GO terms"):
    if not seq:
        st.warning("Por favor, insere uma sequência válida.")
    else:
        st.write("A gerar embeddings por chunks...")

        emb_pb  = embed_sequence("Rostlab/prot_bert", seq, CHUNK_PB)
        emb_bfd = embed_sequence("Rostlab/prot_bert_bfd", seq, CHUNK_PB)
        emb_esm = embed_sequence("facebook/esm2_t33_650M_UR50D", seq, CHUNK_ESM)

        st.write("A fazer predições base...")

        y_pb  = mlp_pb.predict(emb_pb)[:, :597]
        y_bfd = mlp_bfd.predict(emb_bfd)[:, :597]
        y_esm = mlp_esm.predict(emb_esm)[:, :597]

        X_stack = np.concatenate([y_pb, y_bfd, y_esm], axis=1)
        y_pred = stacking.predict(X_stack)

        st.subheader("GO terms com probabilidade ≥ 0.5:")
        predicted = mlb.inverse_transform((y_pred >= 0.5).astype(int))[0]
        if predicted:
            st.code("\n".join(predicted))
        else:
            st.info("Nenhum GO term com probabilidade ≥ 0.5.")

        st.subheader(f"Top {TOP_N} GO terms mais prováveis:")
        top_idx = np.argsort(-y_pred[0])[:TOP_N]
        for i in top_idx:
            st.write(f"{go_terms[i]} : {y_pred[0][i]:.4f}")

