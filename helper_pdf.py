import os
import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModel
import torch

class PDFEmbeddingStore:
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def text_to_embedding(self, text, text_type="passage"):
        # Prefix the text according to its type (query or passage)
        prefixed_text = f"{text_type}: {text}"
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings

    def parse_pdf_and_create_embeddings(self, pdf_file_path):
        pdf_reader = PdfReader(pdf_file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Fallback to empty string if None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        embeddings = np.vstack([self.text_to_embedding(chunk, text_type="passage") for chunk in chunks if chunk.strip()])
        return embeddings, chunks

    def create_faiss_index(self, embeddings):
        embeddings = self.normalize_embeddings(embeddings)  # Normalize before creating the index
        d = embeddings.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatIP(d)  # Use IndexFlatIP for cosine similarity
        index.add(embeddings)
        return index
    
# Functions for loading the FAISS index and chunks, and for searching the index
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_chunks(chunks_path):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return chunks

def search_index(query, pdf_embedding_store, index, top_k=3):
    query_embedding = pdf_embedding_store.text_to_embedding(query, text_type="query")
    query_embedding = pdf_embedding_store.normalize_embeddings(query_embedding)
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    return distances, indices

def get_text_results(indices, chunks):
    return [chunks[idx] if idx >= 0 else "Not found" for idx in indices[0]]