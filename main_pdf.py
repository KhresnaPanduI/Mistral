import gradio as gr
from helper_pdf import PDFEmbeddingStore, load_faiss_index, load_chunks, search_index, get_text_results
from PyPDF2 import PdfReader
from llama_cpp import Llama
import os
import numpy as np
import faiss
import pickle

# Assuming PDFEmbeddingStore, load_faiss_index, load_chunks, search_index, prepare_llm_messages, and get_text_results functions are defined as shown previously.

# Initialize the LLM model
MODEL_PATH = r"/home/pandu/Documents/LLM/llama.cpp/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH, 
    verbose=True,
    chat_format="llama-2",  # Adjust chat_format as necessary
    n_ctx=8096
)
system_prompt = f"""You are helpful assistant that only answer in Bahasa. Your answer is based on the context given.
Answer the question directly without giving any reasoning or notes.
"""
llama_history_global = [{"role": "system", "content": system_prompt}]
pdf_embedding_store = PDFEmbeddingStore()  # Assuming this class is already defined

def prepare_llm_messages(user_query, chunks, indices):
    """
    Prepares messages for LLM chat including the context from retrieved chunks.
    """
    # Retrieve and prepare the context from chunks
    context_texts = get_text_results(indices, chunks)
    context_message = "\n\n".join(context_texts)  # Joining multiple chunks with new lines
    print(f"Context of the query is: \n{context_message}")
    full_message = f"User message is: {user_query}\n The context is: {context_message}. Answer concisely in Bahasa Indonesia"
    user_message_llama_format = {"role":"user", "content": full_message}
    return user_message_llama_format

# Gradio function to process PDF upload and create embeddings & FAISS index
def process_pdf(pdf_file):
    global index, chunks, pdf_embedding_store  # Use global variables for index, chunks, and the embedding store
    pdf_embedding_store = PDFEmbeddingStore()
    embeddings, chunks = pdf_embedding_store.parse_pdf_and_create_embeddings(pdf_file.name)
    index = pdf_embedding_store.create_faiss_index(embeddings)
    return "PDF processed and indexed successfully. You can now start chatting."

# Define function to handle chat with context from uploaded document
def query_with_context(user_input, history):
    global index, chunks, llama_history_global
    
    # Perform semantic search to get indices of relevant chunks
    distances, indices = search_index(user_input, pdf_embedding_store, index, top_k=3)
        
    # Prepare messages for LLM with context from retrieved chunks
    user_message_and_context = prepare_llm_messages(user_input, chunks, indices)
    
    # Append user input to Llama history
    llama_history_global.append(user_message_and_context)

    return "", history + [[user_input, None]]

def bot_response(bot_history):
    global llama_history_global

    # Generate response using LLM
    stream = llm.create_chat_completion(messages=llama_history_global, stream=True)

    bot_history[-1][1] = "" # for gradio output
    assistant_response = "" # for llama cpp chat history

    for item in stream:
        # Check if the delta is 'content'
        if 'content' in item['choices'][0]['delta']:
            content = item['choices'][0]['delta']['content']
            assistant_response += content
            bot_history[-1][1] +=  content
            yield bot_history

    # Append the assistant's response to Llama history
    llama_history_global.append({"role": "assistant", "content": assistant_response})
    
# Define the Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload PDF Document")
            pdf_upload = gr.File(label="Upload PDF")
            process_result = gr.Textbox(label="Processing Result")
            pdf_upload.change(process_pdf, inputs=[pdf_upload], outputs=process_result)
        with gr.Column():
            gr.Markdown("## Chat with Document Context")
            chatbot = gr.Chatbot(height=600)
            message = gr.Textbox()
            clear = gr.ClearButton([message, chatbot])
            message.submit(query_with_context, [message, chatbot], [message, chatbot], queue=False).then(
                bot_response, chatbot, chatbot
            )

app.launch()