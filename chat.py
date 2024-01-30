import gradio as gr
from llama_cpp import Llama

LLM = Llama(
    model_path = "phi-2.Q5_K_M.gguf"
)

# Assuming llm is your language model function
def llm(prompt):
    # Your code to get the response from the model
    response = LLM(prompt)
    return response

def get_model_response(prompt):
    response = llm(prompt)
    return response['choices'][0]['text'].strip()

iface = gr.Interface(
    fn=get_model_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your message..."),
    outputs="text",
    title="Language Model Chat",
    description="This is a chat interface for a language model. Type your message below and get a response.",
    live=True  # This enables the chatbox-like interaction
)

iface.launch()
