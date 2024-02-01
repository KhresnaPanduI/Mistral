from llama_cpp import Llama
import gradio as gr 

MODEL_PATH = r"C:\Users\Aevum\Documents\Belajar\Coding\LLM\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH
    )

# Initialize the global variable for Llama history
llama_history_global = []

def gradio_reply(user_input, history):
    global llama_history_global
    
    # Append the user's input to Llama history
    llama_history_global.append({"role": "user", "content": f"{user_input} answer concisely"})

    stream = llm.create_chat_completion(messages=llama_history_global, stream=True)

    assistant_response = ""
    for item in stream:
        if 'content' in item['choices'][0]['delta']:
            content = item['choices'][0]['delta']['content']
            assistant_response += content

    # Append the assistant's response to history for Gradio
    history.append((user_input, assistant_response))
   # Append the assistant's response to Llama history
    llama_history_global.append({"role": "assistant", "content": assistant_response})

    return "", history

with gr.Blocks() as interface:
    chatbot = gr.Chatbot(height=600)
    message = gr.Textbox()
    clear = gr.ClearButton([message, chatbot])
    message.submit(gradio_reply, [message, chatbot], [message, chatbot])

if __name__ == "__main__":
    interface.launch()