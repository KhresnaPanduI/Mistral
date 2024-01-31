from llama_cpp import Llama
import gradio as gr 

MODEL_PATH = "phi-2.Q5_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=10, # number of CPU threads to use
)

# Initialize the global variable for Llama history
llama_history_global = [{"role": "system", "content": "You are helpful assistant"}]

def gradio_reply(user_input, history):
    global llama_history_global
    
    # Append the user's input to Llama history
    llama_history_global.append({"role": "user", "content": user_input})

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