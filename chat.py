from llama_cpp import Llama
import gradio as gr 

MODEL_PATH = "phi-2.Q5_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=256
)

# Initialize the global variable for Llama history
llama_history_global = [{"role": "system", "content": "You are helpful assistant"}]

def gradio_reply(user_input, history):
    global llama_history_global
    
    # Append the user's input to Llama history
    llama_history_global.append({"role": "user", "content": f"Instruction: {user_input}\nOutput:"})

    response = llm.create_chat_completion(messages=llama_history_global)

    assistant_response = response['choices'][0]['message']['content']

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