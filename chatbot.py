import streamlit as st
from langchain_community.llms import CTransformers
import time
import torch
import os

# Set up the Streamlit page configuration
st.set_page_config(page_title="Chat with Llama2",
                   layout='centered',
                   initial_sidebar_state='collapsed')

# Check CUDA availability and print device details
cuda_available = torch.cuda.is_available()
if cuda_available:
    st.write(f"CUDA Device Count: {torch.cuda.device_count()}")
    st.write(f"Current CUDA Device: {torch.cuda.current_device()}")
    torch.cuda.device(0)
    # torch.device('cuda')
    st.write(f"CUDA Device: {torch.cuda.get_device_name(0)}")

@st.cache_resource  # Caching the model loading process to avoid reloading on every interaction
def load_model():
    config = {
        'max_new_tokens': 500,  # Adjusted to potentially reduce response time
        'temperature': 0.7,     # Adjusted for more coherent responses
    }

    model_path = os.path.join('models', 'llama-2-7b-chat.ggmlv3.q2_K.bin')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    # Initialize the model
    try:
        llm = CTransformers(
            model=model_path,
            model_type='llama',
            config=config
        )
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        raise

    return llm

# Function to get a response from the Llama2 model
def get_llama_response(input_text, conversation_history):
    llm = load_model()
    prompt = "\n".join(conversation_history + [f"User: {input_text}", "Bot:"])
    response = llm.invoke(prompt)
    
    # Ensure we only get the bot's response part
    bot_response = response.split("Bot:")[-1].split("User:")[0].strip()
    return bot_response

# Create the header for the Streamlit app
st.header("Chat with Llama2")

# Initialize the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Text input for user message
input_text = st.text_input("Enter your message")

# Button to generate the response
if st.button("Send"):
    if input_text.strip():  # Ensure that input_text is not empty or just whitespace
        start_time = time.time()
        
        # Append the user's message to the conversation history
        st.session_state.conversation_history.append(f"User: {input_text}")
        
        # Get the response from the Llama2 model
        response = get_llama_response(input_text, st.session_state.conversation_history)
        
        # Append the bot's response to the conversation history
        st.session_state.conversation_history.append(f"Bot: {response}")
        
        # Display the conversation history
        for message in st.session_state.conversation_history:
            st.write(message)
        
        # Clear the input box after sending the message
        st.text_input("Enter your message", value="", key="empty_input")
        
        end_time = time.time()
        st.write(f"Response time: {end_time - start_time:.2f} seconds")
    else:
        st.error("Please enter a message.")
else:
    # Display the conversation history if available
    if st.session_state.conversation_history:
        for message in st.session_state.conversation_history:
            st.write(message)
