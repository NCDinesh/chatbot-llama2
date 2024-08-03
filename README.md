# Legal Advice Chatbot

This project is a legal advice chatbot built using Streamlit, LangChain, and the LLaMA-2-7B model. The chatbot is designed to provide helpful answers to legal questions based on the context provided from a set of legal documents.

## Features

- Process legal documents from a PDF file.
- Convert the text into high-dimensional vectors for efficient retrieval.
- Use a pre-trained language model to generate context-aware responses.
- Streamlit interface for user interaction.

## Prerequisites

- Python 3.7 or higher
- Git
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- LLaMA-2-7B Model

### Download the Model

You need to download the LLaMA-2-7B model file. Place it in the `models` directory.

1. Create a `models` directory in the root of your project:

```bash
mkdir models
Download the LLaMA-2-7B model file (llama-2-7b-chat.ggmlv3.q2_K.bin) and place it in the models directory. You can find the model on the official release page here.
Setup
Clone the Repository
bash
Copy code
git clone https://github.com/NCDinesh/legal-advice-chatbot.git
cd legal-advice-chatbot
Install Dependencies
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Prepare the Embeddings
The project uses pre-computed embeddings for faster processing. If you don't have the embeddings file, it will be created during the first run.

Running the Application
Start the Streamlit application:

bash
Copy code
streamlit run app.py
Using the Chatbot
Open your web browser and go to http://localhost:8501.
Enter your legal question in the text input field.
Click the "Get Answer" button to receive a response based on the provided legal documents.
Project Structure
app.py: Main application file with Streamlit interface.
data/: Directory containing the PDF file with legal documents and the embeddings file.
models/: Directory where the LLaMA-2-7B model file should be placed.
requirements.txt: List of Python dependencies.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.
