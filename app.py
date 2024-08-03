import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import CTransformers
import joblib
import os
import time

# Configuration for the LLM
config = {
    'max_new_tokens': 300,  # Adjusted to potentially reduce response time
    'temperature': 0.7,     # Adjusted for more coherent responses
    'context_length': 1024
}

# Initialize the model
llm = CTransformers(
    model='models/llama-2-7b-chat.ggmlv3.q2_K.bin',
    model_type='llama',
    config=config
)

# Define the prompt template
template = """
<s>[INST] <<SYS>>
You are a helpful AI assistant who has to act as advocate.
Answer based on the context provided. Don't answer unnecessarily if you don't find the context.
<</SYS>>
{context}
Question: {question}
Helpful Answer: [/INST]
"""

prompt = PromptTemplate.from_template(template)

# Load the PDF and process it
reader = PdfReader('data/fine_tune_data.pdf')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=350,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings_file = "./data/cases_genderviolence.joblib"
if os.path.exists(embeddings_file):
    embeddings = joblib.load(embeddings_file)
else:
    embeddings = HuggingFaceEmbeddings()
    joblib.dump(embeddings, embeddings_file)

vectorstore = Chroma.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

# Streamlit interface
st.title("Legal Advice Chatbot")
st.write("Enter your query below ")

query = st.text_input("Your Prompt:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Processing..."):
            start_time = time.time()
            # Get relevant documents
            docs = retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])

            # Create the input for the LLM
            chain_input = {"context": context, "question": query}
            result = prompt.format(**chain_input)
            response = llm(result)

            end_time = time.time()
            st.write(response)
            st.write(f"Response time: {end_time - start_time:.2f} seconds")
    else:
        st.write("Please enter a question.")
