import streamlit as st
from llama_index  import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

openai.api_key = st.secrets.openai_key
st.header("Chat with the Streamlit docs")

if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}]

@st.cache_resource(show_spinner=False)
def load_data(): 
    with st.spinner(text='Load and indexing the data')
    reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(llm = OpenAI(model='gpt-3.5-turbo', 
                                                                temperature=0.5, 
                                                                system_prompt= 'You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.'
                                                                )) 
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

index = load_data()
# Create a chat engine
## Four types of Engine: 
## Condense question
## Context chat 
## ReAct agent
## OpenAI agent
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompts for user input and display message 
if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({'role': 'user', 'content': prompt})

for message in st.session_state.messages: 
    with st.chat_message(message['role']): 
        st.write(message['content'])
        
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history