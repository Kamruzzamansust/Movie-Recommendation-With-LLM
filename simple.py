from langchain_ai21 import ChatAI21
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere  # Use the correct LLM class from langchain
import getpass


import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere
from langchain_cohere import ChatCohere

# Load environment variables
load_dotenv()

# Function to get API key and validate
def get_api_key(key_name):
    api_key = os.getenv(key_name)
    if not api_key:
        st.error(f"API key for {key_name} not found. Please check your .env file.")
        st.stop()
    return api_key

# Streamlit interface
st.title("Movie Recommendation System Using Large Language Models")

# Dropdown for model selection
model_choice = st.selectbox("Select a model", ["Cohere", "AI Lab 21"])

# Input for movie genre
genre = st.text_input("Enter a movie genre:")

# Button to generate recommendations
if st.button("Get Recommendations"):
    if model_choice == "Cohere":
        api_key = get_api_key("COHERE_API_KEY")
        os.environ["COHERE_API_KEY"] = api_key
        llm = ChatCohere(model="command", max_tokens=100, temperature=0.8)
    elif model_choice == "AI Lab 21":
        api_key = get_api_key("AI21_API_KEY")
        os.environ["AI21_API_KEY"] = api_key
        
        llm = ChatAI21(model = 'jamba-instruct',
               temperature=0.8,
               max_tokens=512,
               ) 

    #prompt template
    prompt = PromptTemplate(template="What are 10 suggested  movies for the genre {genre} . give me also the description of those movie ?")

    
    filled_prompt = prompt.format(genre=genre)

    # Format the input 
    messages = [{"role": "user", "content": filled_prompt}]

    # Invoke
    response = llm.invoke(messages)
    
    # Display the response
    if response and hasattr(response, 'content'):
       st.write(response.content)
    else:
       st.write("No content found in the response.")
