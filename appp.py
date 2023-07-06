import streamlit as st
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import PandasAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = ""

def chat_with_csv(df,prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm)
    result = pandas_ai.run(df,prompt=prompt,show_code=True,is_conversational_answer=True)
    return result

st.set_page_config(layout='wide')
st.title("Multiple-CSV ChatApp powered by LLM")

# Upload multiple CSV files
input_csvs = st.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

if input_csvs:
    # Select a CSV file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    #load and display the selected csv file 
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data,use_container_width=True)

    #Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    #Perform analysis
    if input_text:
        if st.button("Chat with csv"):
            st.info("Your Query: "+ input_text)
            result = chat_with_csv(data,input_text)
            st.success(result)




     
