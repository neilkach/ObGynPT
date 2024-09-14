import streamlit as st
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import llms
from openai import OpenAI

from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path
from textwrap import wrap

from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def generate_response(query):
    #docs = [PdfReader(pdf)]
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # pdf_qa = ConversationalRetrievalChain.from_llm(llms.OpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.5) , vectordb.as_retriever(), memory=memory)
    # retrieved_docs = retriever.get_relevant_documents(query)
    # context = retrieved_docs[0].page_content
    # print(context)
    #result = pdf_qa({'question': query}) - not supported in latest openai/langchain versions
    # response = client.chat.completions.create(model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant. For each query, use the text prior to the question \
    #         as context for your answer."},
    #     {"role": "user", "content": context + 'According to the context I\'m providing, ' + query},
    # ])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    response = ""
    for chunk in rag_chain.stream(query):
        response += chunk
        print(chunk, end="", flush=True)

    with open('log.txt', 'a') as f:  
        #f.write('Context: ' + '\n'.join(wrap(context,120)) + '\n')
        f.write('Query: ' + query + '\n')
        f.write('Answer: ' + response + '\n\n')
        #f.write('Answer: ' + '\n'.join(wrap(response.choices[0].message.content,120)) + '\n\n')
    #return 'Answer: ' + response.choices[0].message.content
    return 'Answer: ' + response

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
if uploaded_file:
    save_folder = './'
    save_path = Path(save_folder, '_'.join(uploaded_file.name.split(' ')))
    if not save_path.exists():
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())
    pdf_path = './'+'_'.join(uploaded_file.name.split(' '))
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    embeddings = OpenAIEmbeddings(chunk_size=1)
    vectordb = Chroma.from_documents(docs, embedding=embeddings,
                                 persist_directory=".")
    vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    #openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    #if submitted and openai_api_key.startswith('sk-'):
    if submitted:   
        with st.spinner('Calculating...'):
            #response = generate_response(uploaded_file, openai_api_key, query_text)
            response = generate_response(query_text)
            result.append(response)
            #del openai_api_key
if len(result):
    st.info(response)