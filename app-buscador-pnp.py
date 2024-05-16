import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import json

st.title("Buscador de vídeos da PNP")
st.subheader("Projeto localizador de vídeos da PNP por busca semântica.")

def find_best_passage(model, query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")

  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])  
  idx = np.argmax(dot_products)
  return dataframe.iloc[idx]

def parse_embedding(embedding_text):
    try:
        return json.loads(embedding_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar o embedding: {e}")

#set o modelo de generative AI
GOOGLE_API_KEY = st.sidebar.text_input("Digite sua chave do Gemini:", type="password")
btn_set_model = st.sidebar.button("Set chave Gemini")
st.session_state.model = 'models/text-embedding-004'

if btn_set_model:
    #set o modelo de chat
    genai.configure(api_key=GOOGLE_API_KEY)
    st.sidebar.success("Modelo gemini configurado com sucesso.")

    #set base de videos
    st.session_state.df_video =  pd.read_csv('videos_youtube.csv')
    # Aplica a conversão em todos os embeddings
    st.session_state.df_video['Embeddings'] = st.session_state.df_video['Embeddings'].apply(parse_embedding)
    
    st.dataframe(st.session_state.df_video)
    st.sidebar.success("base de vídeos configurado com sucesso.")

prompt = st.text_input("Digite sua pergunta sobre a PNP:")
st.caption("Exemplo de pergunta: como resolver evasão zero?")
btn_gemini = st.button("Enviar pergunta")

if btn_gemini:
    if 'df_video' in st.session_state:        
        df_video = find_best_passage(st.session_state.model, prompt, st.session_state.df_video)

        #st.dataframe(df_video)

        vd_auxiliar = st.video(df_video['link'])
        texto = f"Nome: {df_video['nome']} - Descrição: {df_video['texto']}"
        st.caption(df_video['texto']) 

