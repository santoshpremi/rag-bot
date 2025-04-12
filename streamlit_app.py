import streamlit as st
import requests

st.set_page_config(page_title="COVID-19 Chatbot", page_icon="🦠")
st.title("🦠 COVID-19 Information Chatbot")
st.caption("Powered by RAG and Fine-Tuned LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a COVID-19 question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": prompt}
        ).json()
        answer = response["answer"]
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})