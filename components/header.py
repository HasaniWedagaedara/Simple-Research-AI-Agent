import streamlit as st


def header():
    st.title("AI Research Assistant")
    st.markdown(
        '<p class="subtitle">Ask me anything and I\'ll research it for you using web search and Wikipedia!</p>',
        unsafe_allow_html=True,
    )
