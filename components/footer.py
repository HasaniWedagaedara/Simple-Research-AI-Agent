import streamlit as st


def footer():
    st.markdown(
        """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Powered by Gemini AI | Built with using Streamlit</p>
        <p style='font-size: 12px;'> Tools : Web Search | Wikipedia </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
