import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from PyPDF2 import PdfReader
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()
import os

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent- Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("Phidata Document AI Analyzer Agent 📄")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Document AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initialize the agent
multimodal_Agent=initialize_agent()

# File uploader
document_file = st.file_uploader(
    "Upload a document file", type=['pdf', 'txt'], help="Upload a document for AI analysis"
)

if document_file is not None:
    try:
        # Process document content
        if document_file.type == "application/pdf":
            pdf_reader = PdfReader(document_file)
            document_text = ""
            for page in pdf_reader.pages:
                document_text += page.extract_text()
        else:
            document_text = document_file.getvalue().decode('utf-8')

        # Display document content
        st.subheader("Document Content Preview:")
        st.text_area("", document_text[:1000] + "...", height=200, disabled=True)

        user_query = st.text_area(
            "What insights are you seeking from the document?",
            placeholder="Ask anything about the document content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the document."
        )

        if st.button("🔍 Analyze Document", key="analyze_document_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the document.")
            else:
                try:
                    with st.spinner("Processing document and gathering insights..."):
                        # Create a context-aware prompt
                        analysis_prompt = f"""
                        Here is the document content:
                        {document_text}

                        Based on the above document content, please answer the following question:
                        {user_query}

                        Provide a detailed, well-structured response using only the information from the document 
                        and relevant web research when necessary. Make sure to cite specific parts of the document 
                        in your response.
                        """

                        # Process with AI agent
                        response = multimodal_Agent.run(
                            analysis_prompt,
                            context={"document_content": document_text}
                        )

                        # Display the result
                        st.subheader("Analysis Result")
                        st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {str(error)}")
                    st.error("Please try again with a different query or document.")

    except Exception as error:
        st.error(f"Error processing document: {str(error)}")
        st.error("Please make sure you've uploaded a valid document file.")

else:
    st.info("Please upload a document file (PDF or TXT) to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)