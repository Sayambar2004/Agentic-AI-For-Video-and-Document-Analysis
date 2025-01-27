import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
from PyPDF2 import PdfReader
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
load_dotenv()
import os

# Configure API
API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="AI Content Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 20px;
        color: #2e7d32;
    }
    .bot-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        background-color: #f5f5f5;
    }
    .creator-section {
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 10px;
        margin-top: 30px;
    }
    .stTextArea textarea {
        height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize agents
@st.cache_resource
def initialize_video_agent():
    return Agent(
        name="Video AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

@st.cache_resource
def initialize_document_agent():
    return Agent(
        name="Document AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Main title
st.markdown("<h1 class='main-title'>AI Content Analysis Hub</h1>", unsafe_allow_html=True)
st.markdown("### Powered by PhiData's Advanced AI Technology")

# Navigation
selected_bot = st.sidebar.radio(
    "Choose Your Analysis Tool",
    ["Home", "Video Analyzer", "Document Analyzer", "About The Creator"]
)

if selected_bot == "Home":
    st.markdown("""
    ## Welcome to AI Content Analysis Hub! 🚀
    
    This platform offers two powerful AI-powered tools:
    
    ### 🎥 Video Analyzer
    - Upload and analyze video content
    - Get detailed insights and summaries
    - Ask questions about video content
    
    ### 📄 Document Analyzer
    - Process PDF and TXT documents
    - Extract key information
    - Get comprehensive analysis
    
    Choose your desired tool from the sidebar to get started!
    """)

elif selected_bot == "Video Analyzer":
    st.title("Video Content Analyzer 🎥")
    multimodal_Agent = initialize_video_agent()
    
    video_file = st.file_uploader(
        "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the video."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """
                        )

                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)
        
elif selected_bot == "Document Analyzer":
    st.title("Document Content Analyzer 📄")
    multimodal_Agent = initialize_document_agent()
    
    document_file = st.file_uploader(
        "Upload a document file", type=['pdf', 'txt'], help="Upload a document for AI analysis"
    )

    if document_file is not None:
        try:
            if document_file.type == "application/pdf":
                pdf_reader = PdfReader(document_file)
                document_text = ""
                for page in pdf_reader.pages:
                    document_text += page.extract_text()
            else:
                document_text = document_file.getvalue().decode('utf-8')

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
                            analysis_prompt = f"""
                            Here is the document content:
                            {document_text}

                            Based on the above document content, please answer the following question:
                            {user_query}

                            Provide a detailed, well-structured response using only the information from the document 
                            and relevant web research when necessary. Make sure to cite specific parts of the document 
                            in your response.
                            """

                            response = multimodal_Agent.run(
                                analysis_prompt,
                                context={"document_content": document_text}
                            )

                            st.subheader("Analysis Result")
                            st.markdown(response.content)

                    except Exception as error:
                        st.error(f"An error occurred during analysis: {str(error)}")
                        st.error("Please try again with a different query or document.")

        except Exception as error:
            st.error(f"Error processing document: {str(error)}")
            st.error("Please make sure you've uploaded a valid document file.")

elif selected_bot == "About The Creator":
    st.markdown("""
    ## About The Creator
    
    ### Sayambar Roy Chowdhury
    
    A passionate 2nd year Computer Science Engineering Student with a keen interest in developing innovative AI and ML applications.
    
    #### Skills & Interests:
    * Artificial Intelligence
    * Machine Learning
    * Application Development
    * Innovation in Technology
    
    This application is a testament to my commitment to creating practical AI solutions that can help users analyze and understand content more effectively.
    """)

     # Add social links in a container
    with st.container():
        st.markdown("""
        ---
        ### Connect With Me
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://www.github.com/Sayambar2004)
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/)
        
        📧 **Email:** [sayambarroychowdhury@gmail.com](mailto:sayambarroychowdhury@gmail.com)
        """)