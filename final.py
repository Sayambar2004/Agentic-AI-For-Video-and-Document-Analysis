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

# Enhanced Custom CSS for dark theme
st.markdown("""
    <style>
    /* Center align headers */
    h1, h2, h3 {
        text-align: center !important;
    }

    /* Gradient border animation */
    @keyframes borderGlow {
        0% { border-color: rgba(255,255,255,0.2); }
        50% { border-color: rgba(255,255,255,0.5); }
        100% { border-color: rgba(255,255,255,0.2); }
    }

    /* Card styling */
    .stCard {
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stCard:hover {
        transform: translateY(-5px);
        background-color: rgba(255, 255, 255, 0.08);
    }

    /* Tool cards */
    .tool-card {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: borderGlow 2s infinite;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        padding: 10px 25px;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }

    /* Text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 120px;
    }

    /* About section styling */
    .about-section {
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }

    /* Social links styling */
    .social-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    .social-links a {
        transition: transform 0.3s ease;
    }
    .social-links a:hover {
        transform: translateY(-3px);
    }

    /* Success message styling */
    .success-message {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        background-color: rgba(25, 135, 84, 0.2);
        border: 1px solid rgba(25, 135, 84, 0.3);
        text-align: center;
    }

    /* Warning message styling */
    .warning-message {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        background-color: rgba(255, 193, 7, 0.2);
        border: 1px solid rgba(255, 193, 7, 0.3);
        text-align: center;
    }

    /* Error message styling */
    .error-message {
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        background-color: rgba(220, 53, 69, 0.2);
        border: 1px solid rgba(220, 53, 69, 0.3);
        text-align: center;
    }

    /* Centered container */
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin: 20px 0;
    }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        padding: 8px 15px;
        margin: 5px;
        border-radius: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
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

# Main title with centered container
st.markdown("""
    <div class="centered-container">
        <h1>🤖 AI Content Analysis Hub</h1>
        <p style="font-size: 1.2em; opacity: 0.8;">Powered by PhiData's Advanced AI Technology</p>
    </div>
""", unsafe_allow_html=True)

# Navigation
selected_bot = st.sidebar.radio(
    "Choose Your Analysis Tool",
    ["Home", "Video Analyzer", "Document Analyzer", "About The Creator"]
)

if selected_bot == "Home":
    st.markdown("""
    <div class="centered-container">
        <h2>Welcome to AI Content Analysis Hub! 🚀</h2>
        <p style="font-size: 1.2em; margin-bottom: 30px;">Your all-in-one solution for intelligent content analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='tool-card'>
            <h3>🎥 Video Analyzer</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin: 10px 0;">✨ Upload and analyze video content</li>
                <li style="margin: 10px 0;">📊 Get detailed insights and summaries</li>
                <li style="margin: 10px 0;">❓ Ask questions about video content</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='tool-card'>
            <h3>📄 Document Analyzer</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li style="margin: 10px 0;">📑 Process PDF and TXT documents</li>
                <li style="margin: 10px 0;">🔍 Extract key information</li>
                <li style="margin: 10px 0;">📈 Get comprehensive analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif selected_bot == "Video Analyzer":
    st.markdown("<h2 style='text-align: center;'>Video Content Analyzer 🎥</h2>", unsafe_allow_html=True)
    multimodal_Agent = initialize_video_agent()
    
    with st.container():
        video_file = st.file_uploader(
            "Upload a video file", type=['mp4', 'mov', 'avi'], 
            help="Upload a video for AI analysis"
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
                    st.markdown('<div class="warning-message">Please enter a question or insight to analyze the video.</div>', unsafe_allow_html=True)
                else:
                    try:
                        with st.spinner("🔄 Processing video and gathering insights..."):
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

                        st.markdown('<div class="success-message">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
                        st.markdown("<h3 style='text-align: center;'>Analysis Result</h3>", unsafe_allow_html=True)
                        st.markdown(response.content)

                    except Exception as error:
                        st.markdown(f'<div class="error-message">❌ An error occurred during analysis: {error}</div>', unsafe_allow_html=True)
                    finally:
                        Path(video_path).unlink(missing_ok=True)

elif selected_bot == "Document Analyzer":
    st.markdown("<h2 style='text-align: center;'>Document Content Analyzer 📄</h2>", unsafe_allow_html=True)
    multimodal_Agent = initialize_document_agent()
    
    with st.container():
        document_file = st.file_uploader(
            "Upload a document file", type=['pdf', 'txt'], 
            help="Upload a document for AI analysis"
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
                        st.markdown('<div class="warning-message">Please enter a question or insight to analyze the document.</div>', unsafe_allow_html=True)
                    else:
                        try:
                            with st.spinner("🔄 Processing document and gathering insights..."):
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

                                st.markdown('<div class="success-message">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
                                st.markdown("<h3 style='text-align: center;'>Analysis Result</h3>", unsafe_allow_html=True)
                                st.markdown(response.content)

                        except Exception as error:
                            st.markdown(f'<div class="error-message">❌ An error occurred during analysis: {str(error)}</div>', unsafe_allow_html=True)

            except Exception as error:
                st.markdown(f'<div class="error-message">❌ Error processing document: {str(error)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="error-message">Please make sure you\'ve uploaded a valid document file.</div>', unsafe_allow_html=True)

elif selected_bot == "About The Creator":
    st.markdown("<h2 style='text-align: center;'>About The Creator</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Sayambar Roy Chowdhury</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <p style='text-align: center; font-size: 1.1em; margin: 20px 0;'>
            A passionate 2nd year Computer Science Engineering Student with a keen interest in developing innovative AI and ML applications.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center; margin: 20px 0;'>Skills & Interests</h4>", unsafe_allow_html=True)
    
    # Skills container with custom styling
    skills_html = """
        <div style='text-align: center; margin: 20px 0;'>
            <div style='display: inline-block; margin: 5px; padding: 8px 15px; background-color: rgba(255, 255, 255, 0.1); border-radius: 20px;'>Artificial Intelligence</div>
            <div style='display: inline-block; margin: 5px; padding: 8px 15px; background-color: rgba(255, 255, 255, 0.1); border-radius: 20px;'>Machine Learning</div>
            <div style='display: inline-block; margin: 5px; padding: 8px 15px; background-color: rgba(255, 255, 255, 0.1); border-radius: 20px;'>Application Development</div>
            <div style='display: inline-block; margin: 5px; padding: 8px 15px; background-color: rgba(255, 255, 255, 0.1); border-radius: 20px;'>Innovation in Technology</div>
        </div>
    """
    st.markdown(skills_html, unsafe_allow_html=True)
    
    st.markdown("""
        <p style='text-align: center; margin: 20px 0;'>
            This application is a testament to my commitment to creating practical AI solutions that can help users analyze and understand content more effectively.
        </p>
    """, unsafe_allow_html=True)
    
    # Social Links section
    st.markdown("<h3 style='text-align: center; margin-top: 30px;'>Connect With Me</h3>", unsafe_allow_html=True)
    
    # Center-aligned container for social links
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align: center;'>
                <a href="https://www.github.com/Sayambar2004" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
                </a>
                &nbsp;&nbsp;
                <a href="https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/" target="_blank">
                    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
                </a>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <p style='text-align: center; margin-top: 20px;'>
                📧 <a href="mailto:sayambarroychowdhury@gmail.com" style="color: inherit; text-decoration: none;">sayambarroychowdhury@gmail.com</a>
            </p>
        """, unsafe_allow_html=True)