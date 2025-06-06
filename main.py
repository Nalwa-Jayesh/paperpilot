import streamlit as st
import logging
from query_engine import QueryEngine
from models import LanguageModel
import os
from pathlib import Path
import tempfile
import hashlib

# Configure page
st.set_page_config(
    page_title="PaperPilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* General adjustments */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1e212b; /* Darker background for sidebar */
        color: #ffffff; /* White text */
        padding-top: 2rem;
    }
    .sidebar .stRadio > label > div:first-child {
        padding-right: 0.5rem;
    }
    .sidebar .stRadio div[data-baseweb="radio"] > label {
        margin-bottom: 0.5rem; /* Add space between radio options */
    }
    .sidebar .stMarkdown h3 {
        color: #ffffff; /* White color for sidebar headers */
        margin-bottom: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; /* Reduce gap between tabs */
    }
    .stTabs [data-baseweb="tab"] {
        height: auto; /* Auto height */
        padding: 0.75rem 1.5rem; /* Adjusted padding */
        background-color: transparent; /* Transparent background for unselected tabs */
        border-bottom: 2px solid transparent; /* Transparent bottom border */
        margin-bottom: -2px; /* Counteract border to keep layout consistent */
        color: #adb5bd; /* Lighter color for unselected text */
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent; /* Transparent background for selected tab */
        color: #4CAF50; /* Green color for selected text */
        border-bottom: 2px solid #4CAF50; /* Green bottom border for selected tab */
    }
    
    /* Chat message styling */
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        width: 90%;
    }
    .chat-message.user {
        background-color: #262730;
        align-self: flex-end;
    }
    .chat-message.assistant {
        background-color: #475063;
        align-self: flex-start;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
    
    /* Source Display Styling */
    .source-box {
        background-color: #2e3440;
        border: 1px solid #4c566a;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 0.9em;
        line-height: 1.6;
        color: #d8dee9;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        border-bottom: 1px solid #4c566a;
        padding-bottom: 8px;
    }
    .source-score {
        font-size: 0.8em;
        color: #a3be8c;
        font-weight: bold;
    }
    .source-metadata {
        margin-top: 10px;
        margin-bottom: 10px;
        color: #b48ead;
        font-size: 0.85em;
    }
     .source-metadata-item {
        margin-right: 15px;
     }
    .source-content {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #4c566a;
        white-space: pre-wrap; /* Preserve line breaks */
    }
    
    /* Streamlit overrides for chat interface */
    .stTextInput > div > div > input {
        padding: 12px 20px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }
    .stButton>button {
        padding: 12px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

logger = logging.getLogger(__name__)

@st.cache_data
def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for the file content."""
    return hashlib.md5(file_content).hexdigest()

@st.cache_resource
def get_query_engine():
    """Initialize and cache the query engine."""
    return QueryEngine()

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files."""
    if not uploaded_files:
        return

    # Create a temporary directory to save uploaded files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        file_paths = []
        for file in uploaded_files:
            # Generate hash of file content
            file_hash = get_file_hash(file.getvalue())
            
            # Check if this file has already been processed
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = set()
            
            if file_hash not in st.session_state.processed_files:
                # Save the uploaded file to the temporary directory
                file_path = tmp_dir_path / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                file_paths.append(str(file_path))
                st.session_state.processed_files.add(file_hash)

        # Add documents to the index
        if file_paths:
            with st.spinner("Indexing documents..."):
                results = st.session_state.query_engine.index.add_documents(file_paths)
                st.success(f"✅ Indexed {results['processed']} new documents with {results['new_chunks']} chunks.")

def display_chat_message(message: str, is_user: bool = False):
    """Display a chat message with appropriate styling."""
    # Use columns to push user message to the right
    col1, col2 = st.columns([1, 10])
    if is_user:
         with col2:
             st.markdown(f"""
                 <div class="chat-message user">
                     <div>{message}</div>
                 </div>
             """, unsafe_allow_html=True)
    else:
        # Use columns to keep assistant message to the left
        with col1:
             st.markdown("**🤖 Assistant:**") # Optional: Add an icon or label
        with col2:
            st.markdown(f"""
                <div class="chat-message assistant">
                    <div>{message}</div>
                </div>
            """, unsafe_allow_html=True)


def main():
    # Initialize session state
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = get_query_engine()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("📚 PaperPilot")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        PaperPilot helps you:
        - 📄 Upload and analyze PDF documents
        - ❓ Ask questions about your documents
        - 🔍 Find information in documents
        """)
        st.markdown("---")
        st.markdown("### Settings")
        st.markdown("Using local Ollama model: `gemma2:2b`") # Indicate the model being used
    
    # Main content area - Only Document Q&A section
    st.header("Ask Questions About Your Documents")
    
    # File upload in the main content area
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to analyze"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    # Chat interface
    user_query = st.chat_input("Enter your question about the uploaded documents...")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message['content'], message['is_user'])
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({'content': user_query, 'is_user': True})
        display_chat_message(user_query, True)
        
        with st.spinner("Thinking..."):
            result = st.session_state.query_engine.query(user_query)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({'content': result['answer'], 'is_user': False})
            display_chat_message(result['answer'])
            
            # Display sources if available
            if result['sources']:
                with st.expander("📚 View Sources", expanded=False):
                    # First show the context
                    st.markdown("### Context")
                    st.markdown(f"<div class='source-box'>{result['context']}</div>", unsafe_allow_html=True)
                    
                    # Then show individual sources
                    st.markdown("### Sources")
                    for i, source_item in enumerate(result['sources'], 1):
                        # Get metadata from the source item
                        source_doc = source_item.get('source', 'Unknown Document')
                        source_page = source_item.get('page', 'Unknown Page')
                        chunk_text = source_item.get('chunk_text', 'Content not available')
                        similarity_score = source_item.get('score', 0)
                        
                        # Format similarity score as percentage
                        score_percentage = f"{similarity_score * 100:.1f}%"
                        
                        st.markdown(f"""
                            <div class="source-box">
                                <div class="source-header">
                                    <h3 style="margin: 0;">Source {i}</h3>
                                    <span class="source-score">Relevance: {score_percentage}</span>
                                </div>
                                <div class="source-metadata">
                                    <span class="source-metadata-item">📄 {source_doc}</span>
                                    <span class="source-metadata-item">📑 Page {source_page}</span>
                                </div>
                                <div class="source-content">
                                    {chunk_text}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 