# SAP EarlyWatch PDF Analyzer
#
# High-level functionality:
# - Upload SAP EarlyWatch PDF(s)
# - Extract text and images from PDFs
# - Use OpenAI Vision to classify and describe charts/graphs
# - Embed text chunks with OpenAI embeddings and build a FAISS vector store
# - Semantic search for both text and charts/graphs
# - Clear all data and reset app state

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import shutil
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import re
import base64
from datetime import datetime
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# Configure Streamlit page
st.set_page_config(
    page_title="SAP Report Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Utility: Clean up whitespace in text
def clean_text_spacing(text):
    return re.sub(r'\s+', ' ', text).strip()

# Cache OpenAI client and embeddings for performance
@st.cache_resource(show_spinner=False)
def load_models():
    """Lazy load OpenAI client and embeddings"""
    with st.spinner("Initializing OpenAI..."):
        client = OpenAI(api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return client, embeddings

# Utility: Convert image bytes to base64 for OpenAI Vision
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

class DocumentProcessor:
    """
    Handles PDF processing, text/image extraction, AI analysis, and search.
    """
    def __init__(self):
        # In-memory state for current session
        self.client = None
        self.embeddings = None
        self.vector_store = None
        self.temp_dir = Path("temp_storage")
        self.temp_dir.mkdir(exist_ok=True)
        self.images = []
        self.image_types = ['chart', 'graph', 'pie', 'bar', 'line', 'performance', 'trend']
        self._models_loaded = False

    def _ensure_models_loaded(self):
        if not self._models_loaded:
            self.client, self.embeddings = load_models()
            self._models_loaded = True

    def _get_image_description(self, image_bytes):
        """
        Use OpenAI Vision to classify and describe the image.
        Returns a string description or error message.
        """
        self._ensure_models_loaded()
        try:
            base64_image = encode_image_to_base64(image_bytes)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an SAP report image classifier. "
                                    "Is this image a chart or graph? If yes, reply with the type (e.g., bar chart, line graph, pie chart) and a brief description of the key information. "
                                    "If not, reply exactly: 'Not a chart or graph.'"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error getting image description: {str(e)}")
            return "Rate limit error"

    def extract_text_and_images_from_pdf(self, pdf_path):
        """
        Extract all text and images from a PDF file.
        For each image, get AI-generated description.
        """
        text = ""
        images = []

        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, 1):
            text += page.get_text()
            for img_index, img in enumerate(page.get_images(full=True), 1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_rect = page.get_image_rects(img)[0]
                surrounding_text = page.get_text("text", clip=img_rect)
                image_description = self._get_image_description(image_bytes)
                images.append({
                    'bytes': image_bytes,
                    'page': page_num,
                    'type': self._determine_image_type(surrounding_text + " " + image_description),
                    'context': surrounding_text,
                    'description': image_description,
                    'position': img_index
                })

        # Additional text extraction with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        return text, images

    def _determine_image_type(self, text):
        text_lower = text.lower()
        for img_type in self.image_types:
            if img_type in text_lower:
                return img_type
        return 'unknown'

    def process_document(self, uploaded_file):
        """
        Process an uploaded PDF: extract, analyze, and index for search.
        """
        with st.spinner("Processing document and analyzing images..."):
            temp_path = self.temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            text, images = self.extract_text_and_images_from_pdf(str(temp_path))
            self.images.extend(images)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Create FAISS index using OpenAI embeddings
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    chunks,
                    self.embeddings,
                    metadatas=[{"source": str(i)} for i in range(len(chunks))]
                )
            else:
                self.vector_store.add_texts(
                    chunks,
                    metadatas=[{"source": str(i)} for i in range(len(chunks))]
                )

            return len(chunks), len(images)

    def search_content(self, query, k=5):
        """
        Semantic search for both text and relevant charts/graphs.
        """
        if self.vector_store is None:
            return [], []
        text_results = self.vector_store.similarity_search(query, k=k)
        image_results = self._search_images(query)
        return text_results, image_results

    def _search_images(self, query):
        """
        Filter and rank images: only show charts/graphs as classified by AI.
        """
        query_terms = query.lower().split()
        relevant_images = []
        for img_data in self.images:
            score = 0
            description = img_data.get('description', '').lower()
            if not description or "rate limit" in description:
                continue
            if description.startswith('not a chart or graph'):
                continue
            if not ("chart" in description or "graph" in description):
                continue
            if any(term in description for term in query_terms):
                score += 3
            if any(term in img_data['type'] for term in query_terms):
                score += 2
            if any(term in img_data['context'].lower() for term in query_terms):
                score += 1
            if any(term in query.lower() for term in ['chart', 'graph', 'pie', 'bar', 'line', 'trend']):
                if img_data['type'] != 'unknown':
                    score += 1
            if score > 0:
                relevant_images.append((img_data, score))
        relevant_images.sort(key=lambda x: x[1], reverse=True)
        return [img_data for img_data, _ in relevant_images[:5]]

    def clear_storage(self):
        """
        Delete all temp files and reset in-memory state.
        """
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        self.vector_store = None
        self.images = []

# --- Streamlit UI ---
def main():
    st.title("📊 SAP EarlyWatch Alert Report Analyzer")

    # Initialize processor in session state
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()

    # Sidebar for file upload and clear
    with st.sidebar:
        st.header("📂 Upload PDF Reports")
        uploaded_files = st.file_uploader("Upload SAP EarlyWatch PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    chunks, img_count = st.session_state.processor.process_document(file)
                    st.success(f"✅ {chunks} text chunks and {img_count} images extracted.")
        st.markdown("---")
        if st.button("🧹 Clear All Data"):
            st.session_state.processor.clear_storage()
            st.session_state.processor = DocumentProcessor()  # Reset the processor instance
            st.session_state['query'] = ""  # Reset the search box
            st.success("All data cleared.")

    # Main content area: search and results
    st.header("🔍 Search Document Content")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query",
            placeholder="Try: 'CPU usage chart' or 'performance trends' or 'critical alerts'",
            key="query"
        )
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    if search_button and query:
        with st.spinner("Searching..."):
            text_results, image_results = st.session_state.processor.search_content(query)
            # Display text results
            if text_results:
                st.markdown("### 📝 Text Results")
                for i, result in enumerate(text_results, 1):
                    cleaned = clean_text_spacing(result.page_content)
                    st.markdown(
                        f"""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <p style='font-size:14px; line-height:1.4; margin:0;'>{cleaned}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            # Display image results
            if image_results:
                st.markdown("### 📊 Chart & Graph Results")
                cols = st.columns(min(3, len(image_results)))
                for idx, img_data in enumerate(image_results):
                    with cols[idx % 3]:
                        try:
                            image = Image.open(io.BytesIO(img_data['bytes']))
                            st.image(image, 
                                   caption=f"Page {img_data['page']} - {img_data['type'].title()} Chart",
                                   use_container_width=True)
                            if img_data['description']:
                                st.markdown(
                                    f"""
                                    <div style='background-color: #f8f9fa; padding: 8px; border-radius: 4px; margin-top: 5px;'>
                                        <p style='font-size:12px; color: #444; margin:0;'><strong>AI Description:</strong> {img_data['description']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            if img_data['context']:
                                st.markdown(
                                    f"<p style='font-size:12px; color: #666; margin-top: 5px;'><strong>Context:</strong> {img_data['context'][:100]}...</p>",
                                    unsafe_allow_html=True
                                )
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            if not text_results and not image_results:
                st.info("No relevant results found. Try different search terms or check if the document has been uploaded.")

if __name__ == "__main__":
    main()
