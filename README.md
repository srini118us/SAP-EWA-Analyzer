# SAP EarlyWatch PDF Analyzer

A modern Streamlit app to analyze SAP EarlyWatch Alert PDF reports. Uses OpenAI's GPT-4o Vision for chart/graph understanding and OpenAI embeddings for semantic search. Upload your SAP PDFs, search for key metrics, and get AI-powered chart/graph analysis.

## Features
- 📂 Upload and process multiple SAP EarlyWatch PDF reports
- 🔍 Semantic search for text and charts/graphs
- 🤖 AI-powered chart/graph classification and description (OpenAI Vision)
- 🧹 Clear all data and reset the app state
- Fast, modern UI with Streamlit

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/srini118us/ewapdf.git
cd ewapdf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the app
```bash
streamlit run appnew.py
```

## Usage
- Upload one or more SAP EarlyWatch PDF reports using the sidebar.
- Enter a search query (e.g., `CPU usage chart`, `performance trends`, `critical alerts`).
- The app will display relevant text and chart/graph results, with AI-generated descriptions.
- Use the "🧹 Clear All Data" button to reset the app and remove all uploaded data and queries.

## Requirements
- Python 3.8+
- OpenAI API key with access to GPT-4o (vision)

## High-Level Functionality
- Extracts text and images from PDFs
- Uses OpenAI Vision to classify and describe charts/graphs
- Embeds text chunks with OpenAI embeddings and builds a FAISS vector store for semantic search
- Filters and displays only relevant charts/graphs based on AI classification

## Security & Privacy
- Uploaded PDFs are processed locally and stored in a temporary directory, which is cleared when you use the "Clear All Data" button.
- Your OpenAI API key is kept private in the `.env` file and never shared.

## License
MIT

---

## Contributing
Pull requests and issues are welcome!

---

## Author
- [Srinivas](https://github.com/srini118us) 