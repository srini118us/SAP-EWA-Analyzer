# SAP EarlyWatch PDF Analyzer

A Streamlit application that analyzes SAP EarlyWatch Alert (EWA) PDF reports using AI. It applies OpenAI GPT-4o Vision to interpret charts and graphs, and OpenAI embeddings with a FAISS vector store for semantic search across report content. Upload EWA reports, search for key metrics, and get AI-generated analysis of performance charts and alerts.

> **Status:** Proof of concept. Production path planned via SAP Cloud ALM integration.

## Features

- Upload and process multiple SAP EarlyWatch Alert PDF reports
- Semantic search across report text, charts, and graphs
- AI-powered chart and graph classification and description (OpenAI Vision)
- Clear-all-data control to reset application state
- Streamlit-based interactive UI

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/srini118us/SAP-EWA-Analyzer.git
cd SAP-EWA-Analyzer
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

- Upload one or more SAP EarlyWatch Alert PDF reports using the sidebar.
- Enter a search query (for example: `CPU usage chart`, `performance trends`, `critical alerts`).
- The app displays relevant text and chart/graph results with AI-generated descriptions.
- Use the "Clear All Data" control to reset the app and remove all uploaded data and queries.

## Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4o (vision)

## How It Works

- Extracts text and images from uploaded PDF reports
- Uses OpenAI Vision to classify and describe charts and graphs
- Embeds text chunks with OpenAI embeddings and builds a FAISS vector store for semantic search
- Filters and displays only the charts and graphs relevant to the search query

## Security and Privacy

- Uploaded PDFs are processed locally in a temporary directory, cleared via the "Clear All Data" control.
- The OpenAI API key is kept in a local `.env` file and is never committed or shared.

## License

MIT

## Author

[Srinivas](https://github.com/srini118us)
