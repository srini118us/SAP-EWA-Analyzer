# High-Level Flow: SAP EarlyWatch PDF Analyzer

```mermaid
graph TD
    A[User uploads PDF(s)] --> B[Extract text & images from PDF]
    B --> C[For each image: Use OpenAI Vision to classify & describe]
    B --> D[Split text into chunks]
    D --> E[Embed text chunks with OpenAI embeddings]
    E --> F[Build FAISS vector store]
    C --> G[Store image metadata & AI description]
    F --> H[User enters search query]
    G --> H
    H --> I[Semantic search for text]
    H --> J[Filter images: Only charts/graphs by AI]
    I --> K[Display relevant text]
    J --> L[Display relevant charts/graphs]
    M[Clear All Data] --> N[Delete temp files & reset state]
```

## Steps Explained
1. **User uploads PDF(s):** PDFs are uploaded via the Streamlit sidebar.
2. **Extract text & images:** The app extracts all text and images from each PDF.
3. **Classify images:** Each image is sent to OpenAI Vision to determine if it's a chart/graph and to get a description.
4. **Text chunking & embedding:** The text is split into chunks and embedded using OpenAI embeddings.
5. **Build vector store:** All embeddings are stored in a FAISS vector store for fast semantic search.
6. **User search:** The user enters a query; the app searches both text and images.
7. **Filter images:** Only images classified as charts/graphs by the AI are shown.
8. **Display results:** Relevant text and charts/graphs are displayed with AI-generated descriptions.
9. **Clear All Data:** Deletes all temp files and resets the app state. 