import streamlit as st
from ingest import ingest_file
from retriever import retrieve
from agent import run_agent
from utils import save_as_pdf
import tempfile
import os

from fpdf import FPDF

def text_to_pdf_fpdf(text_content):
    """
    Converts a string of text content to a PDF using FPDF and returns it as bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Encode to latin-1 and replace unsupported characters to prevent errors
    encoded_text = text_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=encoded_text)
    return pdf.output(dest='S').encode('latin-1')

st.set_page_config(page_title="Deep Researcher Agent", layout="wide")
st.title("🧠 Deep Researcher Agent")
st.markdown("Upload a document and ask queries. The system retrieves relevant chunks and provides reasoning.")

# File Upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    st.success(f"File uploaded: {uploaded_file.name}")
    chunks_count = ingest_file(file_path)
    st.info(f"Document ingested into {chunks_count} chunks.")

# Query Input
query = st.text_input("Enter your research query:")
if st.button("Run Research") and query:
    if not uploaded_file:
        st.warning("Please upload a document first.")
    else:
        docs = retrieve(query, top_k=5)
        st.subheader("Retrieved Chunks:")
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {d[:300]}...")

        st.subheader("Agent Response:")
        response = run_agent(query, docs)
        st.write(response)

        # Download PDF
        pdf_bytes = text_to_pdf_fpdf(response)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="research_output.pdf",
            mime="application/pdf",
        )
