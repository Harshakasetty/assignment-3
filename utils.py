from fpdf import FPDF
from io import BytesIO

def save_as_pdf(content):
    """
    Generate a PDF in-memory and return a BytesIO buffer.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Split content into lines
    lines = content.split("\n")
    for line in lines:
        pdf.multi_cell(0, 8, line)

    # Save to in-memory bytes buffer
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)  # Important: reset cursor to start
    return pdf_buffer
