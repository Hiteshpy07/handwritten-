# for text (handwritten text) in my pdf, using Indra OCR directly
# pass the page_image_path (or pdf path) to indra_ocr and prepare markdown for it

import indra_ocr

# Initialize the model on CPU
client = indra_ocr.load(threads=4, math=True)

def process_text_path(page_image_path: str) -> str:
    """
    Sends the handwritten page image or PDF to Indra OCR to extract Markdown text.
    """
    # client.convert accepts the path directly!
    markdown_chunks = []
    
    for page_idx, page in client.convert(page_image_path):
        content = page.markdown or page.text        
        if content:
            markdown_chunks.append(content)

    return "\n\n".join(markdown_chunks)