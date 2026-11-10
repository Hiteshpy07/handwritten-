import indra_ocr

_client = None

def get_indra_client():
    global _client
    if _client is None:
        _client = indra_ocr.load(threads=2, math=True)
    return _client

def process_text_path(page_image_path: str) -> str:
    """
    Sends the handwritten page image or PDF to Indra OCR to extract Markdown text.
    """
    client = get_indra_client()
    markdown_chunks = []
    
    for _, page in client.convert(page_image_path):
        content = page.markdown or page.text or ""
        if content.strip():
            markdown_chunks.append(content)
            
    return "\n\n".join(markdown_chunks)
