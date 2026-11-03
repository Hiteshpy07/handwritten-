from hpack import struct
from yaml import FullLoader
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os 
import uuid
from dotenv import load_dotenv
from google import genai


from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct




load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant_client= QdrantClient("http://localhost:6333") 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_dimension = 384 
# learn more about vector dimension, what happens if it varies and all ???


collection_name = "handwrittennotes_to_rag"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
)


prompt = """
Transcribe all handwritten text from this image into clean GitHub-flavored Markdown:
- Preserve headings, bullet points, and tables.
- Convert any mathematical formulas to LaTeX (using $ or $$).
- Ignore crossed-out scribbles or stray marks.
- Output pure markdown without wrapping commentary.
"""



images = convert_from_path('./test.pdf',dpi=200)
print(images)
full_markdown=[]

for i,img in enumerate(images):
    response = client.models.generate_content(
    model="gemini-3.0-flash",
    contents=[img, prompt]
    )

    full_markdown.append(f"## page {i+1}\n{response.text}\n")



print(full_markdown)
print("################################################")   


text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)


# chunks =text_splitter(full_markdown). its failing currently 
# ✅ Fixed
combined_markdown = "\n".join(full_markdown)
chunks = text_splitter.split_text(combined_markdown)

# just to maek it work , will give proper formatting to it all later


for i, chunk in enumerate(chunks): 
    print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
    print(chunk)


points=[]
for index,chunk in enumerate(chunks):
    vector = embedding_model.encode(chunk).tolist()


    point=PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={            
            "chunk_id": index,
            "text": chunk,
            "source": "./test.pdf"
        }

    )
    points.append(point)

qdrant_client.upsert(collection_name=collection_name, points=points)
print(f"Successfully chunked and stored {len(points)} vectors in Qdrant!")



query=input("ask questions:")
query_vector=embedding_model.encode(query).tolist()


search_results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_vector,       # 👈 use 'query' instead of 'query_vector'
    limit=2
)

for result in search_results.points:  # 👈 iterate over .points
    print(f"\n[Score: {result.score:.3f}] Found Match:")
    print(f"Text: {result.payload['text']}")

'''
things to improve in this code 

1. prompt- give a m=automated prompt , just staright convert the text to md , but can asdd falbacks for text and diagrams , from gemini vision to ocr model, etc to increase efficiency
1-ans-there are special multi-modal models -CLIP which can handle both tex and images , the ocr and gemini vision is not a accurate path
  The better way is not using OCR and Gemini vision differently. A more optimized way is using a CLIP model, which is both vision transformer and text transformer. I will send the handwritten PDF and separate handwritings because CLIP is not very good for detecting handwriting and converting it into texts and vectors. So, what I did is I will separate the images and give it to CLIP vision transformer, and I will separate the handwriting part and give it to OCR models like Indra OCR, which will convert it into perfect and very accurate typed texts, which will be then converted into vectors, and the images will be converted into vectors by vision transformer. This is the plan for me.
2. chunk size
3. chunk overlap
experiment cthe chunck size and chunk overlap , adn find out th seett pint with most efficent way and accurate
4. vector dimension
learn more about it -what is it , learn more about vectors
5. distance metric
6. qdrant client
experimnet different types of vertorisations and what will happen in each case
7. embedding model

decorete it more and give the output in more good json format.and usable way
8. search results
9. search results score
10. search results text

'''