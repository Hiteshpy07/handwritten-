# from hpack import struct
# from yaml import FullLoader
# from pdf2image import convert_from_path, convert_from_bytes
# from pdf2image.exceptions import (
#     PDFInfoNotInstalledError,
#     PDFPageCountError,
#     PDFSyntaxError
# )
# import os 
# import uuid
# from dotenv import load_dotenv
# from google import genai


# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct




# load_dotenv()
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# qdrant_client= QdrantClient("http://localhost:6333") 

# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# vector_dimension = 384 
# # learn more about vector dimension, what happens if it varies and all ???


# collection_name = "handwrittennotes_to_rag"
# qdrant_client.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
# )


# prompt = """
# Transcribe all handwritten text from this image into clean GitHub-flavored Markdown:
# - Preserve headings, bullet points, and tables.
# - Convert any mathematical formulas to LaTeX (using $ or $$).
# - Ignore crossed-out scribbles or stray marks.
# - Output pure markdown without wrapping commentary.
# """



# images = convert_from_path('./test.pdf',dpi=200)
# print(images)
# full_markdown=[]

# for i,img in enumerate(images):
#     response = client.models.generate_content(
#     model="gemini-3.0-flash",
#     contents=[img, prompt]
#     )

#     full_markdown.append(f"## page {i+1}\n{response.text}\n")



# print(full_markdown)
# print("################################################")   


# text_splitter=RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False
# )


# # chunks =text_splitter(full_markdown). its failing currently 
# # ✅ Fixed
# combined_markdown = "\n".join(full_markdown)
# chunks = text_splitter.split_text(combined_markdown)

# # just to maek it work , will give proper formatting to it all later


# for i, chunk in enumerate(chunks): 
#     print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
#     print(chunk)


# points=[]
# for index,chunk in enumerate(chunks):
#     vector = embedding_model.encode(chunk).tolist()


#     point=PointStruct(
#         id=str(uuid.uuid4()),
#         vector=vector,
#         payload={            
#             "chunk_id": index,
#             "text": chunk,
#             "source": "./test.pdf"
#         }

#     )
#     points.append(point)

# qdrant_client.upsert(collection_name=collection_name, points=points)
# print(f"Successfully chunked and stored {len(points)} vectors in Qdrant!")



# query=input("ask questions:")
# query_vector=embedding_model.encode(query).tolist()


# search_results = qdrant_client.query_points(
#     collection_name=collection_name,
#     query=query_vector,       # 👈 use 'query' instead of 'query_vector'
#     limit=2
# )

# for result in search_results.points:  # 👈 iterate over .points
#     print(f"\n[Score: {result.score:.3f}] Found Match:")
#     print(f"Text: {result.payload['text']}")

# '''
# things to improve in this code 

# 1. prompt- give a m=automated prompt , just staright convert the text to md , but can asdd falbacks for text and diagrams , from gemini vision to ocr model, etc to increase efficiency
# 1-ans-there are special multi-modal models -CLIP which can handle both tex and images , the ocr and gemini vision is not a accurate path
#   The better way is not using OCR and Gemini vision differently. A more optimized way is using a CLIP model, which is both vision transformer and text transformer. I will send the handwritten PDF and separate handwritings because CLIP is not very good for detecting handwriting and converting it into texts and vectors. So, what I did is I will separate the images and give it to CLIP vision transformer, and I will separate the handwriting part and give it to OCR models like Indra OCR, which will convert it into perfect and very accurate typed texts, which will be then converted into vectors, and the images will be converted into vectors by vision transformer. This is the plan for me.
# 2. chunk size
# 3. chunk overlap
# experiment cthe chunck size and chunk overlap , adn find out th seett pint with most efficent way and accurate
# 4. vector dimension
# learn more about it -what is it , learn more about vectors
# 5. distance metric
# 6. qdrant client
# experimnet different types of vertorisations and what will happen in each case
# 7. embedding model

# decorete it more and give the output in more good json format.and usable way
# 8. search results
# 9. search results score
# 10. search results text

# '''




##the above code was a simple rag try - updated code is below 


import os 
import uuid
from dotenv import load_dotenv

from PIL import Image


from preprocessing.loading import pdf_to_images
from preprocessing.splitting import extract_visual_crops

from callingmodels.texttransformer import process_text_path
from callingmodels.visiontransformer import get_clip_image_embedding
from callingmodels.visiontransformer import get_clip_text_embedding
from callingmodels.visiontransformer import process_visual_path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_text_splitters import MarkdownHeaderTextSplitter
#it convert the normal text in to a markdown format , making it more clean 

from google import genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct



# initialling qdrant vectordb and gemini api 
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant_client = QdrantClient(path="./qdrant_data")

text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# vector db collections
TEXT_COLLECTION = "handwritten text notes"
if not qdrant_client.collection_exists(TEXT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=TEXT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

IMG_COLLECTION = "handwritten images notes"
if not qdrant_client.collection_exists(IMG_COLLECTION):
    qdrant_client.create_collection(
        collection_name=IMG_COLLECTION,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )
 #A COMMON DOUBT HERE? WHY IN CASE OF TEXT THE SIZE OF VECTOR IS 384 WHWEARS IN CASE OF IMAGES  IT IS 512???


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


def indexing_pdf(pdf_path:str):
    '''this function will work as a parent fucntion which takes pdf file link as input 
    then brekas it in to images
    pass is throung indra OCR to cahc the hanwritten texts 
    adn pass throung cv2 edge detction to detct images and crop them 
    and the images cropped will passed to the CLip MUTLIMODAL MODEL
    and finall store the respective vectors embeddinsg to qdrant db '''
    
    print(f"converting th pdf:{pdf_path} to images")
    images_frompdf=pdf_to_images(pdf_path)

    text_points=[]
    image_points=[]

    for page_num , img_path in enumerate(images_frompdf, start=1):
        print(f" -> Processing Page {page_num}: {img_path}")
        #now will doOCR on this image to get the texts

        page_md=process_text_path(img_path)#OCRed it 
        #chunking th  markdown texts
        if page_md.strip():
            chunks = text_splitter.split_text(page_md)
            #vectorising them
            for id_chunk , chunk in enumerate(chunks):
                vector=text_embedder.encode(chunk).tolist()
                text_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "page": page_num,
                            "chunk_id": id_chunk,
                            "text": chunk,
                            "source": pdf_path
                        }
                    )
                )
                #leran all this pointer structure path , what is happening here? how and in which format is it storing the embeddings in the text_points

                #cropping images and converting it into embeddings
        image_crops=extract_visual_crops(img_path)
        for image_num,img in enumerate(image_crops):
            #converting the diagram cropped in to embeddings using CLIP
            img_embedding = get_clip_image_embedding(img["image"])
         
            #in case we need to e images cropped to show or judge the accurance of cv2 edge detction , saving it to disk
            os.makedirs("./temp_images", exist_ok=True) 
            crop_save_path = f"./temp_images/p{page_num}_crop_{image_num}.png"
            img["image"].save(crop_save_path)
            


            image_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=img_embedding,
                    payload={
                        "page": page_num,
                        "crop_path": crop_save_path,
                        "bounding_box": img["box"],
                        "source": pdf_path
                    }
                )
            )


        #now will feedd these embeddings to to the different collections in qdract vector db
        print(f"Saving vectors into Qdrant...")
        if text_points:
            qdrant_client.upsert(collection_name=TEXT_COLLECTION, points=text_points)
            print(f" Saved {len(text_points)} text chunks.")
        
        elif image_points:
            qdrant_client.upsert(collection_name=IMG_COLLECTION, points=image_points)
            print(f" Saved {len(image_points)} visual diagram crops.")


#taking user query 
def query_frm_user(query: str = "", query_image_path: str = None):
    print(f"\nProcessing user query: '{query}' | Image: {query_image_path}")
    
    user_query_image = None
    if query_image_path and os.path.exists(query_image_path):
        user_query_image = Image.open(query_image_path)
    
    # -------------------------------------------------------------
    # 1. TEXT SEARCH (in handwritten text notes collection)
    # -------------------------------------------------------------
    context_text = ""
    search_text = query
    
    # If the user ONLY gave an image without text, ask Gemini to summarize its topic
    if not search_text and user_query_image:
        ocr_prompt = "What is the main topic or question shown in this image? Summarize in one short sentence."
        temp_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_query_image, ocr_prompt]
        )
        search_text = temp_resp.text.strip()
        print(f"-> Extracted search topic from query image: '{search_text}'")

    if search_text:
        query_embedding = text_embedder.encode(search_text).tolist()
        text_results = qdrant_client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_embedding,
            limit=3
        )
        for res in text_results.points:
            context_text += f"\n[From Page {res.payload['page']}]:\n{res.payload['text']}\n"

    # -------------------------------------------------------------
    # 2. DIAGRAM SEARCH (in handwritten images collection)
    # -------------------------------------------------------------
    # If user provided an image, embed IT directly with CLIP (Image-to-Image search)
    # Otherwise, embed the text query with CLIP (Text-to-Image search)
    if user_query_image:
        diagram_query_vector = get_clip_image_embedding(user_query_image)
    else:
        diagram_query_vector = get_clip_text_embedding(query)

    diagram_results = qdrant_client.query_points(
        collection_name=IMG_COLLECTION,
        query=diagram_query_vector,
        limit=1
    )
    
    retrieved_diagram_image = None
    diagram_info = ""
    if diagram_results.points:
        top_diagram = diagram_results.points[0]
        crop_path = top_diagram.payload.get("crop_path")
        if crop_path and os.path.exists(crop_path):
            retrieved_diagram_image = Image.open(crop_path)
            diagram_info = f"\nA relevant diagram from Page {top_diagram.payload['page']} was retrieved from your notes."
            print(f"-> Attached diagram from notes: {crop_path}")
        
    prompt = f"""
You are an expert tutor helping a student understand their handwritten study notes.
Answer the user's question clearly, thoroughly, and accurately based on the notes and the attached diagram (if any).

- If the user uploaded a screenshot/image, explain what is shown and connect it to the handwritten notes.
- If a relevant diagram was retrieved from notes, reference it in your explanation.
- Always cite the page numbers where you found the information (e.g. "[Page 2]").
- If the notes do not contain enough information to answer, state that honestly.

---
RELEVANT NOTES CONTEXT:
{context_text}
{diagram_info}
---
USER QUESTION / REQUEST: {query if query else "Explain this image in the context of my notes."}
"""

    print("Generating AI response with Gemini...")
    content_payload = [prompt]
    if user_query_image:
        content_payload.append(user_query_image)
    if retrieved_diagram_image:
        content_payload.append(retrieved_diagram_image)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content_payload
    )
    print("\n================ FINAL AI ANSWER ================")
    print(response.text)
    print("=================================================\n")
        
if __name__ == "__main__":
    pdf_file = "./test2.pdf"

    indexing_pdf(pdf_file)
    while True:
        question = input("\nAsk something about your notes (or 'quit'): ")
        if question.lower() == 'quit':
            break
        img_input = input("Enter path to a screenshot/image (or press Enter to skip): ").strip()
        img_path = img_input if img_input else None
        query_frm_user(query=question, query_image_path=img_path)
