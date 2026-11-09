import os
import streamlit as st
from PIL import Image

# Import backend functions from main
from main import (
    indexing_pdf,
    query_frm_user,
    text_embedder,
    qdrant_client,
    TEXT_COLLECTION,
    IMG_COLLECTION,
    get_clip_image_embedding,
    get_clip_text_embedding,
    client
)

st.set_page_config(
    page_title="Handwritten Notes Multimodal RAG",
    page_icon="📝",
    layout="wide"
)

# Custom header
st.title("📝 Handwritten Notes Multimodal RAG")
st.caption("Ask questions about your handwritten study notes, diagrams, and sketches using OCR + CLIP + Gemini!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# -------------------------------------------------------------
# SIDEBAR: PDF Upload & Indexing
# -------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload Notes")
    uploaded_pdf = st.file_uploader("Upload Handwritten PDF", type=["pdf"])

    if uploaded_pdf is not None and not st.session_state.indexed:
        if st.button("🚀 Index Document", use_container_width=True):
            save_path = os.path.join("./", uploaded_pdf.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.read())

            with st.spinner("Extracting pages, transcribing handwriting, and detecting diagrams..."):
                indexing_pdf(save_path)
                st.session_state.indexed = True
            st.success("Indexing complete! You can now ask questions.")

    if st.session_state.indexed:
        st.info("✅ Current notebook is indexed and ready in Qdrant.")
        if st.button("🔄 Reset / Index Another PDF", use_container_width=True):
            st.session_state.indexed = False
            st.session_state.messages = []
            st.rerun()

# -------------------------------------------------------------
# MAIN CHAT DISPLAY
# -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "diagram_path" in msg and msg["diagram_path"]:
            st.image(msg["diagram_path"], caption="Retrieved Diagram from Notes", width=350)
        if "user_image" in msg and msg["user_image"]:
            st.image(msg["user_image"], caption="Attached Screenshot", width=300)

# -------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -------------------------------------------------------------
# Optional screenshot upload in an expander
with st.expander("📸 Attach screenshot / sketch to your question (Optional)"):
    query_image_file = st.file_uploader("Upload Image to search with", type=["png", "jpg", "jpeg"], key="query_img")

user_prompt = st.chat_input("Ask a question about your notes...")

if user_prompt:
    # Save uploaded query image if provided
    saved_query_img_path = None
    if query_image_file is not None:
        os.makedirs("./temp_images", exist_ok=True)
        saved_query_img_path = os.path.join("./temp_images", f"query_{query_image_file.name}")
        with open(saved_query_img_path, "wb") as f:
            f.write(query_image_file.read())

    # Record User Message in chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
        "user_image": saved_query_img_path
    })
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if saved_query_img_path:
            st.image(saved_query_img_path, caption="Attached Query Image", width=300)

    # ---------------------------------------------------------
    # ASSISTANT RESPONSE
    # ---------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Searching handwritten notes & diagrams..."):
            
            user_query_image = Image.open(saved_query_img_path) if saved_query_img_path else None
            search_text = user_prompt

            # 1. Text Search in Qdrant
            if not search_text and user_query_image:
                ocr_prompt = "What is the main topic or question shown in this image? Summarize in one short sentence."
                temp_resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[user_query_image, ocr_prompt]
                )
                search_text = temp_resp.text.strip()

            context_text = ""
            if search_text:
                query_embedding = text_embedder.encode(search_text).tolist()
                text_results = qdrant_client.query_points(
                    collection_name=TEXT_COLLECTION,
                    query=query_embedding,
                    limit=3
                )
                for res in text_results.points:
                    context_text += f"\n[From Page {res.payload['page']}]:\n{res.payload['text']}\n"

            # 2. Diagram Search in Qdrant
            if user_query_image:
                diagram_query_vector = get_clip_image_embedding(user_query_image)
            else:
                diagram_query_vector = get_clip_text_embedding(user_prompt)

            diagram_results = qdrant_client.query_points(
                collection_name=IMG_COLLECTION,
                query=diagram_query_vector,
                limit=1
            )

            retrieved_diagram_image = None
            retrieved_diagram_path = None
            diagram_info = ""
            if diagram_results.points:
                top_diagram = diagram_results.points[0]
                crop_path = top_diagram.payload.get("crop_path")
                if crop_path and os.path.exists(crop_path):
                    retrieved_diagram_image = Image.open(crop_path)
                    retrieved_diagram_path = crop_path
                    diagram_info = f"\nA relevant diagram from Page {top_diagram.payload['page']} was retrieved from your notes."

            # 3. Gemini Generation
            prompt = f"""
You are an expert tutor helping a student understand their handwritten study notes.
Answer the user's question clearly, thoroughly, and accurately based on the notes and the attached diagram (if any).

- If the user uploaded a screenshot/image, explain what is shown and connect it to the handwritten notes.
- If a relevant diagram was retrieved from notes, reference it in your explanation.
- Always cite the page numbers where you found the information (e.g. "[Page 2]").
- Format mathematical formulas in LaTeX notation using $ or $$.

---
RELEVANT NOTES CONTEXT:
{context_text}
{diagram_info}
---
USER QUESTION / REQUEST: {user_prompt}
"""
            content_payload = [prompt]
            if user_query_image:
                content_payload.append(user_query_image)
            if retrieved_diagram_image:
                content_payload.append(retrieved_diagram_image)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=content_payload
            )
            ai_answer = response.text

            # Render text answer
            st.markdown(ai_answer)
            # Render retrieved diagram if found
            if retrieved_diagram_path:
                st.image(retrieved_diagram_path, caption="Relevant Diagram from your Notes", width=400)

            # Save in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_answer,
                "diagram_path": retrieved_diagram_path
            })

