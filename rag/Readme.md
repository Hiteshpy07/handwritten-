building this project besidfe learning 

Commit 239a390-
 used a simple gewmini vision to detect the text and then converted to veector embeddings 

 what to do for the next commit -
  The better way is not using OCR and Gemini vision differently. A more optimized way is using a CLIP model, which is both vision transformer and text transformer. I will send the handwritten PDF and separate handwritings because CLIP is not very good for detecting handwriting and converting it into texts and vectors. So, what I did is I will separate the images and give it to CLIP vision transformer, and I will separate the handwriting part and give it to OCR models like Indra OCR, which will convert it into perfect and very accurate typed texts, which will be then converted into vectors, and the images will be converted into vectors by vision transformer. This is the plan for me.



commit 25d0dde- 
added the pdf loader- split it into images using PymuPDF
and created the fuction for splitting uimages and diagrams from the pages to snd to CLIP model

---

### 📌 Roadmap / Task Queue (To Do Next)

1. **Multi-Turn Chat History & Conversational Context**:
   - Maintain multi-turn conversational memory across user queries.
   - Summarize or pass preceding Q&A turns to Gemini so it understands follow-up questions (e.g. *"Can you explain the formula from your previous answer in more detail?"*).

2. **Personal Handwriting PDF Synthesis (Full-Loop)**:
   - Convert the AI's synthesized answer back into your personal handwritten style.
   - Render it on lined/notebook paper background using a custom handwriting vector font (`.ttf`) or stroke synthesis.
   - Paste the retrieved diagram crops directly alongside the handwritten answer.
   - Provide a 1-click **"Download Handwritten PDF"** button in the UI.


