import os
from pathlib import Path

import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    st.error(
        f"""
        GROQ_API_KEY was not found.

        Expected .env file:
        {ENV_FILE}

        Your .env file should contain:

        GROQ_API_KEY=gsk_your_new_key_here
        """
    )

    st.stop()


try:

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

except Exception as e:

    st.error(
        f"Could not initialize Groq client: {e}"
    )

    st.stop()

GROQ_MODEL = "openai/gpt-oss-120b"

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "metadata" not in st.session_state:
    st.session_state.metadata = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def extract_text_from_pdf(uploaded_file):

    pages = []

    try:

        reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                # Normalize whitespace
                text = " ".join(
                    text.split()
                )

                if text.strip():

                    pages.append(
                        {
                            "text": text,
                            "source": uploaded_file.name,
                            "page": page_number
                        }
                    )

    except Exception as e:

        st.error(
            f"Error reading {uploaded_file.name}: {e}"
        )

    return pages


def create_chunks(
    pages,
    chunk_size=700,
    overlap=100
):

    chunks = []
    metadata = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(chunk)

                metadata.append(
                    {
                        "source": page["source"],
                        "page": page["page"]
                    }
                )

            # Move forward
            start += chunk_size - overlap

    return chunks, metadata


def create_embeddings(chunks):

    if not chunks:

        return np.array(
            [],
            dtype="float32"
        )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # Normalize embeddings
    faiss.normalize_L2(
        embeddings
    )

    return embeddings


def create_faiss_index(embeddings):

    if len(embeddings) == 0:

        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def process_pdfs(uploaded_files):

    all_pages = []

    for uploaded_file in uploaded_files:

        pages = extract_text_from_pdf(
            uploaded_file
        )

        all_pages.extend(
            pages
        )

    if not all_pages:

        return False, "No readable text was found in the PDF files."

 
    chunks, metadata = create_chunks(
        all_pages,
        chunk_size=700,
        overlap=100
    )

    if not chunks:

        return False, "No text chunks could be created."

    embeddings = create_embeddings(
        chunks
    )

    if len(embeddings) == 0:

        return False, "Could not create embeddings."


    index = create_faiss_index(
        embeddings
    )

    if index is None:

        return False, "Could not create FAISS index."

 
    st.session_state.chunks = chunks

    st.session_state.metadata = metadata

    st.session_state.faiss_index = index

    st.session_state.processed = True

    # Clear old chat history
    st.session_state.chat_history = []

    return True, f"{len(chunks)} chunks created successfully."


def search_documents(
    question,
    top_k=5
):

    index = st.session_state.faiss_index

    chunks = st.session_state.chunks

    metadata = st.session_state.metadata

    if index is None:

        return []


    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype(
        "float32"
    )


    faiss.normalize_L2(
        question_embedding
    )

    # Don't request more results than available chunks
    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "source": metadata[idx]["source"],
                "page": metadata[idx]["page"],
                "score": float(score)
            }
        )

    return results


def generate_answer(
    question,
    retrieved_results
):

    if not retrieved_results:

        return (
            "I could not find this information "
            "in the uploaded document."
        )


    context_parts = []

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
File: {result["source"]}
Page: {result["page"]}

Content:
{result["chunk"]}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    
    prompt = f"""
You are an AI PDF question-answering assistant.

Your job is to answer the user's question ONLY using
the information provided in the document context below.

IMPORTANT RULES:

1. Use ONLY the supplied document context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer cannot be found in the context,
   say exactly:

   "I could not find this information in the uploaded document."

5. Give a clear and concise answer.
6. When possible, mention the source file and page number.
7. If multiple document sections are relevant, combine them
   carefully.
8. Do not claim something is present in the document unless
   the supplied context supports it.

------------------------------------------------------------
DOCUMENT CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""


    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded document "
                        "question-answering assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=1000
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:

        return (
            f"Groq API error:\n\n{str(e)}"
        )



with st.sidebar:

    st.header("📚 Document Upload")

    st.write(
        "Upload one or more PDF files "
        "to create your RAG knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True
    )

    st.divider()

    st.subheader("🔄 RAG Pipeline")

    st.write("📄 PDF Documents")

    st.write("↓")

    st.write("📝 Text Extraction")

    st.write("↓")

    st.write("✂️ Text Chunking")

    st.write("↓")

    st.write("🧠 Embeddings")

    st.write("↓")

    st.write("🔎 FAISS Vector Database")

    st.write("↓")

    st.write("💬 User Question")

    st.write("↓")

    st.write("🔍 Similarity Search")

    st.write("↓")

    st.write("🤖 Groq LLM")

    st.write("↓")

    st.write("✅ Grounded Answer")


if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF file."
        )

    else:

        with st.spinner(
            "Processing documents..."
        ):

            success, message = process_pdfs(
                uploaded_files
            )

        if success:

            st.success(
                message
            )

        else:

            st.error(
                message
            )


st.title(
    "📚 AI PDF RAG Assistant"
)

st.markdown(
    """
Ask questions about your uploaded PDF documents.

The system extracts the PDF text, divides it into chunks,
creates embeddings, stores them in FAISS, retrieves the
most relevant chunks, and uses a Groq LLM to generate a
grounded answer.
"""
)



if st.session_state.processed:

    st.success(
        "✅ Documents processed successfully."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Text Chunks",
            len(
                st.session_state.chunks
            )
        )

    with col2:

        st.metric(
            "Embedding Dimension",
            384
        )

    with col3:

        st.metric(
            "Vector Database",
            "FAISS"
        )

else:

    st.info(
        "👈 Upload PDF files from the sidebar and "
        "click **Process Documents**."
    )


st.divider()


st.subheader(
    "💬 Ask a Question"
)

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is inheritance in Python?"
)


ask_button = st.button(
    "🤖 Ask AI",
    type="primary",
    use_container_width=True
)



if ask_button:

    if not st.session_state.processed:

        st.warning(
            "Please upload and process a PDF first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        
        with st.spinner(
            "Searching the documents..."
        ):

            retrieved_results = search_documents(
                question,
                top_k=5
            )

        if not retrieved_results:

            st.warning(
                "No relevant document content was found."
            )

        else:

            
            with st.spinner(
                "Generating grounded answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_results
                )

            
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": retrieved_results
                }
            )



if st.session_state.chat_history:

    st.divider()

    st.subheader(
        "📝 Conversation"
    )

    for i, chat in enumerate(
        st.session_state.chat_history
    ):

        
        st.markdown(
            f"### ❓ Question {i + 1}"
        )

        st.write(
            chat["question"]
        )

        st.markdown(
            "### 🤖 Answer"
        )

        st.write(
            chat["answer"]
        )

        # ----------------------------------------------------
        # Retrieved sources
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Retrieved Sources"
        ):

            for j, source in enumerate(
                chat["sources"],
                start=1
            ):

                st.markdown(
                    f"""
                    **Source {j}**

                    **File:** {source["source"]}

                    **Page:** {source["page"]}

                    **Similarity Score:** {source["score"]:.4f}
                    """
                )

                st.text(
                    source["chunk"]
                )

                st.divider()

st.divider()

st.caption(
    "AI PDF RAG Assistant | "
    "PDF → Chunking → Embeddings → FAISS → "
    "Semantic Search → Groq LLM"
)
