import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from groq import Groq
from config import GROQ_API_KEY

#PAGE CONFIGURATION

st.set_page_config(
    page_title = "Telecom AI Assistant",
    page_icon = "📡",
    layout = "wide"
)

# Groq Client

client = Groq(api_key=GROQ_API_KEY)

#Embedding Model

embedding_function = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#Connect to ChromaDB

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function
)

#Session State

if"messages" not in st.session_state:
    st.session_state.messages = []

#Sidebar

with st.sidebar:

    st.title("🔩 Settings")
    st.markdown("---")

    show_context = st.checkbox(
        "Show Retrived Context",
        value=False
    )

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

#Main Page

st.title("👾Telecom AI Assistant")
st.caption("Ask questions about GPON, OLT, ONU, FTTH and telecom networking concepts.")


#Display Previous Messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Chat Input

query = st.chat_input(
    "Ask your telecom question..."
)

#User Asked Something

if query:

    #Show user message
    with st.chat_message("user"):
        st.markdown(query)

    #Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Search Similar Documents
    results = db.similarity_search(query, k=3)

    # Build Context
    context = "\n\n".join(
        [doc.page_content for doc in results]   
    )

    # Optional Context Display
    if show_context:
        st.subheader("📜 View Retrieved Context")
        st.write(context)

       
# Build Prompt
    prompt = f"""
Retrieved Context:
{context}

User Question:
{query}
"""

# Prepare Conversation
    messages = [
        {
            "role": "system",
            "content": """
You are a friendly Telecom AI Assistant.

Your job is to answer telecom-related questions using ONLY the retrieved context.

Rules:

1. If the answer exists in the retrieved context, answer clearly.

2. If the answer is NOT found in the retrieved context, reply exactly:
"I couldn't find that information in the provided documents."

3. If the user greets you (Hi, Hello, Hey), greet them politely.

4. If the user thanks you, reply politely.

5. If the user says Goodbye, Bye, Exit or See you, respond politely.

6. Never make up information.

7. Keep your answers short, simple and friendly.
"""
        }
    ]


# Add Previous Conversation
    messages.extend(
        st.session_state.messages
    )

# Add Latest RAG Prompt
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

# Call Groq
   # Call Groq

    try:
        with st.spinner("🔎 Searching documents..."):
            response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        answer = response.choices[0].message.content

    except Exception:
        answer = "⚠️ Something went wrong. Please try again."


# Show Assistant Message
    with st.chat_message("assistant"):
        st.markdown(answer)

# Save Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )