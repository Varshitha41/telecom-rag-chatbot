from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

# Load the same embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the existing Chroma database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function
)

print("Connected to ChromaDB!")
print(f"Total documents in database: {db._collection.count()}")


chat_history = []
while True:

    query = input("\nAsk your telecom question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye")
        break

    if not query.strip():
        print("⚠️ Please enter a question.")
        continue

    try:
        results = db.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in results])

        history = "\n".join(chat_history)

        prompt = f"""
You are a telecom expert.

Previous Conversation:
{history}

Context:
{context}

Current Question:
{query}

Answer the current question using the context and previous conversation.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        print("\n" + "=" * 60)
        print("Groq's Answer")
        print("=" * 60)
        print(answer)

        chat_history.append(f"User: {query}")
        chat_history.append(f"Assistant: {answer}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

# print(f"Found {len(results)} results")

# for i, doc in enumerate(results):
#     print("=" * 60)
#     print(f"Result {i+1}")
#     print(doc.page_content[:500])