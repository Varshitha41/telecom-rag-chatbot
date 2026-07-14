# print("THIS IS MY FILE")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


loader1 = PyPDFLoader("data/gpon_technology.pdf")
loader2 = PyPDFLoader("data/gpon_specification.pdf")

docs1 = loader1.load()
docs2 = loader2.load()

documents = docs1 + docs2

print(f"Total pages loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

print("Before model")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embedding_function = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

print("After model")

print("Generating embeddings...")

embeddings = embedding_model.encode(
    [chunk.page_content for chunk in chunks]
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

print("Storing embeddings in ChromaDB...")

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_function,
    persist_directory = "chroma_db"
)

print("Embeddings stored successfully!")