from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

dir_loader = DirectoryLoader(
    path="data/",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
    )

documents = dir_loader.load()
#print(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents)

""" for i, chunk in enumerate(chunks[:5]):  # First 5
    print(f"\n--- Chunk {i+1} ---")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Preview: {chunk.page_content[:200]}...")
    print(f"Length: {len(chunk.page_content)} chars")
 """

vector_store = FAISS.from_documents(chunks, embedding)
vector_store.save_local("Quaterly_Reports")

