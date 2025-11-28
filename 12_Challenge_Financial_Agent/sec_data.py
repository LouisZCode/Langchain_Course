"""
The tool used to get clean data from the SEC official website.

You can find:
(
download_clean_fillings
)

"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from edgar import Company
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tqdm


def download_clean_filings(ticker, keep_files=False): # <--- Added flag
    """
    Gets filings, processes them, and saves to VectorStore.
    keep_files=True: Saves HTMLs in 'data/' forever.
    keep_files=False: Deletes HTMLs after processing (Clean).
    """
    
    # 1. Setup
    DATA_FOLDER = "data"
    DB_PATH = "Quarterly_Reports_DB"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    company = Company(ticker)
    filings = company.get_filings(form="10-Q").latest(8)
    
    print(f"Found {len(filings)} filings for {ticker}. Processing...")
    all_chunks = [] 

    # 2. Loop through filings
    for filing in tqdm(filings, desc=f"Processing {ticker} Reports", unit="filing"):
        
        # --- FIX: Define ONE consistent path ---
        # We save cleanly as: data/AAPL_2024-01-01.html
        clean_filename = f"{ticker}_{filing.filing_date}.html"
        full_file_path = os.path.join(DATA_FOLDER, clean_filename)
        
        try:
            # A. Get HTML
            html_content = filing.html()
            if not html_content:
                print(f"Skipping {filing.date}: No HTML.")
                continue

            # B. Write to the DATA folder
            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # C. Load from the DATA folder
            loader = UnstructuredHTMLLoader(full_file_path, mode="elements")
            docs = loader.load()

            # D. Add Metadata
            for doc in docs:
                doc.metadata["ticker"] = ticker
                doc.metadata["date"] = filing.filing_date
                doc.metadata["source"] = full_file_path # Point to real file

            # E. Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            file_chunks = text_splitter.split_documents(docs)
            all_chunks.extend(file_chunks)
            
            # (Note: We do NOT delete here anymore, we let 'finally' handle it)

        except Exception as e:
            print(f"Error processing {clean_filename}: {e}")

        finally:
            # --- LOGIC: Delete only if we don't want to keep them ---
            if not keep_files and os.path.exists(full_file_path):
                os.remove(full_file_path)
                # print(f"Deleted temp file: {clean_filename}")

            
            # 3. Create Vector Store
            if all_chunks:
                if os.path.exists(DB_PATH):
                    print("Appending to existing Vector Store...")
                    vector_store = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
                    vector_store.add_documents(all_chunks)
                else:
                    print("Creating NEW Vector Store...")
                    vector_store = FAISS.from_documents(all_chunks, embedding_model)
                    
                vector_store.save_local(DB_PATH)
                print("Success! Vector store saved.")
            else:
                print("No chunks were generated.")