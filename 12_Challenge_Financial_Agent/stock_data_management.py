


def _stock_market_data(ticker_symbol: str) -> str:
    ticket_symbol = ticker_symbol.upper()
    #Sadly, API calls are only 25 per day, so will be using mocking data for this exercise:
    lower_price = random.randint(10 , 200)
    higher_price = random.randint(201 , 500)

    pe_ratio = random.randint(10 , 40)

    return f"the ticket symbol {ticket_symbol} has a lowest price of {lower_price}, and highest of {higher_price}, with a pe ratio of {pe_ratio} times per sales"

@tool(
    "stock_market_data",
    parse_docstring=True,
    description="gives you the stock market prices necessary to answer, alongside the p/e ratio of the company"
)
def stock_market_data_tool(ticker_symbol : str) -> str:
    """
    Description:
        Gets you the lowest and highest price of a stock in the last 2 years and the pe ratio

    Args:
        ticker_symbol (str): The ticker symbol to research

    Returns:
        ticker symbols highest and lowest price in the last 2 years, plus the pe ratio

    Raises:
        If there is not wnough information about the symbol and or an error in the API Call
    """

    return _stock_market_data(ticker_symbol)

def _save_stock_evals(ticket_symbol : str, LLM_1 : str, LLM_2 : str, LLM_3 : str, price : float, price_description : str,  p_e : str, selected_reason : str) -> str:
    """
    Description:
        Saves the stock evals in a csv file

    Args:
        ticket_symbol (str): The ticket symbol to research
        recommendations_list (list): The list of recommendations
        price (float): The price of the stock
        p_e (float): The p/e ratio of the stock
        selected_reason (str): The reason for the recommendation

    Returns:
        Lets the user know a new sell has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    

    df = pd.read_csv(STOCK_EVALS)
    new_row = pd.DataFrame({
        "stock": [ticket_symbol],
        "LLM_1": [LLM_1],
        "LLM_2" :[LLM_2],
        "LLM_3" : [LLM_3],
        "price": [price],
        "price_description": [price_description],
        "p/e": [p_e],
        "one_sentence_reasoning": [selected_reason]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(STOCK_EVALS, index=False)

    return "Succcessfully saved the stock recommendation into the stock evaluations database"


def ticker_admin_tool(ticker_symbol):
    """
    A function that checks if the ticker symbol requested is in the database already or not.
    If it is, returns True, if it is not, returns False.
    """
    #Check the first column in db
    df = pd.read_csv(STOCK_EVALS)
    ticker_column  = df["stock"].values

    if ticker_symbol in ticker_column:
        print(f"The ticker symbol {ticker_symbol} its already in the db")
        return True
    else:
        print(f"The ticker symbol {ticker_symbol} it not in the DB.\nGathering info from the SEC now...")
        return False

def ticker_info_db(ticker_symbol):
    df = pd.read_csv(STOCK_EVALS)
    if df[df['stock'] == ticker_symbol].empty:
        return f"We do not have information about {ticker_symbol} in the Database. Ask user to go to the Councel of LLMs"
    else:
        ticker_row  = df[df["stock"] == ticker_symbol].to_markdown(index=False)
        
        return f"Here the info about {ticker_symbol}:\n{ticker_row}"


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


#Tool for Expert Financial Consultant

@tool(
    "review_stock_data",
    parse_docstring= True,
    description="gives back information about the sotck the user is asking about. If no info, lets you know next steps"
)
def review_stock_data(ticker_symbol : str) -> str:
    """
    Description:
        Gives back information about the sotck the user is asking about. If no info, lets you know next steps
    
    Args:
        ticker_symbol (str): The stock or ticker symbol of the company that will be reviewed agains users risk tolerance and portfolio.
    
    Returns:
        The ticker symbol information we have in the database. 

    Raises:
        If not info in the database, lets you know next steps.

    """
    return ticker_info_db(ticker_symbol)