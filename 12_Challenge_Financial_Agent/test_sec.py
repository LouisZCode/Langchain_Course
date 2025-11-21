from sec_edgar_downloader import Downloader

from edgar import set_identity, Company


"""
# Initialize downloader to save files in a "sec_filings" folder
dl = Downloader("YourCompanyName", "your_email@example.com", "./sec_filings")

# Download the last 8 10-Q (Quarterly) reports for Apple (AAPL)
q_apple = dl.get("10-Q", "AAPL", limit=1)

# Download the last 2 10-K (Annual) reports for Apple
#q_apple = dl.get("10-K", "AAPL", limit=2)

print(q_apple)
"""

set_identity("Juan Perez juan.perezzgz@hotmail.com")

def download_clean_filings(ticker):
    # 2. Initialize Company
    company = Company(ticker)
    
    # 3. Get only the 10-Q filings (you can also add 10-K)
    # The 'latest=8' grabs the last 8 available reports
    filings = company.get_filings(form="10-Q").latest(2)
    
    print(f"Found {len(filings)} filings for {ticker}. Downloading clean HTML...")

    for filing in filings:
        # This matches the filename structure: "AAPL-10Q-2024-Q3.html"
        # filing.filing_date is YYYY-MM-DD
        filename = f"{ticker}-{filing.form}-{filing.filing_date}.html"
        
        # 4. The Magic: .html() fetches ONLY the primary document content
        clean_html_content = filing.html()
        
        # 5. Save it directly
        if clean_html_content:
            with open(f'data/{filename}', "w", encoding="utf-8") as f:
                f.write(clean_html_content)
            print(f"Saved: {filename}")
        else:
            print(f"Skipped (No HTML found): {filing.accession_number}")

# Example Usage
download_clean_filings("AAPL")