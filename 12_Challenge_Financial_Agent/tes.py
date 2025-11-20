import random

def get_stock_market_data(ticket_symbol : str) -> str:
    """
    Description:
        Gets you the lowest and highest price of a stock in the last 2 years and the pe ratio

    Args:
        ticket_symbol (str): The ticket symbol to research

    Returns:
        ticket symbols highest and lowest price in the lasz 2 years, plus the pe ratio

    Raises:
        If there is not wnough information about the symbol and or an error in the API Call
    """

    ticket_symbol = ticket_symbol.upper()
    #Sadly, API calls are only 25 per day, so will be using mocking data for this exercise:
    lower_price = random.randint(10 , 200)
    higher_price = random.randint(201 , 500)

    pe_ratio = random.randint(10 , 40)

    return f"the ticket symbol {ticket_symbol} has a lowest price of {lower_price}, and highest of {higher_price}, with a pe ratio of {pe_ratio} times per sales"

print(get_stock_market_data("PLTR"))