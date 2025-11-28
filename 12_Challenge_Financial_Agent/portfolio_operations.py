"""
Here we have all the functions and data related ot the Portfolio and its operations,
Like Read, update, and show it.

Here we find:
- read_my_portfolio
- add_to_portfolio
- remove_from_portfolio

"""

from config import TRADE_LOG, CASH_LOG
import pandas as pd
from langchain.tools import tool
from datetime import datetime
from cash_management_tools import _add_cash, _withdraw_cash

@tool(
    "read_my_portfolio",
    parse_docstring=True,
    description="reads the current potfolio information of the user"
)
def read_my_portfolio():
    """
    Description:
        Reads the current information saved in the users portfolio

    Args:
        None

    Returns:
        The dataframe information inside the portfolio

    Raises:
        Lets the user know if there is no information inside the portfolio
    """
    dataframe = pd.read_csv(TRADE_LOG)
    trading_log = dataframe.to_markdown(index=False)
    return trading_log


@tool(
    "add_to_portfolio",
    parse_docstring=True,
    description="Automatically checks if there is enough cash, and if so, uses it to buy the stock."
)
def add_to_portfolio(ticket_symbol : str, number_of_stocks : float, individual_price_bought : float) -> str:
    """
    Description:
        Automatically checks if there is enough cash, and if so, uses it to buy the stock.

    Args:
        ticket_symbol (str) : the initials of the stock bought, number_of_stocks (float) : the quantity of stocks bought in this transaction, individual_price_bought (float): the proce of each individual stock, total_cost_trade (float): the result of the multiplication of number of stocks bough by the cost per individual stock, date_bought: todays date.

    Returns:
        Lets the user know a new buy has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()
    total_cost_trade = number_of_stocks * individual_price_bought

    if cash_column_total < total_cost_trade:
        return "You dont have enough funds to buy this ammount of this stock"

    else:

        date_bought = datetime.now()
        
        df = pd.read_csv(TRADE_LOG)
        new_row = pd.DataFrame([{
            "buy_or_sell" : "buy",
            "ticket_symbol" : ticket_symbol,
            "number_of_stocks" : number_of_stocks,
            "individual_price" : +individual_price_bought,
            "total_cost_trade" : -total_cost_trade,
            "date_transaction" : date_bought
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(TRADE_LOG, index=False)

        _withdraw_cash(total_cost_trade)
        
        return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_bought': {individual_price_bought}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_bought}. Cash already taken form the available cash"
    

@tool(
    "remove_from_portfolio",
    parse_docstring=True,
    description="Sells stock and automatically adds that cash to the total cash balance."
)
def remove_from_portfolio(ticket_symbol : str, number_of_stocks : float, individual_price_sold : float) -> str:
    """
    Description:
        Sells stock and automatically adds that cash to the total cash balance.

    Args:
        ticket_symbol (str) : the initials of the stock sold, number_of_stocks (float) : the quantity of stocks sold in this transaction, individual_price_sold (float): the price of each individual stock, total_return_trade (float): the result of the multiplication of number of stocks bough by the cost per individual stock, date_sold: todays date.

    Returns:
        Lets the user know a new sell has been done and gives the details of that new transaction

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    date_sold = datetime.now()

    total_cost_trade = number_of_stocks * individual_price_sold
    
    df = pd.read_csv(TRADE_LOG)
    new_row = pd.DataFrame([{
        "buy_or_sell" : "sell",
        "ticket_symbol" : ticket_symbol,
        "number_of_stocks" : number_of_stocks,
        "individual_price" : -individual_price_sold,
        "total_cost_trade" : +total_cost_trade,
        "date_transaction" : date_sold
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRADE_LOG, index=False)

    _add_cash(total_cost_trade)
    print(f"Added {total_cost_trade} cash to the cash position")
    
    return f"New trade saved with these details:\n 'ticket_symbol': {ticket_symbol}\n'number_of_stocks': {number_of_stocks}\n'individual_price_sold': {individual_price_sold}\n'total_cost_trade': {total_cost_trade}\n'date_bought': {date_sold}. The cash already was added to the cash balance"