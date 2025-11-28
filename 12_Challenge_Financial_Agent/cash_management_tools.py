from langchain_core.tools import tool
import pandas as pd
from datetime import datetime
from config import CASH_LOG
from cash_management_helper_functions import _update_portfolio_info


def _withdraw_cash(cash_ammount : float) -> str:
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    if cash_column_total < cash_ammount:
        return "You dont have enough funds to withdraw that ammount"

    else:
        date_transaction = datetime.now()
        
        new_row = pd.DataFrame([{
        "add_or_withdraw" : "withdraw",
        "cash_ammount" : -cash_ammount, 
        "date_of_transaction" : date_transaction
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CASH_LOG, index=False)

        _update_portfolio_info()

    return f"Added {cash_ammount} usd to the cah position"


def _add_cash(cash_ammount: float) -> str:
    date_transaction = datetime.now()
    
    df = pd.read_csv(CASH_LOG)
    new_row = pd.DataFrame([{
    "add_or_withdraw" : "add",
    "cash_ammount" : cash_ammount, 
    "date_of_transaction" : date_transaction
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CASH_LOG, index=False)

    _update_portfolio_info()

    return f"Added {cash_ammount} usd to the cah position"


@tool(
    "add_cash",
    parse_docstring=True,
    description="adds cash to the portfolio cash position"
)
def add_cash_tool(cash_ammount : float) -> str:
    """
    Description:
        adds cash to the account

    Args:
        cash_ammount (str) : the cash ammount to be added to the account

    Returns:
        Lets the user know that cash has been added

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    success = _add_cash(cash_ammount)

    return success

@tool(
    "withdraw_cash",
    parse_docstring=True,
    description="Checks if the user has enough cash, and if so, removes cash from the available cash of the users account"
)
def withdraw_cash_tool(cash_ammount : float) -> str:
    """
    Description:
        Checks if the user has enough cash, and if so, removes cash from the available cash of the users account

    Args:
        cash_ammount (str) :to be removed from the acocunt

    Returns:
        Lets the user know that cash has been withrawn

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """

    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    if cash_column_total < cash_ammount:
        return "You dont have enough funds to withdraw that ammount"

    else:
        success = _withdraw_cash(cash_ammount)
        return success


@tool(
    "count_cash",
    parse_docstring=True,
    description="tell the total of available cash in cash position"
)
def cash_position_count() -> str:
    #Sum and Rest all the cash logs to give a final cash position
    """
    Description:
        gives you the total ammount of available cash in the users account

    Returns:
        Lets the user know how much cash is available

    Raises:
        Lets the user know if there is a lack of information to create this transaction
    """
    df = pd.read_csv(CASH_LOG)
    cash_column_total = df["cash_ammount"].sum()

    return f"the user has {cash_column_total} available usd"