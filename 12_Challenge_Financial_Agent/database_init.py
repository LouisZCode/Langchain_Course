"""
Database initialization and setup
Creates CSV files if they don't exist
"""

import os
import pandas as pd


def initialize_databases(
    trade_log_path: str,
    portfolio_path: str,
    cash_log_path: str,
    stock_evals_path: str
):
    """
    Creates empty database CSV files if they don't exist.
    
    Args:
        trade_log_path: Path to trades log CSV
        portfolio_path: Path to portfolio CSV
        cash_log_path: Path to cash log CSV
        stock_evals_path: Path to stock evaluations CSV
    """
    
    # Create trades log if doesn't exist
    if not os.path.exists(trade_log_path):
        new_dataframe = pd.DataFrame(
            columns=["buy_or_sell", "ticket_symbol", "number_of_stocks", 
                    "individual_price", "total_cost_trade", "date_transaction"]
        )
        new_dataframe.to_csv(trade_log_path, index=False)
        print(f"✓ Created {trade_log_path}")
    
    # Create portfolio if doesn't exist
    if not os.path.exists(portfolio_path):
        new_dataframe = pd.DataFrame(
            columns=["ticket_symbol", "porcentage_weight", "number_of_stocks", 
                    "average_price", "total_cost_stock", "total_PL"]
        )
        new_dataframe.to_csv(portfolio_path, index=False)
        print(f"✓ Created {portfolio_path}")
    
    # Create cash log if doesn't exist
    if not os.path.exists(cash_log_path):
        new_dataframe = pd.DataFrame(
            columns=["add_or_withdraw", "cash_ammount", "date_of_transaction"]
        )
        new_dataframe.to_csv(cash_log_path, index=False)
        print(f"✓ Created {cash_log_path}")
    
    # Create stock evaluations if doesn't exist
    if not os.path.exists(stock_evals_path):
        new_dataframe = pd.DataFrame(
            columns=["stock", "LLM_1", "LLM_2", "LLM_3", "price", 
                    "price_description", "p/e", "one_sentence_reasoning"]
        )
        new_dataframe.to_csv(stock_evals_path, index=False)
        print(f"✓ Created {stock_evals_path}")