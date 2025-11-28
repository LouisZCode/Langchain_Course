import pandas as pd
from config import CASH_LOG, TRADE_LOG, PORTFOLIO
import json
import re
import ast

"""
This is a helper function that will update the information in the portfolio database, based ont the information of the 
trades logs database, and will organize it in a way that make sit understandable at a glamce.
Called everytime there is a movement or change in the portfolio like cash movements, or trades.
"""

def _update_portfolio_info(trade_log_path=TRADE_LOG, portfolio_path=PORTFOLIO, cash_log_path=CASH_LOG):
    """
    Rebuilds the portfolio.csv from scratch based on trade_log.csv,
    including cash position and percentage weights.
    """
    trade_log_df = pd.read_csv(trade_log_path)
    cash_log_df = pd.read_csv(cash_log_path)

    # Calculate total cash (already has +/- signs in the data)
    total_cash = cash_log_df['cash_ammount'].sum()

    # Start with empty portfolio dictionary
    portfolio = {}

    # Process all trades to rebuild portfolio from scratch
    for index, row in trade_log_df.iterrows():
        symbol = row['ticket_symbol']
        quantity = row['number_of_stocks']
        price = row['individual_price']
        buy_or_sell = row['buy_or_sell']

        if buy_or_sell == 'buy':
            if symbol not in portfolio:
                # New stock
                portfolio[symbol] = {
                    'number_of_stocks': quantity,
                    'total_cost_stock': quantity * price,
                    'average_price': price
                }
            else:
                # Add to existing position
                current_qty = portfolio[symbol]['number_of_stocks']
                current_cost = portfolio[symbol]['total_cost_stock']
                
                new_qty = current_qty + quantity
                new_cost = current_cost + (quantity * price)
                new_avg_price = new_cost / new_qty
                
                portfolio[symbol]['number_of_stocks'] = new_qty
                portfolio[symbol]['total_cost_stock'] = new_cost
                portfolio[symbol]['average_price'] = new_avg_price

        elif buy_or_sell == 'sell':
            if symbol in portfolio:
                # Reduce position based on average cost
                current_qty = portfolio[symbol]['number_of_stocks']
                current_avg_price = portfolio[symbol]['average_price']
                
                new_qty = current_qty - quantity
                cost_of_sold = quantity * current_avg_price
                new_cost = portfolio[symbol]['total_cost_stock'] - cost_of_sold
                
                if new_qty <= 0:
                    # Fully sold - remove from portfolio
                    del portfolio[symbol]
                else:
                    # Partial sale - keep average price the same
                    portfolio[symbol]['number_of_stocks'] = new_qty
                    portfolio[symbol]['total_cost_stock'] = new_cost
                    portfolio[symbol]['average_price'] = current_avg_price

    # Convert dictionary to DataFrame
    portfolio_data = []
    for symbol, data in portfolio.items():
        portfolio_data.append({
            'ticket_symbol': symbol,
            'number_of_stocks': data['number_of_stocks'],
            'average_price': data['average_price'],
            'total_cost_stock': data['total_cost_stock'],
            'porcentage_weight': 0.0,  # Calculate below
            'total_PL': 0.0
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)

    # Check if portfolio has any stocks
    if len(portfolio_df) > 0:
        # Has stocks - calculate percentages
        total_portfolio_value = portfolio_df['total_cost_stock'].sum()
        total_account_value = total_portfolio_value + total_cash
        
        if total_account_value > 0:
            portfolio_df['porcentage_weight'] = round((portfolio_df['total_cost_stock'] / total_account_value) * 100, 2)
            cash_percentage = round((total_cash / total_account_value) * 100, 2)
        else:
            cash_percentage = 100.0
    else:
        # No stocks yet - 100% cash
        total_account_value = total_cash
        cash_percentage = 100.0

    # Add CASH row at the top
    cash_row = pd.DataFrame([{
        'ticket_symbol': 'CASH',
        'porcentage_weight': cash_percentage,
        'number_of_stocks': 0,
        'average_price': 0,
        'total_cost_stock': total_cash,
        'total_PL': 0.0
    }])

    # Combine cash row with portfolio
    portfolio_df = pd.concat([cash_row, portfolio_df], ignore_index=True)

    # Save to CSV
    portfolio_df.to_csv(portfolio_path, index=False)

"""
This is a helper function that will clean the answers form the open_ai, Claude and different LLMS into the same format,
so it is better digested by the "councel"
"""

def _extract_structured_data(response_content):
    """
    Extract structured data from LLM response, handling different formats.
    
    Works for:
    - OpenAI: '{"financials":"Strong",...}'
    - Claude: "Returning structured response: {'financials': 'Strong',...}"
    """
    content = response_content.replace("Returning structured response:", "").strip()
    # Try to find JSON object
    json_match = re.search(r'\{[^}]+\}', content)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            # First try standard JSON (double quotes)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try Python literal syntax (single quotes)
            try:
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass # Fallback
    
    # Fallback
    try:
        return json.loads(content)
    except:
        return {} # Return empty dict or handle error gracefully