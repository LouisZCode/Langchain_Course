"""
Configuration file for Investment Portfolio Assistant
Contains all constants, file paths, and environment setup
"""

import os
from dotenv import load_dotenv
from edgar import set_identity
import yaml


# Load environment variables
load_dotenv()


# File paths for databases
TRADE_LOG = "trades_log.csv"
PORTFOLIO = "my_portfolio.csv"
CASH_LOG = "my_cash.csv"
STOCK_EVALS = "stock_evaluations.csv"
DB_PATH = "Quarterly_Reports_DB"

# SEC Edgar identity
set_identity("Juan Perez juan.perezzgz@hotmail.com")

# Load prompts from YAML
def load_prompts():
    """Load all prompts from prompts.yaml file"""
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts


# API Keys (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"