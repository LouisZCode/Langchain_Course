from dotenv import load_dotenv
from typing import Literal
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool(
    "calculator",
    parse_docstring=True,
    description=(
        "Perform basic arithmetic operations on two real numbers"
        "Use this whenever you have operations on any numbers, even if they are integers"
    ),
)
def real_number_calculator(
    a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"]
) -> float:
    """Perform basic arithmetic operations on two real numbers.

    Args:
        a (float): The first number.
        b (float) The second number.
        operation (Literal["add", "substract", "multiply", "divide"]): The arithmetic operation to perform: - "add": returns the sum of 'a' and 'b'. - "subtract": returns the result of 'a - b' - "multiply": returns the product of 'a' and 'b' - "divide": returns the result of 'a / b'. Raises an error is b is zero.

    Returns:
        float: the numerical result of the specified operation
    
    Raises:
        ValueError: if an invalid operation is provided or division by zero is attempted
    """
    print("Invoking calculator tool")
    #Perform the specific operation
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    else:
        return ValueError(f"invalid operation: {operation}")


agent = create_agent(
    model="openai:gpt-4",
    tools=[real_number_calculator],
    system_prompt="you are a math student who always uses its calculator to check the results of given numbers"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 15 + 27?"}]
})

print(response)