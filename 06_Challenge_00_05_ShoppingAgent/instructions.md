Build an interactive shopping assistant with a Gradio interface that manages a shopping list, calculates budgets, and remembers conversations.

Requirements
1. Data Storage (Pandas + CSV)

Create CSV file with columns: item, quantity, category, added_date
Build functions to: load, save, add, remove, view, and calculate from DataFrame
Handle file creation if CSV doesn't exist

2. Custom Tools (Minimum 3)
Create using @tool decorator with parse_docstring=True:
Tool 1: Add to Shopping List

Parameters: item (str), quantity (int), category (Literal with 3-4 options)
Google-style docstring
Returns confirmation message

Tool 2: View Shopping List

Parameter: category (str, optional)
Returns formatted list
Filter by category if provided

Tool 3: Calculate Budget

Parameters: item (str), price_per_unit (float), quantity (int)
Validate quantity > 0
Return total cost

3. MCP Integration

Connect to at least 1 MCP server
Load MCP tools
Combine with custom tools

4. Agent

Model: Your choice
System prompt: Shopping assistant persona
Tools: Combined custom + MCP tools
Checkpointer: InMemorySaver()

5. Gradio Interface
Required Components:

gr.Chatbot() - conversation display
gr.Textbox() - user input
gr.State() - stores thread_id
"Clear Conversation" button - resets thread_id
"View Shopping List" button - displays current list
Title and description

Functionality:

Chat function handles messages and maintains thread_id
Single user session (one thread_id)
Async agent invocation
Clean layout using gr.Blocks()


Deliverables

Complete working code
Screenshot of Gradio interface
Example conversation output
Any questions encountered


Success Criteria

✅ Gradio interface launches
✅ Can add items through chat
✅ Can view shopping list
✅ Can calculate budgets
✅ Memory persists in conversation
✅ Data saves to CSV
✅ MCP tools integrated