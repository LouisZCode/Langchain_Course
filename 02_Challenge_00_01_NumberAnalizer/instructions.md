Requirements:

Create an agent that analyzes numbers
Create a tool called analyze_number that:

Takes an integer as input
Uses get_stream_writer() to emit progress updates (at least 2 different updates)
Checks if the number is:

Even or odd
Prime or not prime
Positive, negative, or zero


Returns a summary of all findings


Stream the agent's response using stream_mode=["messages", "custom"]

Print the custom tool updates with a [TOOL] prefix
Print the LLM token stream as it arrives (without prefix)


User input: Ask the user to provide a number to analyze
System prompt: Make the agent a "mathematical assistant" that always uses the tool to analyze numbers