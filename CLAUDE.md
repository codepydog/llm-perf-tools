# Claude Code

## Background
Implement a tool for monitoring llm inference metrics. Specifically, we use openai api completion endpoint to send requests to llm sever, and use the metrics mention in https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html to monitor the performance of the llm server.

## Code Implementation Rules
- Always implement for a small piece of functionality. Specifically, the changes should small than 50 lines of code at once.
- MUST do not write any comments, typing hints and docstrings
- Use pydantic to define the data model
- I prefer following SOLID principles and clean code practices
- When designing architecture and implementing code, you must follow the design philosophy of sklearn and huggingface transformers