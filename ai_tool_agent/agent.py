import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import TOOLS


load_dotenv()


def create_ai_agent():
    """Create a Gemini-powered LangChain tool-calling agent."""

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise ValueError(
            "Gemini API key missing. Add GOOGLE_API_KEY to your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
    )

    system_prompt = """
You are a helpful AI research assistant powered by Google Gemini.

Available tools:
1. web_search: search the internet using DuckDuckGo.
2. python_calculator: safely calculate mathematical expressions.
3. read_local_file: read TXT, Markdown, and JSON files from data/.

Instructions:
- Choose the appropriate tools for the user's request.
- Use multiple tools when the task requires them.
- Use web search for current information.
- Use the calculator for arithmetic.
- Read local files when their contents are needed.
- Combine observations into a useful final answer.
- Be honest when a tool fails or information is unavailable.
- Never invent tool results.
- Provide concise explanations of your decisions and results.
"""

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=system_prompt,
    )

    return agent


def ask_agent(agent, question: str) -> str:
    """Run Gemini and display tool calls and observations."""

    print("\n" + "=" * 60)
    print("GEMINI AGENT EXECUTION TRACE")
    print("=" * 60)
    print(f"USER: {question}")

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    final_answer = ""

    for message in result["messages"]:
        if message.type == "ai":
            tool_calls = getattr(message, "tool_calls", [])

            for call in tool_calls:
                print("\nTHOUGHT: Gemini selected a tool.")
                print(f"ACTION: {call['name']}")
                print(f"INPUT: {call['args']}")

            if message.content and not tool_calls:
                final_answer = message.content

        elif message.type == "tool":
            print(f"\nOBSERVATION [{message.name}]:")
            print(str(message.content)[:3000])

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)

    return str(final_answer)
