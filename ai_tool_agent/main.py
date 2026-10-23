from agent import create_ai_agent, ask_agent


def main():
    print("=" * 60)
    print("       LANGCHAIN AI TOOL AGENT")
    print("=" * 60)
    print("Available tools:")
    print("1. Web Search")
    print("2. Python Calculator")
    print("3. Local File Reader")
    print("\nType 'exit' or 'quit' to stop.")

    try:
        agent = create_ai_agent()
        print("\nAI agent initialized successfully.")

    except Exception as exc:
        print(f"\nInitialization error: {exc}")
        return

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Agent stopped. Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            ask_agent(agent, question)

        except KeyboardInterrupt:
            print("\nAgent stopped.")
            break

        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()
