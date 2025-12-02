from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from pydantic import BaseModel, Field
from tools import search_tool, wiki_tool, save_tool
from dotenv import load_dotenv
import os

load_dotenv()

# Define the output structure
class ResearchResponse(BaseModel):
    topic: str = Field(description="The main research topic")
    summary: str = Field(description="A comprehensive summary of the research findings")
    sources: list[str] = Field(description="List of sources used in the research")
    tools_used: list[str] = Field(description="List of tools utilized during research")


# Initialize the LLM with explicit API key
# Using gemini-2.5-flash

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# simpler prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are a helpful research assistant. Use the available tools to gather information and answer questions thoroughly.
    
Available tools:
- web_search: Search the web for current information
- wikipedia: Search Wikipedia for encyclopedic information
- save_text_to_file: Save your research to a file

Question: {input}

When you have gathered sufficient information, provide a comprehensive answer.""",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# tools list
tools = [search_tool, wiki_tool, save_tool]

# agent
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

# agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=True,
)

if __name__ == "__main__":
    query = input("What can I help you research today? ")

    try:
        print("\nStarting research...\n")

        # Run the agent
        result = agent_executor.invoke({"input": query})

        print("\n" + "=" * 70)
        print("RESEARCH RESULTS")
        print("=" * 70)
        print(result.get("output", "No output generated"))
        print("=" * 70)

        # Optionally save the output
        save_output = (
            input("\nWould you like to save this research? (y/n): ").strip().lower()
        )
        if save_output == "y":
            from tools import save_to_txt

            filename = input("Enter filename (or press Enter for default): ").strip()
            if not filename:
                filename = "research_output.txt"
            save_to_txt(result.get("output", ""), filename)
            print(f"✓ Saved to {filename}")

    except Exception as e:
        print(f"\nError during research: {e}")
        import traceback

        traceback.print_exc()
