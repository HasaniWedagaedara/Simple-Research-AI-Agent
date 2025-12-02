from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
)
from langchain.tools import Tool
from datetime import datetime
import os


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """Save research data to a text file with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n{'=' * 50}\n\n"

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)

        return f"✓ Data successfully saved to {filename}"
    except Exception as e:
        return f"✗ Error saving to file: {str(e)}"


# save tool
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file. Input should be the text content to save.",
)


# search tool with DuckDuckGoSearchAPIWrapper
def safe_search(query: str) -> str:
    """Wrapper for DuckDuckGo search with error handling"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Search error: {str(e)}. Try rephrasing your query."


search_tool = Tool(
    name="web_search",
    func=safe_search,
    description="Search the web for current information. Input should be a search query string. Use this for recent events, statistics, and up-to-date information.",
)

# Wikipedia tool with better configuration
api_wrapper = WikipediaAPIWrapper(
    top_k_results=2, doc_content_chars_max=1000, load_all_available_meta=False
)
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="Search Wikipedia for encyclopedic information. Input should be a search query string. Use this for historical facts, scientific concepts, and general knowledge.",
)


all_tools = [search_tool, wiki_tool, save_tool]
