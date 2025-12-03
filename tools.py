from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import os


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """Save research data to a text file with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n{'=' * 50}\n\n"

        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)

        return f"✓ Data successfully saved to {filename}"
    except Exception as e:
        return f"✗ Error saving to file: {str(e)}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file. Input should be the text content to save.",
)


def safe_search(query: str) -> str:
    """Wrapper for DuckDuckGo search with error handling"""
    try:
        from duckduckgo_search import DDGS

        # Use DDGS 
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

            if not results:
                return f"No search results found for: {query}"

           
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                body = result.get("body", "No description")
                link = result.get("href", "No link")
                formatted_results.append(f"{i}. {title}\n{body}\nSource: {link}\n")

            return "\n".join(formatted_results)
    except Exception as e:
        # If DuckDuckGo fails, try Wikipedia as backup
        try:
            wiki_wrapper = WikipediaAPIWrapper(
                top_k_results=2, doc_content_chars_max=1000
            )
            wiki_tool_backup = WikipediaQueryRun(api_wrapper=wiki_wrapper)
            return f"Web search failed, using Wikipedia instead:\n\n{wiki_tool_backup.run(query)}"
        except:  # noqa: E722
            return f"Search error: {str(e)}. Please try rephrasing your query or check your internet connection."


search_tool = Tool(
    name="web_search",
    func=safe_search,
    description="Search the web for current information. Input should be a search query string. Use this for recent events, statistics, cities, countries, and up-to-date information.",
)

# Create Wikipedia tool
api_wrapper = WikipediaAPIWrapper(
    top_k_results=3, doc_content_chars_max=2000, load_all_available_meta=False
)
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="Search Wikipedia for encyclopedic information. Input should be a search query string. Use this for historical facts, scientific concepts, geography, and general knowledge about cities, countries, and places.",
)


all_tools = [search_tool, wiki_tool, save_tool]
