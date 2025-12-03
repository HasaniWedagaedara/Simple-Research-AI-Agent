import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .research-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # API Key input
    api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Enter your Google AI API key",
    )

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        help="Choose which Gemini model to use",
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random",
    )

    # Max iterations
    max_iterations = st.slider(
        "Max Tool Calls",
        min_value=3,
        max_value=15,
        value=10,
        help="Maximum number of tool calls the agent can make",
    )

    st.divider()

    # Tools info
    st.subheader("🛠️ Available Tools")
    st.markdown("""
    - 🌐 **Web Search**: Search current information
    - 📚 **Wikipedia**: Encyclopedia lookup
    - 💾 **Save to File**: Save research results
    """)

    st.divider()

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# Initialize agent
def initialize_agent(api_key, model_name, temperature):
    if not api_key:
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

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

        tools = [search_tool, wiki_tool, save_tool]

        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            return_intermediate_steps=True,
        )

        return agent_executor
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None


# Main content
st.title("🔍 AI Research Assistant")
st.markdown(
    "Ask me anything and I'll research it for you using web search and Wikipedia!"
)

# Check if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to get started.")
    st.info("Get your API key from: https://makersuite.google.com/app/apikey")
else:
    # Initialize agent if not already done
    if st.session_state.agent_executor is None:
        st.session_state.agent_executor = initialize_agent(
            api_key, model_name, temperature
        )

    # Query input
    query = st.text_area(
        "What would you like to research?",
        placeholder="e.g., What is the importance of Sri Lanka in global trade?",
        height=100,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        research_button = st.button("🚀 Start Research", use_container_width=True)
    with col2:
        example_button = st.button("📝 Use Example", use_container_width=True)

    # Example query
    if example_button:
        query = "What are the latest developments in artificial intelligence?"
        st.rerun()

    # Research execution
    if research_button and query:
        if st.session_state.agent_executor is None:
            st.error("Agent initialization failed. Please check your API key.")
        else:
            with st.spinner("🔍 Researching... This may take a moment..."):
                try:
                    # Create progress placeholder
                    progress_placeholder = st.empty()

                    # Run the agent
                    result = st.session_state.agent_executor.invoke({"input": query})

                    # Clear progress
                    progress_placeholder.empty()

                    # Display results
                    st.success("✅ Research Complete!")

                    # Result container
                    with st.container():
                        st.markdown("### 📊 Research Results")
                        st.markdown(
                            f'<div class="research-box">{result.get("output", "No output generated")}</div>',
                            unsafe_allow_html=True,
                        )

                    # Show intermediate steps
                    if result.get("intermediate_steps"):
                        with st.expander("🔧 View Tool Calls"):
                            for i, step in enumerate(result["intermediate_steps"], 1):
                                action, observation = step
                                st.markdown(f"**Step {i}: {action.tool}**")
                                st.code(f"Input: {action.tool_input}")
                                st.text(f"Output: {str(observation)[:200]}...")
                                st.divider()

                    # Save to history
                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": query,
                            "result": result.get("output", ""),
                        }
                    )

                    # Download button
                    st.download_button(
                        label="💾 Download Results",
                        data=result.get("output", ""),
                        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"❌ Error during research: {e}")
                    with st.expander("View Error Details"):
                        st.code(str(e))

# Display history
if st.session_state.history:
    st.divider()
    st.subheader("📜 Research History")

    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"{item['timestamp']} - {item['query'][:50]}..."):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown(f"**Result:**")
            st.markdown(item["result"])

            # Download individual result
            st.download_button(
                label="💾 Download",
                data=item["result"],
                file_name=f"research_{i}.txt",
                mime="text/plain",
                key=f"download_{i}",
            )

# Footer
st.divider()
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <p>Powered by Gemini AI | Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
