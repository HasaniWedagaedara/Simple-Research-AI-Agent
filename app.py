import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from dotenv import load_dotenv
import os
from datetime import datetime
from components.footer import footer
from components.header import header
from styles.styles import load_css

load_dotenv()


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_css()


if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None


@st.cache_resource
def initialize_agent(
    api_key, model_name="gemini-2.5-flash", temperature=0.7, max_iterations=10
):
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

When you have gathered sufficient information, provide a comprehensive answer with proper formatting.""",
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


header()


api_key = os.getenv("GOOGLE_API_KEY", "")

if not api_key:
    st.warning("⚠️ Google API Key not found in environment variables.")
    api_key = st.text_input(
        "Please enter your Google API Key:",
        type="password",
        help="Get your API key from: https://makersuite.google.com/app/apikey",
    )
    if not api_key:
        st.info("💡 Get your API key from: https://makersuite.google.com/app/apikey")
        st.stop()



if st.session_state.agent_executor is None:
    with st.spinner("Initializing AI Agent..."):
        st.session_state.agent_executor = initialize_agent(
            api_key, model_name="gemini-2.5-flash", temperature=0.7, max_iterations=10
        )


query = st.text_area(
    "What would you like to research?",
    placeholder="e.g., What is the importance of Sri Lanka in global trade?",
    height=100,
    key="query_input",
)


col = st.columns([1])[0]
with col:
    research_button = st.button(
        "Start Research", use_container_width=True, type="primary"
    )


if research_button and query:
    if st.session_state.agent_executor is None:
        st.error("❌ Agent initialization failed. Please check your API key.")
    else:
        with st.spinner("🔍 Researching... This may take a moment..."):
            try:
                result = st.session_state.agent_executor.invoke({"input": query})

                output_text = result.get("output", "No output generated")

                st.success("✅ Research Complete!")

                st.markdown("### 📊 Research Results")

                st.markdown(
                    f'<div class="research-box">{output_text}</div>',
                    unsafe_allow_html=True,
                )

                if result.get("intermediate_steps"):
                    with st.expander("🔧 View Tool Calls & Process"):
                        for i, step in enumerate(result["intermediate_steps"], 1):
                            action, observation = step
                            st.markdown(f"**Step {i}: {action.tool}**")
                            st.code(f"Input: {action.tool_input}", language="text")
                            st.text(f"Output: {str(observation)[:300]}...")
                            st.divider()

                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "result": output_text,
                    }
                )
                
                if st.session_state.get("clear_query", False):
                    st.session_state["query_input"] = ""
                    st.session_state["clear_query"] = False

               
                st.download_button(
                    label="💾 Download Results as TXT",
                    data=output_text,
                    file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as e:
                if "429" in str(e):
                    st.warning(
                        "⚠️ Rate limit exceeded. Please wait a minute and try again."
                    )
                else:
                    st.error(f"❌ Error during research: {e}")
                    with st.expander("📋 View Error Details"):
                        import traceback

                        st.code(traceback.format_exc())
                    

if st.session_state.history:
    st.divider()


    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📜 Research History")
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("✅ History cleared!")
            st.rerun()

    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.container():
            st.markdown('<div class="history-item">', unsafe_allow_html=True)
            st.markdown(f"**🕐 {item['timestamp']}**")
            st.markdown(f"**❓ Query:** {item['query']}")

            with st.expander("View Full Result"):
                st.markdown(item["result"])

                st.download_button(
                    label="💾 Download This Result",
                    data=item["result"],
                    file_name=f"research_{item['timestamp'].replace(':', '-').replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"download_history_{i}",
                )
            st.markdown("</div>", unsafe_allow_html=True)


footer()