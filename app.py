import streamlit as st
import os
from uuid import uuid4
import requests
from bs4 import BeautifulSoup

from langchain_xai import ChatXAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ───────────────────────────────────────────────
# Page config & title
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="AetherAgent",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌌 AetherAgent")
st.markdown(
    "**Production-grade Agentic AI**  •  Powered by **xAI Grok** + **LangGraph ReAct**  •  "
    "Tools: web search + calculator  •  Persistent memory"
)

# ───────────────────────────────────────────────
# Sidebar – API key & model selection
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Prefer secret → fallback to text input
    xai_key = st.secrets.get("XAI_API_KEY", None)
    if not xai_key:
        xai_key = st.text_input(
            "xAI API Key",
            type="password",
            help="Get it at https://console.x.ai • Add to Secrets for permanent access"
        )
    
    model_name = st.selectbox(
        "Grok Model",
        options=["grok-4", "grok-beta"],
        index=0
    )
    
    st.caption("💡 Pro tip: Use Secrets tab in Streamlit dashboard for auto-loading key")

if not xai_key:
    st.error("Please provide your xAI API key above or in Streamlit Cloud Secrets → Restart app")
    st.stop()

os.environ["XAI_API_KEY"] = xai_key

# ───────────────────────────────────────────────
# Tools
# ───────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Execute simple math / Python expressions safely.
    Input must be a valid expression, e.g. '2 * (3 + 4)' or 'import math; math.sqrt(16)'"""
    try:
        # Very restricted globals/locals
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Basic web search using Google results scraping (no external API needed).
    Returns top snippets for current information."""
    try:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for block in soup.select("div.g")[:4]:
            title_tag = block.select_one("h3")
            snippet_tag = block.select_one("div.VwiC3b, span.st")
            if title_tag and snippet_tag:
                title = title_tag.get_text(strip=True)
                snippet = snippet_tag.get_text(strip=True)[:320]
                results.append(f"**{title}**\n{snippet}…")

        if not results:
            return "No clear results found. Try a more specific query."

        return "\n\n".join(results)

    except Exception as e:
        return f"Web search failed: {str(e)}. Continuing without fresh web data."


tools = [web_search, calculator]

# ───────────────────────────────────────────────
# LLM + Agent
# ───────────────────────────────────────────────
llm = ChatXAI(model=model_name, temperature=0.65)

memory = MemorySaver()
agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    # Optional: you can add interrupt_before=["tools"] for human approval later
)

# ───────────────────────────────────────────────
# Session management
# ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ───────────────────────────────────────────────
# Chat input & agent execution
# ───────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything (news, math, planning, cricket…)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking + using tools…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            try:
                response = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                answer = response["messages"][-1].content
                st.markdown(answer)
            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                answer = f"⚠️ Something went wrong: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.divider()
st.caption(
    "AetherAgent v1 • Built for real-time reasoning • "
    "CSK & Bengaluru vibes 🏏🌆 • Memory preserved across chats"
        )
