import streamlit as st
from langchain_xai import ChatXAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import os
from uuid import uuid4
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="AetherAgent", page_icon="🌌", layout="wide")
st.title("🌌 AetherAgent")
st.markdown("**Production-Scale Agentic AI** • Powered by **xAI Grok** + LangGraph ReAct")

with st.sidebar:
    st.header("⚙️ Configuration")
    xai_key = st.text_input("xAI API Key (console.x.ai)", type="password", value=st.secrets.get("XAI_API_KEY", ""))
    model = st.selectbox("Grok Model", ["grok-4", "grok-beta"], index=0)

if not xai_key:
    st.error("Add your xAI API key in Streamlit Cloud Dashboard → your app → Settings → Secrets as XAI_API_KEY → Restart")
    st.stop()

os.environ["XAI_API_KEY"] = xai_key

# Calculator tool
@tool
def calculator(expression: str) -> str:
    """Useful for math/calculations. Input: valid Python expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {str(e)}"

# Simple web search tool (no DuckDuckGo dependency)
@tool
def web_search(query: str) -> str:
    """Search the web for current info/news. Input: search query string."""
    try:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.find_all('div', class_='g')[:4]:  # top 4 snippets
            title = g.find('h3')
            snippet = g.find('div', class_='VwiC3b') or g.find('span', class_='st')
            if title and snippet:
                results.append(f"**{title.text}**\n{snippet.text[:300]}...\n")
        return "\n".join(results) if results else "No useful results found. Try rephrasing."
    except Exception as e:
        return f"Search failed: {str(e)}. Agent will reason without web info."

tools = [web_search, calculator]

# LLM + Agent setup
llm = ChatXAI(model=model, temperature=0.7)
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything — e.g. latest CSK news, Bengaluru weather, calculate something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking & using tools..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response = agent_executor.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
            final_answer = response["messages"][-1].content
            st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

st.caption("✅ Agentic AI with memory & tools | Built for Bengaluru/CSK fan vibes 🏏🌆 | No DuckDuckGo issues!")        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Calculation error: {str(e)}"

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, calculator]

# LLM + Agent (with memory)
llm = ChatXAI(model=model, temperature=0.6)
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("What should the agent do? (e.g. Research latest AI news and calculate ROI on investing in it)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking & using tools..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=prompt)]}, 
                config=config
            )
            final_answer = response["messages"][-1].content
            st.markdown(final_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

# Quick examples
st.subheader("Try these")
cols = st.columns(3)
with cols[0]:
    if st.button("📈 Research & calculate Tesla stock impact"):
        st.session_state.messages.append({"role": "user", "content": "Research latest Tesla news and calculate potential 10% portfolio impact"})
        st.rerun()
with cols[1]:
    if st.button("🌍 Plan a 3-day trip to Bengaluru"):
        st.session_state.messages.append({"role": "user", "content": "Plan a perfect 3-day budget trip to Bengaluru including weather, food & attractions"})
        st.rerun()
with cols[2]:
    if st.button("🔬 Explain quantum computing simply"):
        st.session_state.messages.append({"role": "user", "content": "Explain quantum computing to a 12-year-old and calculate 2^10"})
        st.rerun()

st.caption("✅ Fully agentic (plans → tools → observes → answers) | Memory enabled | Ready for production")
