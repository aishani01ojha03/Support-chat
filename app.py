import streamlit as st
from bot import SupportChatbot

st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬", layout="centered")

st.title("Customer Support Chatbot")
st.caption("Semantic FAQ bot using sentence embeddings (fast, no training).")

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence threshold", 0.30, 0.85, 0.55, 0.01)
show_debug = st.sidebar.checkbox("Show match + confidence", value=True)

@st.cache_resource
def load_bot(th: float):
    return SupportChatbot(threshold=th)

bot = load_bot(threshold)

if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_debug and msg.get("debug"):
            st.caption(msg["debug"])

# Input box
user_text = st.chat_input("Ask a support question (e.g., refund, shipping, tracking)...")

if user_text:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": user_text})

    # Bot reply
    result = bot.reply(user_text)

    debug_text = None
    if show_debug:
        debug_text = f"Matched: {result['matched_question']} | Tag: {result['tag']} | Confidence: {result['confidence']:.3f}"

    st.session_state.chat.append({"role": "assistant", "content": result["answer"], "debug": debug_text})

    # Rerun to show new messages
    st.rerun()