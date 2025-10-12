import uuid
import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state.message_history = []
    st.session_state.thread_id = generate_thread_id()
    add_thread(st.session_state.thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def load_conversation(thread_id):
    CONFIG = {'configurable': {'thread_id': thread_id}}
    state = chatbot.get_state(config=CONFIG)
    return state.values.get('messages', [])

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

add_thread(st.session_state.thread_id)

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})
        
        st.session_state.message_history = temp_messages

for message in st.session_state.message_history:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role': 'user', 'content': 'Hi!'}
# {'role': 'assistant', 'content': "Hello! How can I assist you today?"}

user_input = st.chat_input("Talk to me")

if user_input:

    st.session_state.message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state.thread_id}}
    CONFIG = {
        "configurable": {'thread_id': st.session_state.thread_id},
        "metadata": {
            "thread_id": st.session_state.thread_id
        },
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state.message_history.append({'role': 'assistant', 'content': ai_message})

    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    # ai_message = response['messages'][-1].content
    # st.session_state.message_history.append({'role': 'assistant', 'content': ai_message})
    # with st.chat_message("assistant"):
    #     st.text(ai_message)