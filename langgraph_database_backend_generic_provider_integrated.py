import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_generic import ChatGeneric
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGeneric(model="Meta-Llama-3.1-8B-Instruct")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def unique_threads(thread_lst):
    result = []
    existing_threads = set()
    for item in thread_lst:
        if item.config['configurable']['thread_id'] not in existing_threads:
            result.append(item.config['configurable']['thread_id'])
            existing_threads.add(item.config['configurable']['thread_id'])

    return result

def retrieve_all_threads() -> list[str]:
    all_threads = unique_threads(checkpointer.list(None))
    return all_threads[::-1]  # Reverse the list to show the most recent threads first

# def retrieve_all_threads() -> list[str]: # Example function to retrieve all distinct thread IDs from the database using SQLite connection
#     cursor = conn.cursor()
#     cursor.execute("SELECT DISTINCT thread_id FROM states")
#     rows = cursor.fetchall()
#     return [row[0] for row in rows]