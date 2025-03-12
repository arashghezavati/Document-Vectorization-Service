"""
Agent workflow management using LangGraph.
This module handles dynamic workflow creation based on user requests.
"""

import os
import json
import time
import chromadb
import google.generativeai as genai
from typing import Dict, List, Any, Optional, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Configure the Gemini API
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# Define the state schema
class AgentState(TypedDict):
    messages: List[Any]
    documents: List[Dict]
    user_id: str
    task: str
    credentials: Dict

# Document tools
@tool
def search_documents(query: str, collection_name: str) -> List[Dict]:
    """
    Search for documents in the user's collection based on a query.
    
    Args:
        query: The search query
        collection_name: The name of the collection to search in
        
    Returns:
        A list of matching documents with their content and metadata
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient("./chroma_db")
    
    # Get the collection
    try:
        collection = client.get_collection(collection_name)
    except ValueError:
        return []
    
    # Search for documents
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    # Format the results
    documents = []
    for i, doc_id in enumerate(results["ids"][0]):
        documents.append({
            "id": doc_id,
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {}
        })
    
    return documents

@tool
def get_document_by_id(document_id: str, collection_name: str) -> Dict:
    """
    Get a specific document by its ID.
    
    Args:
        document_id: The ID of the document to retrieve
        collection_name: The name of the collection
        
    Returns:
        The document content and metadata
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient("./chroma_db")
    
    # Get the collection
    try:
        collection = client.get_collection(collection_name)
    except ValueError:
        return {"error": "Collection not found"}
    
    # Get the document
    results = collection.get(
        ids=[document_id]
    )
    
    if not results["ids"]:
        return {"error": "Document not found"}
    
    return {
        "id": results["ids"][0],
        "content": results["documents"][0],
        "metadata": results["metadatas"][0] if results["metadatas"] else {}
    }

# Task planning
def plan_task(state: AgentState) -> AgentState:
    """
    Plan the execution of a task based on the user request.
    """
    # Extract the task from the state
    task = state["task"]
    user_id = state["user_id"]
    
    # Create a system message for the planner
    system_prompt = f"""You are a task planner for a document processing system.
    The user has requested: "{task}"
    
    Analyze this request and determine:
    1. What documents are needed
    2. What actions need to be performed
    3. What external services might be required
    
    Respond with a clear plan of action.
    """
    
    # Get a response from Gemini
    try:
        response = gemini_model.generate_content(system_prompt)
        plan = response.text
    except Exception as e:
        plan = f"Error planning task: {str(e)}"
    
    # Add the response to the messages
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": plan})
    
    # Update the state
    state["messages"] = messages
    
    return state

# Document retrieval
def retrieve_documents(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents based on the task.
    """
    # Get the most recent AI message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    plan = messages[-1]["content"] if messages else ""
    
    # Extract document needs from the plan
    query_prompt = f"""Based on the plan: "{plan}", 
    generate a search query to find relevant documents.
    Respond with ONLY the search query, nothing else.
    """
    
    # Get a search query from Gemini
    try:
        query_response = gemini_model.generate_content(query_prompt)
        query = query_response.text.strip()
    except Exception as e:
        query = "document search"
    
    # Search for documents - using direct function call instead of tool
    try:
        # Initialize ChromaDB client using the same path as in chatbot_api.py
        DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # Get the collection for the current user only
        try:
            user_id = state.get("user_id")
            if not user_id:
                state["documents"] = []
                state["messages"].append({
                    "role": "assistant", 
                    "content": "I couldn't identify your user account. Please try logging in again."
                })
                return state
                
            # Get the proper collection name using the same format as in chatbot_api.py
            collection_name = f"user_{user_id}_docs"
            print(f"Retrieving documents for user: {user_id} (Collection: {collection_name})")
            
            try:
                # Get or create the collection without specifying embedding function
                # This matches the implementation in chatbot_api.py
                collection = client.get_or_create_collection(name=collection_name)
                
                # Build query filter - no filter to get all documents
                where_filter = {}
                
                # Get all documents (matching chatbot_api.py implementation)
                docs = collection.get(include=["documents", "metadatas"])
                
                # Format the results
                documents = []
                if docs and "documents" in docs and docs["documents"]:
                    for i, doc_content in enumerate(docs["documents"]):
                        if i < len(docs["metadatas"]):
                            documents.append({
                                "id": docs["ids"][i] if "ids" in docs and i < len(docs["ids"]) else f"doc_{i}",
                                "content": doc_content,
                                "metadata": docs["metadatas"][i]
                            })
                
                # Update the state with the documents
                state["documents"] = documents
                
                # Add a message about the documents found
                if documents:
                    doc_message = {
                        "role": "assistant", 
                        "content": f"I found {len(documents)} documents in your collection that might help with your task."
                    }
                else:
                    doc_message = {
                        "role": "assistant", 
                        "content": "I couldn't find any documents in your collection. Please upload some documents first."
                    }
                state["messages"].append(doc_message)
                
            except ValueError:
                # Collection doesn't exist for this user
                state["documents"] = []
                state["messages"].append({
                    "role": "assistant", 
                    "content": "I couldn't find your document collection. Please upload some documents first."
                })
                return state
                
        except Exception as e:
            print(f"Error with user collection: {str(e)}")
            state["documents"] = []
            state["messages"].append({
                "role": "assistant", 
                "content": f"I encountered an error while accessing your document collection: {str(e)}"
            })
            return state
            
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        state["documents"] = []
        state["messages"].append({
            "role": "assistant", 
            "content": f"I encountered an error while searching for your documents: {str(e)}"
        })
    
    return state

# Task execution
def execute_task(state: AgentState) -> AgentState:
    """
    Execute the planned task using the retrieved documents.
    """
    # Get the plan and documents
    messages = state.get("messages", [])
    plan = messages[0]["content"] if messages else ""
    documents = state.get("documents", [])
    
    # Create a context with document content
    doc_context = ""
    if documents:
        doc_context = "\n\n".join([
            f"Document {i+1}: {doc['content'][:500]}..." 
            for i, doc in enumerate(documents)
        ])
    else:
        doc_context = "No relevant documents found."
    
    # Create a prompt for execution
    execution_prompt = f"""You are a document processing assistant.
    
    The user's task is: "{state['task']}"
    
    You have the following documents:
    {doc_context}
    
    Based on these documents and the user's request, complete the task.
    If you need additional information or credentials that are not available,
    politely ask the user for what you need.
    """
    
    # Get a response from Gemini
    try:
        execution_response = gemini_model.generate_content(execution_prompt)
        response_text = execution_response.text
    except Exception as e:
        response_text = f"Error executing task: {str(e)}"
    
    # Add the response to the messages
    state["messages"].append({"role": "assistant", "content": response_text})
    
    return state

# Check if credentials are needed
def check_credentials(state: AgentState) -> str:
    """
    Check if the task requires credentials that are not available.
    Returns 'needs_credentials' or 'execute' based on the check.
    """
    # Get the most recent AI message
    messages = state.get("messages", [])
    if not messages:
        return "execute"
    
    last_message = messages[-1]["content"].lower()
    
    # Check if the message asks for credentials
    credential_keywords = [
        "api key", "login", "credentials", "account", "password", 
        "authenticate", "token", "access", "connect to"
    ]
    
    if any(keyword in last_message for keyword in credential_keywords):
        return "needs_credentials"
    
    return "execute"

# Request credentials
def request_credentials(state: AgentState) -> AgentState:
    """
    Request necessary credentials from the user.
    """
    # Get the most recent AI message
    messages = state.get("messages", [])
    credential_request = messages[-1]["content"] if messages else ""
    
    # Create a more structured credential request
    credential_prompt = f"""Based on this credential request: "{credential_request}",
    create a clear, structured request for the user.
    Explain exactly what credentials are needed and why they are necessary for the task.
    """
    
    # Get a structured request from Gemini
    try:
        structured_request_response = gemini_model.generate_content(credential_prompt)
        structured_request = structured_request_response.text
    except Exception as e:
        structured_request = f"Error creating credential request: {str(e)}"
    
    # Add the structured request to the messages
    state["messages"].append({"role": "assistant", "content": structured_request})
    
    return state

# Create the agent workflow
def create_agent_workflow():
    """
    Create the agent workflow for processing tasks.
    """
    from langgraph.graph import StateGraph
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("plan", plan_task)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("execute", execute_task)
    
    # Add the edges
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "execute")
    
    # Set the entry point
    workflow.set_entry_point("plan")
    
    # Compile the workflow
    return workflow.compile()

# Create the workflow graph
def create_workflow_graph() -> StateGraph:
    """
    Create the workflow graph for processing user tasks.
    """
    # Create a new graph
    graph = StateGraph(AgentState)
    
    # Add nodes to the graph
    graph.add_node("plan", plan_task)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("execute", execute_task)
    graph.add_node("request_credentials", request_credentials)
    
    # Add edges to the graph
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "execute")
    graph.add_conditional_edges(
        "execute",
        check_credentials,
        {
            "needs_credentials": "request_credentials",
            "execute": END
        }
    )
    graph.add_edge("request_credentials", END)
    
    # Set the entry point
    graph.set_entry_point("plan")
    
    return graph

# Initialize the workflow graph
workflow_graph = create_workflow_graph().compile()

async def process_task(task: str, user_id: str, credentials: Dict = None) -> List[Dict]:
    """
    Process a user task using the workflow graph.
    
    Args:
        task: The task to be performed
        user_id: The ID of the user
        credentials: Optional credentials for external services
        
    Returns:
        The messages generated during task processing
    """
    # Initialize the state
    state = {
        "messages": [],
        "documents": [],
        "user_id": user_id,
        "task": task,
        "credentials": credentials or {}
    }
    
    # Run the workflow
    result = workflow_graph.invoke(state)
    
    # Extract and format the messages
    messages = []
    for message in result["messages"]:
        if isinstance(message, (HumanMessage, AIMessage)):
            messages.append({
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content
            })
    
    return messages
