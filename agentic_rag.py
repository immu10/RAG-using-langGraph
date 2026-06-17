# from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
import os
import hashlib
import chromadb

import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    relevancy: bool
    reply: str = "this is a reply"
    vector_store: list



def indexMaker(embeddings):
    pages =DirectoryLoader(path ="docs", glob= "**/*.pdf", show_progress=True).load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(pages)

    # use_multithreading=True
    # vector_store = Chroma(
    #     embedding_function=embeddings,
    #     persist_directory="chroma_db",
    #     collection_name="collection",
    #     )
  
    


    # vector_store.add_documents(documents=documents, ids=ids)

    client = chromadb.PersistentClient(path="chroma_db")
    # embeddings =EMBEDDINGS.embed_documents()
    collection = client.get_or_create_collection("collection")
    ids = [hashlib.sha512(doc.page_content.encode("utf-8")).hexdigest() for doc in documents]
    collection.upsert(documents=[doc.page_content for doc in documents], ids=ids)

   


# indexMaker(embeddings)

def ChunkRetrieval(state: State):
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    quest = state["messages"][-1].content
    vector_store = Chroma(
        embedding_function=EMBEDDINGS,
        persist_directory="chroma_db",
        collection_name="collection"
        
        )
    chunks = vector_store.similarity_search(quest) 
    state["vector_store"] = chunks if chunks else ["No relevant chunks found."]
    # return {"vector_store": chunks if chunks else ["No relevant chunks found."]}
    return state

def relevancyCheck(state: State):
    llm = ChatOpenAI(model="gpt-4o-mini")
    query = state["messages"][-1].content
    response = llm.predict(f"Is this query relevant to our documents ( list of documents we have data on = {[ "monopoly", "chess"]})? Answer Yes or No: {query}")
    state["relevancy"] = ("Yes" in response)
    # return {"relevancy": "Yes" in response}
    return state

def chatbot(state: State):
    llm = ChatOpenAI(model="gpt-4o-mini")
    chunks = state['vector_store']
    if not chunks:
        print("No relevant chunks found.")
    prompt = f"from the retireved chunks {chunks}, answer the question {state['messages'][-1].content}"
    response = llm.invoke(f"{prompt}")
    state["reply"] = response
    # return {"reply": response}
    # return f" question: {state["messages"]} \n reply: {state['reply']}"
    return state

def wrongAnswer(state: State):
    return {"reply":"Query not relevant to our documents."}


def build_graph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("relevancyCheck", relevancyCheck)
    graphBuilder.add_node("ChunkRetrieval", ChunkRetrieval)
    graphBuilder.add_node("chatbot", chatbot)
    graphBuilder.add_node("wrongAnswer", wrongAnswer)
    graphBuilder.add_edge(START, "relevancyCheck")
    graphBuilder.add_conditional_edges("relevancyCheck",
                                       path= lambda state: state["relevancy"],
                                       path_map={ True:"ChunkRetrieval", False :"wrongAnswer"})
    graphBuilder.add_edge("ChunkRetrieval", "chatbot")
    graphBuilder.add_edge("chatbot", END)
    graphBuilder.add_edge("wrongAnswer", END)
    


    # print("making index")
    # # indexMaker(embeddings)
    # print("index made")
    agentic_rag = graphBuilder.compile()


    

    reply = agentic_rag.invoke({"messages":"how much money do you start within monopoly?"})
    print("\n\n\n\n",reply["reply"].content)

if __name__ == "__main__":
    indexMaker(EMBEDDINGS)

    build_graph()



