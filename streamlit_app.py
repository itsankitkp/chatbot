import streamlit as st
from openai import OpenAI
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker, RerankRequest

# Show title and description.
st.title("💬 Mammoth Q&A Chatbot")
st.write("Your AI Documentation Assistant")


# Create an OpenAI client.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=st.secrets["OPEN_API_KEY"]
)
client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def decompose_query(query):
    # Simple rule-based decomposition using "and" as separator
    sub_queries = [q.strip() for q in query.split("and")]
    return sub_queries


def enhanced_retrieval(query, retriever):
    # Increase the number of retrieved documents
    docs = retriever.invoke(query)
    return docs


def create_multi_task_prompt(query, docs):
    prompt = f"""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If the query contains multiple tasks, make sure to address each one separately.
    Question: {query}
    Documents:
    {docs}
    Use three sentences maximum for each task and keep the answer concise.
    Do not answer the question if it is not related to the documents provided.
    """
    return prompt


prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPEN_API_KEY"])

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sub_queries = decompose_query(prompt)
    if len(sub_queries) > 1:
        # Get relevant docs for each sub-query
        all_docs = []
        for sub_query in sub_queries:
            docs = enhanced_retrieval(sub_query, retriever)
            all_docs.extend(docs)
        all_docs = [_.page_content for _ in all_docs]
        unique_docs = list(set(all_docs))
        llm_prompt = create_multi_task_prompt(prompt, unique_docs)
        iter = llm.stream(llm_prompt)
    else:
        iter = rag_chain.stream(prompt)

    # Get the response from the chatbot.
    with st.chat_message("assistant"):
        response = st.write_stream(iter)

    # Stream the response to the chat using `st.write_stream`, then store it in
    # session state.

    st.session_state.messages.append({"role": "assistant", "content": response})
