import streamlit as st
from openai import OpenAI
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub

# Show title and description.
st.title("💬 Mammoth Q&A Chatbot")
st.write(
    "Your AI Documentation Assistant"
)


# Create an OpenAI client.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=st.secrets['OPEN_API_KEY'])
client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets['OPEN_API_KEY'])
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
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



    with st.chat_message("assistant"):
        response = st.write_stream(rag_chain.stream(prompt))

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    
    st.session_state.messages.append({"role": "assistant", "content": response})
