import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.llms.openai import OpenAI
import openai
import pandas as pd
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, PodSpec
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.indices.postprocessor import SimilarityPostprocessor
#from llama_index.postprocessor import SentenceTransformerRerank
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from FunctionTools import ScriptureDescriptionToolSpec, MantraToolSpec



#load keys
openai_api_key = st.secrets["OPENAI_APIKEY_CS"]
pinecone_api_key = st.secrets["PINECONE_API_KEY_SAM"]

#llm
llm_AI4 = OpenAI(temperature=0, model="gpt-4-1106-preview",api_key=openai_api_key, max_tokens=512)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4-1106-preview").encode
    )

# global settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    embed_batch_size=8
)
Settings.llm = llm_AI4 
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.callback_manager = CallbackManager([token_counter])
#memory for bot
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

#load vector database
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("pod-index")
vector_store_pine = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context_pine = StorageContext.from_defaults(vector_store=vector_store_pine)
index_store = VectorStoreIndex.from_vector_store(vector_store_pine,storage_context=storage_context_pine)
query_engine_vector = index_store.as_query_engine(similarity_top_k=5,vector_store_query_mode ='hybrid',alpha=0.6)
#pandas Engine
df_veda_details = pd.read_csv("Data/veda_content_details.csv",encoding='utf-8')
query_engine_pandas = PandasQueryEngine(df=df_veda_details)

# Query Engine Tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine_vector,
        metadata=ToolMetadata(
            name="vector_engine",
            description=(
                '''Helpful to get semantic information from the documents. These documents containing comprehensive information about the Vedas.\
                They also covers various aspects, including general details about the Vedas, fundamental terminology associated with Vedic literature, \
                and detailed information about Vedamantras for each Veda. The Vedamantra details encompass essential elements such as padapatha, rishi, chandah,\
                devata, and swarah.This tool is very useful to answer general questions related to vedas.\
                Sample Query:\
                1. What is the meaning of devata ?\
                2. What are the different Brahmanas associated with SamaVeda?\
                3. What is the difference between Shruti and Smriti.
               '''
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=query_engine_pandas,
        metadata=ToolMetadata(
            name="pandas_engine",
            description=(
                '''Helpful to answer the queries related to count from the documents. This document is a .csv file with different columns containing comprehensive information about the Vedas.\
                The column names as follows:\
                'mantra_id', 'scripture_name', 'KandahNumber', 'PrapatakNumber','AnuvakNumber', 'MantraNumber', 'DevataName', 'RishiName', 'SwarahName', 'ChandaName',\
                'padapatha', 'vedamantra', 'AdhyayaNumber', 'ArchikahNumber', 'ArchikahName', 'ShuktaNumber', 'keyShukta', 'ParyayaNumber', 'MandalaNumber'
                ''This tool is very useful to answer questions related to vedas on.\
                Sample Query:\
                1. How many mantras are there in RigVeda whose swarah is gāndhāraḥ?\
                2. How many different devata present in rigveda?\
                3. Which Kandah has the maximum number of in KrishnaYajurVeda?
                4. How many mantras are there in RigVeda?
               '''
            ),
        ),
    )
    ]

# tools
mantra_tools = MantraToolSpec().to_tool_list()
description_tools = ScriptureDescriptionToolSpec().to_tool_list()
tools = [*mantra_tools,*description_tools,*query_engine_tools]

# context
context = """
  You are an expert on Vedas and related scriptures.\
  Your role is to respond to questions about vedic scriptures and associated information based on available sources.\
  For every query, you must use either any one of the tool or use available history/context.
  Please provide well-informed answers. Don't use prior knowledge.
"""

# Function to create ReActAgent instance (change it based on your initialization logic)
@st.cache_resource(show_spinner=False)  # Set allow_output_mutation to True for mutable objects like instances
def create_react_agent():
    return ReActAgent.from_tools(tools, llm=llm_AI4, context=context, verbose=True)

# Example usage
react_agent_instance = create_react_agent()

# Streamlit Components Initialization
st.title("Svarupa Bot ")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. I am Svarupa AI Assistant. Ask me a question about Vedas!"}
    ]

if "chat_engine" not in st.session_state.keys():
    # Using st.cache_resource for caching the unserializable react_agent
    st.session_state.chat_engine = create_react_agent()

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Using the cached chat_engine
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
