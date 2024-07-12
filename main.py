import os
# from langchain.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
# from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


loader = ObsidianLoader("/Users/ibarahime/Documents/obsidian-notes/Resources")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
splitdocs = text_splitter.split_documents(docs)
# print(splitdocs[0].metadata)
# embeddings = OpenAIEmbeddings()
# print(splitdocs)

persist_directory = './db/chroma'
model = "llama3"
llm = ChatOllama(model=model)
# model_name = "BAAI/bge-base-en-v1.5"
# model_kwargs = {
#     'device': 'mps',
#     'trust_remote_code':True
# }
# encode_kwargs = {'normalize_embeddings': True}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction = "search_query:",
#     embed_instruction = "search_document:"
# )
embeddings = FastEmbedEmbeddings(threads=6)

if os.path.isdir(persist_directory) == False:
    print("Embedding documents...")
    vectordb = Chroma.from_documents(documents=splitdocs, embedding=embeddings, persist_directory=persist_directory) # https://github.com/langchain-ai/langchain/issues/24039
    print("Done embedding documents!")
else:
    print(f"Using embeddings from {persist_directory}")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "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" # replace dots with your API key

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Filename",
        type="string",
    ),
    AttributeInfo(
        name="path",
        description="Source of where the file originates from",
        type="string",
    ),
    AttributeInfo(
        name="created",
        description="UNIX timestamp of when the file was first created",
        type="float",
    ),
    AttributeInfo(
        name="last_accessed",
        description="UNIX timestamp of when the user last opened the file",
        type="float",
    ),
]

document_content_description = "Notes on a certain topic written in Markdown"
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore=vectordb,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "ONLY USE THE GIVEN CONTEXT. DO NOT USE YOUR PRETRAINED DATA."
    "Use five sentences maximum. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(vectordb.as_retriever(), question_answer_chain)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# print(memory)
# Create the conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    get_chat_history=lambda h : h
)

while True:
    x = chain.invoke({"question": input("Ask your question: ")})
    # print(x)
    print(x["answer"])