from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import langchain

langchain.debug = True


model_path = 'Alibaba-NLP/gte-large-en-v1.5'


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=HuggingFaceEmbeddings(model_name=model_path,model_kwargs={"trust_remote_code":True}),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "Make the answer descriptive and informative."
    "Do not mention context in the answer"
    "Include all the points in the answer make it informative"
    "Mention all the clause and rules"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retrieved_doc = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 2})

rag_chain = create_retrieval_chain(retrieved_doc, question_answer_chain)

response = rag_chain.invoke({"input": "What are the conditions for a person to obtain separate registration for multiple places of business within a State or Union territory under Rule 11?"})
response["answer"]
print(response["answer"])
