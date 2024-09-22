
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader

loader_1 = TextLoader(file_path=r"Data\doc1.txt",autodetect_encoding=True)
loader_2 = TextLoader(file_path=r"Data\doc2.txt",autodetect_encoding=True)

docs = loader_1.load()
docs += loader_2.load()




text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    )
splits = text_splitter.split_documents(docs)
model_path = 'Alibaba-NLP/gte-large-en-v1.5'
#retriever = vectorstore.as_retriever()

array = [text.page_content for text in  splits]

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=HuggingFaceEmbeddings(model_name=model_path,show_progress=True,model_kwargs={"trust_remote_code":True}),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

vector_store.add_documents(splits)