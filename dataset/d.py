from tqdm import tqdm
from langchain_community.vectorstores import FAISS
import pickle
import pandas as pd
import numpy as np
from langchain.schema import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class CFG:
    # store="프랭크버거"
    output_path = "/home/user09/beaver/data/shared_files/dataset"
    # data_path = ""
    # save_path = ""
    embedding_model= "BAAI/bge-m3" # "BAAI/bge-m3"
    # retriever_k=5
    # retriever_bert_weight=0.7
    version='4_general'
    # info_type='STORE_INFO'   # STORE_INFO, TIME_INFO, MENU_INFO
    store_num = 'd'  # 102496, 102506, 103807, 104570, 104933
    fill_nan = "None"  # 없는 정보의 경우에는 "정보없음"보다는 "None"이 나은 듯
    # basic_info = ["주차장", "씨씨티비", "영업시간", "예약가능여부", "전화번호"]
    seed=42


path = f"/home/user09/beaver/data/shared_files/dataset/{CFG.store_num}.xlsx"


df = pd.read_excel(path)

#### 엑셀파일 DB화(pickle파일로 변환) ##### 
docs = []
for _, row in df.iterrows():
    details = row['contents']
    
    docs.append(Document(page_content=details))#, metadata=metadata))

# Embeddings and vector store setup
encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {'device': 'cpu'}

hf = HuggingFaceEmbeddings(
    model_name=CFG.embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

db = FAISS.from_documents(docs, hf)

db.save_local(f"{CFG.output_path}/{CFG.store_num}_faiss")

with open(f"{CFG.output_path}/{CFG.store_num}_docs.pkl", "wb") as f:
    pickle.dump(docs, f)
    

# db = FAISS.load_local(f"/home/user09/beaver/data/db/{CFG.store_num}_faiss{CFG.version}", hf, allow_dangerous_deserialization=True)

# # Create retrievers
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": CFG.retriever_k})
# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = CFG.retriever_k

# ensemble_retriever = EnsembleRetriever(
#     retrievers=[retriever, bm25_retriever],
#     weights=[CFG.retriever_bert_weight, 1 - CFG.retriever_bert_weight]
# )
