import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
from flask import Flask, request, Response, stream_with_context, jsonify
import time
import pandas as pd
import json
import pickle
import ast
import logging
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from threading import Thread
from transformers import BitsAndBytesConfig
import torch

# Initialize Flask app
app = Flask(__name__)

def load_logger(log_dir, log_level):
    logger = logging.getLogger(__name__)
    if log_level == 'INFO':
        lv = logging.INFO
    elif log_level == 'ERROR':
        lv = logging.ERROR
    elif log_level == 'DEBUG':
        lv = logging.DEBUG
    else:
        raise NotImplementedError
    logger.setLevel(lv)

    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir, encoding='utf-8-sig')
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

# Set logger
curtime = time.strftime("%Hh-%Mm-%Ss")
date = time.strftime("%Y-%m-%d")
log_folder = os.path.join('/home/user09/beaver/prompt_engineering/our_log', date)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)

logdir = os.path.join(log_folder, curtime + '.log')
logger = load_logger(logdir, 'INFO')
logger.info(f'*** {curtime} START ***')
logger.info(f'*** PID: {os.getpid()} ***')




class args:
    store = '102506'   # 102496(히스피커피), 102506(삼청당), 103807(노모어피자), 104570(롤링파스타), 104933(홍콩반점)
    output_dir = "/home/user09/beaver/log/"
    llama_v1_model_name_or_path = "MLP-KTLim/llama-3-Korean-Bllossom-8B" #"Bllossom/llama-3.1-Korean-Bllossom-Vision-8B"
    instruction_template = "/home/user09/beaver/data/shared_files/prompt/incontext_v8.txt"
    embedding_model = "BAAI/bge-m3"
    retriever_k = 10
    retriever_bert_weight = 0.7
    cache_dir = "/nas/.cache/huggingface"
    model_revision = "main"
    config_name = None
    rag_threshold = 0.32

def read_db(output_dir, db_type, name, hf=None):
    if db_type == "faiss":
        return FAISS.load_local(f"{output_dir.split('log')[0]+'data'}/db/{name}_faiss1_general", hf, allow_dangerous_deserialization=True)
    elif db_type == "docs":
        with open(f"{output_dir.split('log')[0]+'data'}/db/{name}_docs1_general.pkl", "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("db_type should be either faiss or docs")

# https://wikidocs.net/234094 ==> 매개변수 확인
def get_retriever(db, docs):  # 이건 유사도 임계값 넘는 문서들을 반환 후, 앙상블에서 결국 k개를 반환.
    db_reteriver = db.as_retriever(    # 리트리버 정의
        search_type="similarity_score_threshold",  # 기존: similarity
        search_kwargs={"score_threshold": args.rag_threshold},  # 최소 유사도 임계값
    )
    docs_reteriver = BM25Retriever.from_documents(docs)
    print("docs_reteriver:", docs_reteriver)
    docs_reteriver.k = args.retriever_k
    retriever = EnsembleRetriever(
        retrievers=[db_reteriver, docs_reteriver],
        weights=[args.retriever_bert_weight, 1 - args.retriever_bert_weight],
    )
    return retriever, db_reteriver, docs_reteriver


def format_docs(docs):
    return "\n".join(f"- {doc}".replace('"', '') for doc in docs[:args.retriever_k])

def invoke_format(example):
    #text1 = "" if example['before_input'] == "이전 대화 없음" else example['before_input']
    #text2 = "" if example['before_response'] == "이전 대화 없음" else example['before_response']
    text3 = example['current_input']
    text = text3
    return text

def remove_special_characters(string):
    # 정규 표현식을 사용하여 특정 특수기호들을 제거
    string = re.sub(r'[(){}\[\],]', '', string)
    return string

encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {'device': 'cpu'}
hf = HuggingFaceBgeEmbeddings(
    model_name=args.embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


ensemble, db, docs = get_retriever(read_db(args.output_dir, "faiss", f"{args.store}", hf), read_db(args.output_dir, "docs", f"{args.store}"))
retriever_dict = {
    f"{args.store}": ensemble,
    "db_retriever": db,
    "docs_retriever": docs
}

# menu_df = pd.read_excel("/home/user09/beaver/data/shared_files/dataset/홍콩반점_all_menus_v6.xlsx")

# 각 매장별로 모든 메뉴를 딕셔너리로 저장.
# all_menus_dict = {
#     row["메뉴명"]: {"종류": row["종류"], "가격": row["가격"]}
#     for _, row in menu_df.iterrows()
# }

# print(all_menus_dict)
def initialize_model():
    global prompt_chain, llm_chain, streamer, model  # 함수 밖에서도 변수의 유효범위를 유지하기 위해, 전역변수로 선언.

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.llama_v1_model_name_or_path,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.llama_v1_model_name_or_path)

    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)  # 실시간으로 중간 결과를 출력하는 스트리머.
    
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_v1_model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        device_map="auto",
    )

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        streamer=streamer,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.9
    )

    llm_chain = HuggingFacePipeline(pipeline=text_generation_pipeline)  # 허깅페이스의 pipeline을 감싸는 langchain의 wrapper

    with open(args.instruction_template, "r") as f:
        prompt_template = f.read()
    # print("prompt_template:\n", prompt_template)

    prompt_chain = PromptTemplate(  # 프롬프트 템플릿을 채워주는 역할.
        input_variables=["store", "current_input", "store_info"],
        template=prompt_template
    )

    # "all_menu", -> 프롬프트에 들어갔었던 메뉴판
    
class InputHistory:  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화를 보관하는 클래스.
    current_input = None

def initialize_history():
    global endoforder, logger
    logger.info(' ** Initialize history **')
    endoforder = False
    InputHistory.current_input = None

# def update_history(response):
#     updated_before_input = InputHistory.current_input
#     updated_before_response = response
    
#     InputHistory.before_input = updated_before_input
#     InputHistory.before_response = updated_before_response

def run_llm_chain(message):  # Thread의 target 매개변수로 입력되는 함수.
    return llm_chain.invoke(message)  # 입력된 message를 langchain 파이프라인을 통해 처리해, 언어 모델로부터 출력 생성.

is_initial = False  # 초기화 되었는지 여부
endoforder = False  # 주문 종료 여부
args.model_name_or_path = args.llama_v1_model_name_or_path

initialize_model()  # config, tokenizer, streamer, model, pipline, langchain wrapper, prompt template 한꺼번에 초기화
initialize_history()  # 이전 대화, 이전 슬롯, 이전 응답, 현재 대화 초기화


is_first = True


@app.route('/order', methods=['POST'])
def order():
    try: 
        global is_initial, endoforder, model, llm_chain, logger, is_first, is_response_yielding, message
        
        event = request.get_json(force=True)
        print("event:", event)
        message = event['body']['text'] #+ "?"
        print("message:", message)
        # message = event['inputTranscript']
        
        logger.info(event)
        
        if message == "":  # 요청이 안들어온 경우
            return "다시 말씀해 주시겠어요?"
        
        # @stream_with_context
        def generate():
            global is_initial, endoforder, is_first
            
            if is_initial == False:
                is_initial = True
                logger.info(' --- Start order ----')
                
            if is_initial and message == '초기화':
                #initialize_history()
                # response_dict['body']['text'] = "슬롯을 초기화 했습니다. 주문을 시작합니다."
                
                # init_text = "슬롯을 초기화 했습니다. 주문을 시작합니다."
                # yield init_text
                # yield json.dumps(response_dict)
                
                # yield "슬롯을 초기화 했습니다.\n\n"
                return

            InputHistory.current_input = message
            
            inputs = {
                'store': f"{args.store}",
                'current_input': InputHistory.current_input,
                # 'all_menu': all_menus_dict
            }
            logger.info(f' >> Current input: {message} <<')
            # -----------
            
            # BERT
            db_retriever = retriever_dict['db_retriever']  # 리트리버 정의
            db_retriever_result = format_docs([f"{i.page_content}" for i in db_retriever.invoke(invoke_format(inputs))])
            print("db_retriever:\n", db_retriever_result)
            if db_retriever_result == "":
                # return "죄송합니다. 관련 정보가 없습니다."
                inputs["store_info"] = "관련 정보 없음"
            else:
                # # TF-IDF
                # docs_retriever = retriever_dict['docs_retriever']  # 리트리버 정의
                # docs_retriever_result = format_docs([f"{i.page_content}" for i in docs_retriever.invoke(invoke_format(inputs))])
                # print("docs_retriever:\n", docs_retriever_result)
                
                # 앙상블
                en_retriever = retriever_dict[inputs['store']]  # 리트리버 정의
                #retriever_result = format_docs([f"{i.page_content}: {i.metadata['옵션']}" for i in retriever.invoke(invoke_format(inputs))])
                retriever_result = format_docs([f"{i.page_content}" for i in en_retriever.invoke(invoke_format(inputs))])
                print("retriever_result:\n", retriever_result)
                inputs["store_info"] = retriever_result

            start_time = time.time()
            model_input = prompt_chain.invoke(inputs)
            output = run_llm_chain(model_input)
            
            if output[:3] == "답변:":
                output = output.split("답변:")[1]
                if output[0] == " ":
                    output = output[1:]
                
            output = remove_special_characters(output) 

            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f' >> Yield streaming text:{output}<<')
            logger.info(f'*** Time taken: {total_time:.2f}s ***')
            return output
                # t= Thread(target=run_llm_chain, args=(model_input,))
                # t.start()
                # -----------
        
        return Response(stream_with_context(generate()), content_type='application/json')
    except Exception as e:
        logger.error(f'\n\n !!!!! Error raised >>> \n{e}')
    
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=False)   #5000포트에서 요청을 받음. 따라서 클라이언트에서는 해당 로컬 서버의 IP주소에서 포트 넘버 5006으로 보내야함.