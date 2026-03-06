import os 
import joblib
import pickle
import numpy as np
from rag.model import RagModel
from transformers import AutoTokenizer
from utils import batch_load_data, load_model
from router.router_trainer import BGE3ForClassification
from retriever.rerank_retriever import Retriever
from huggingface_hub import snapshot_download

# data_loader = load_data('crag_task_3_dev_v4', 32)
# data_loader = batch_load_data('crag_task_3_dev_v4', 32)
batch_size = 32

embedding_model_path = "models/bge-m3"
reranker_model_path = "models/bge-reranker-v2-m3"
if not os.path.exists(reranker_model_path):
    snapshot_download(
        repo_id='BAAI/bge-reranker-v2-m3',
        local_dir=reranker_model_path,
        local_dir_use_symlinks=False
    )

if not os.path.exists(embedding_model_path):
    snapshot_download(
        repo_id='BAAI/bge-m3',
        local_dir=embedding_model_path,
        local_dir_use_symlinks=False
    )

retriever = Retriever(
    top_k=5,
    top_n=3,
    top_preliminary=15,
    embedding_model_path=embedding_model_path,
    rerank_model_path=reranker_model_path,
    rerank=True,
    device='cuda',
    timeout=90   
)
api_key = os.getenv("OPENAI_API_KEY")
llm_model = load_model(model_name='gpt-4o', api_key=api_key)
tfidf_lr = joblib.load('models/tfidf_lr.joblib')
domain_router = BGE3ForClassification(num_labels=5)
dynamic_router = BGE3ForClassification(num_labels=4)

batches = []
retrieval_results = []
rag_model = RagModel(
    chat_model=llm_model,
    retriever=retriever,
    tfidf_lr=tfidf_lr,
    domain_router=domain_router,
    dynamic_router=dynamic_router
)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
queries = np.load('artifacts/test_queries.npy')
interaction_ids = np.load('artifacts/test_interaction_id.npy')
query_time = np.load('artifacts/test_query_time.npy')
answers = np.load('artifacts/test_answers.npy')
with open('artifacts/test_search_results.pkl', 'rb') as f:
    search_results = pickle.load(f)
total_len = len(queries)

query_list, answer_list, responses = [], [], []
for i in range(0, total_len, batch_size):
    query = queries[i: i+batch_size]
    interaction_id = interaction_ids[i: i+batch_size]
    search_result = search_results[i: i+batch_size]
    query_time = query_time[i: i+batch_size]
    answer = answers[i: i+batch_size]
    query_tokens = tokenizer(
        list(query),
        truncation=True,
        max_length=50,
        padding='max_length',
        return_tensors='pt'
    )
    batch = {'query': query,
             'query_token': query_tokens,
             'search_results': search_result,
             'interaction_id': interaction_id,
             'query_time': query_time}
    response = rag_model.batch_generate_answer(batch)
    query_list.append(query)
    answer_list.append(answer)
    responses.append(response)
    break
    

# for batch in data_loader:
#     query = batch['query']
#     query_tokens = tokenizer(
#         list(query),
#         truncation=True,
#         max_length=50,
#         padding='max_length',
#         return_tensors='pt'
#     )
#     batch['query_token'] = query_tokens
#     retrieval_result = rag_model.batch_generate_answer(batch)
#     retrieval_results.append(retrieval_result)
#     # batches.append(batch)
#     break


#%%
response, answer