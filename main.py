import os 
from utils import load_data
from rag.model import RagModel
from retriever.rerank_retriever import Retriever
from huggingface_hub import snapshot_download

data_loader = load_data('crag_task_3_dev_v4', 32)

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

batches = []
retrieval_results = []
rag_model = RagModel(retriever=retriever)
for batch in data_loader:
    retrieval_result = rag_model.batch_generate_answer(batch)
    retrieval_results.append(retrieval_result)
    batches.append(batch)


#%%
# batches[0]['query'][0], retrieval_results[0]