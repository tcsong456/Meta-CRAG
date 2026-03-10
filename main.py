import os
import json
import joblib
import pickle
import numpy as np
from tqdm import tqdm
from rag.model import RagModel
from transformers import AutoTokenizer
from utils import load_model, logger
from router.router_trainer import BGE3ForClassification
from retriever.rerank_retriever import Retriever
from huggingface_hub import snapshot_download
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.no_shot_without_kg import EVALUATION_INSTRUCTIONS

def parse_response(resp):
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if model_resp['accuracy'] is True or model_resp['accuracy'].lower() == 'true':
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")
        return answer
    except:
        return -1

def evaluation(queries, ground_truths, predictions, evaluation_model):
    n_miss, n_correct, n_exact_correct = 0, 0, 0
    system_message = EVALUATION_INSTRUCTIONS
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n")]
    )
    chain = prompt_template | evaluation_model | StrOutputParser()
    
    messages = []
    for i, prediction in enumerate(tqdm(predictions, total=len(predictions), desc='evaluating the predictions')):
        query = queries[i]
        ground_truth = ground_truths[i].lower()
        prediction = prediction.strip().lower()
        
        if "i don't know" in ground_truth:
            n_miss += 1
            continue
        elif prediction == ground_truth:
            n_correct += 1
            n_exact_correct += 1
            continue
        messages.append({"query": query, "ground_truth": ground_truth, "prediction": prediction})

    evaluations = chain.batch(messages)

    for evaluation in evaluations:
        eval_res = parse_response(evaluation)
        if eval_res == 1:
            n_correct += 1
    
    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_exact_correct / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_exact_correct,
        "total": n,
    }
    logger.info(results)
    return results
  
if __name__ == '__main__':
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
    
    evaluation_model = load_model(model_name='gpt-4.1', api_key=api_key)
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
        
        results = evaluation(query, answer, response, evaluation_model)
        break

#%%