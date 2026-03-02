import os
import json
import pickle
import logging
import numpy as np
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_name='gpt-4o', base_url=None, api_key=None, **kwargs):
    if api_key is None and base_url is None:
        model = ChatOpenAI(model_name=model_name, **kwargs)
    elif api_key is not None and base_url is None:
        model = ChatOpenAI(model_name=model_name, api_key=api_key, **kwargs)
    elif api_key is None and base_url is not None:
        model = ChatOpenAI(model_name=model_name, base_url=base_url, **kwargs)
    else:
        model = model = ChatOpenAI(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
    return model

def load_data(path):
    data = {'query': [], 'domain': [], 'static_or_dynamic': [], 'query_time': [],
            'interaction_id': [], 'search_results': []}
    file_paths = os.listdir(path)
    for i, file_path in enumerate(file_paths):
        with open(f'{path}/{file_path}', 'r') as f:
            total_files = 300 if i < 9 else 6
            tqf = tqdm(f, total=total_files, desc=f'downloading files from file number: {i}')
            for line in tqf:
                file = json.loads(line)
                for key in data:
                    data[key].append(file[key])
    return data

def batch_load_data(path, batch_size):
    def initilize_batch():
        return {'interaction_id': [], 'query':[], 'search_results':[], 'query_time':[], 'answer': [],
                'domain': [], 'static_or_dynamic': []}
    
    file_paths = os.listdir(path)
    for file_path in file_paths:
        with open(f'{path}/{file_path}', 'r') as f:
            batch = initilize_batch()
            for line in f:
                try:
                    data = json.loads(line)
                    for key in batch:
                        batch[key].append(data[key])
                    
                    if len(batch['query']) == batch_size:
                        yield batch
                        batch = initilize_batch()
                except json.JSONDecodeError:
                    logger.warninig('Failed to decode a line')
            
            if batch['query']:
                yield batch

def make_indices_split(n, n_splits=3, test_size=0.1, seed=8719, pick_fold=0):
    indices = np.arange(n)
    y_indices = np.random.randint(0, 2, (n,))
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_indices, test_indices) in enumerate(skf.split(indices, y_indices)):
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
        if i == pick_fold:
            return train_indices, val_indices, test_indices
    
    return [], [], []
    

#%%
                    