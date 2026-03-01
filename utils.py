import os
import json
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(path, batch_size):
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
train_indices, val_indices, test_indices = make_indices_split(2500, n_splits=5, pick_fold=2)



#%%
len()
                    