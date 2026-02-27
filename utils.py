import os
import json
import logging

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
                    