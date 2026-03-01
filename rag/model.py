from langchain_core.runnables import RunnableLambda

class RagModel:
    def __init__(self,
                 retriever):
        self.retriever = retriever
    
    def retrieve(self, input):
        query = input['query']
        interaction_id = input['interaction_id']
        search_results = input['search_results']
        return self.retriever.retrieve(query, interaction_id, search_results)
    
    def batch_generate_answer(self, batch):
        queries = batch['query']
        interaction_ids = batch['interaction_id']
        search_results = batch['search_results']
        
        batch_retrival_results = RunnableLambda(self.retrieve).batch([
            {'query': query, 'interaction_id': interaction_id, 'search_results': search_result}
            for query, interaction_id, search_result in zip(queries, interaction_ids, search_results)
            ]
        )
        return batch_retrival_results