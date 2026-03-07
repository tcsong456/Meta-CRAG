import os
import torch
import pickle
from prompts.no_shot_without_kg import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

class RagModel:
    def __init__(self,
                 chat_model,
                 retriever,
                 tfidf_lr,
                 domain_router,
                 dynamic_router):
        self.retriever = retriever
        domain_ckpt = 'domain_best.pth'
        dynamic_ckpt = 'dynamic_best.pth'
        domain_path = f'checkpoints/{domain_ckpt}'
        dynamic_path = f'checkpoints/{dynamic_ckpt}'
        if not os.path.exists(domain_path) or not os.path.exists(dynamic_path):
            raise KeyError('run router/router_trainer.py script first to obtain the model checkpoint first')
        
        domain_ckpt = torch.load(domain_path)
        dynamic_ckpt = torch.load(dynamic_path)
        domain_router.load_state_dict(domain_ckpt['model'])
        dynamic_router.load_state_dict(dynamic_ckpt['model'])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.domain_router = domain_router.to(device)
        self.dynamic_router = dynamic_router.to(device)
        self.tfidf_lr = tfidf_lr
        domain2id = {
            'finance': 0,
            'music': 1,
            'movie': 2,
            'sports': 3,
            'open': 4
        }
        dynamic2id = {
            'static': 0,
            'slow-changing': 1,
            'fast-changing': 2,
            'real-time': 3
        }
        self.id2domain = {v: k for k, v in domain2id.items()}
        self.id2dynamic = {v: k for k, v in dynamic2id.items()}
        
        self.initialize_models(chat_model)
    
    def initialize_models(self, chat_model):
        SYSTEM_PROMPT = 'You are a helpful assistant'
        
        self.domain2template = {
            "open": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", OPEN_PROMPT)],
            ),
            "movie": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", MOVIE_PROMPT)],
            ),
            "music": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", MUSIC_PROMPT)],
            ),
            "sports": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", SPORTS_PROMPT)],
            ),
            "finance": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", FINANCE_PROMPT)],
            )
        }
        self.rag_chain = self.format_messages_without_kg | chat_model | StrOutputParser() | self.get_final_answer_content
    
    def get_final_answer_content(self, text):
        marker = "## Final Answer"
        marker_index = text.find(marker)
        if marker_index == -1:
            return "I don't know"
        start_index = marker_index + len(marker)
        answer = text[start_index:].strip().lower()
        return answer
    
    def get_references(self, retrival_results):
        reference = ''
        if len(retrival_results) > 1:
            for retrival_result in retrival_results:
                reference = '<DOC>\n'
                reference += f'{retrival_result}\n'
                reference += '<DOC>\n\n'
        elif len(retrival_results) == 0 and len(retrival_results[0]) > 0:
            reference = retrival_results[0]
        else:
            reference = 'No References'
        return reference
    
    def format_messages_without_kg(self, input):
        query = input['query']
        domain = input['domain']
        query_time = input['query_time']
        retrieval_results = input['retrival_results']
        references = self.get_references(retrieval_results)
        messages = self.domain2template[domain].format_messages(
            query=query,
            query_time=query_time,
            domain=domain,
            references=references,
            # domain_hint=self.domain_hints[domain],
            # invalid_question_examples=self.invalid_question_examples[domain]
        )
        return messages
    
    def _normalize(self, x):
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    
    def retrieve(self, input):
        query = input['query']
        interaction_id = input['interaction_id']
        search_results = input['search_results']
        return self.retriever.retrieve(query, interaction_id, search_results)
    
    def batch_generate_answer(self, batch):
        queries = batch['query']
        interaction_ids = batch['interaction_id']
        search_results = batch['search_results']
        query_token = batch['query_token']
        query_times = batch['query_time']
        for k, v in query_token.items():
            query_token[k] = v.to('cuda')
        with torch.no_grad():
            self.domain_router.eval()
            self.dynamic_router.eval()
            domain_logits = self.domain_router(**query_token)
            dynamic_logits = self.dynamic_router(**query_token)
        
        with open('best_alpha.pkl', 'rb') as f:
            best_alpha_dict = pickle.load(f)
        best_alpha = best_alpha_dict['best_alpha']
        
        domain_logits = self._normalize(domain_logits)
        tfidf_score = self.tfidf_lr.decision_function(queries)
        tfidf_score = torch.tensor(tfidf_score, dtype=domain_logits.dtype, device=domain_logits.device)
        tfidf_score = self._normalize(tfidf_score)
        domain_logits = best_alpha * domain_logits + (1 - best_alpha) * tfidf_score
        domain_label = torch.argmax(domain_logits, dim=1)
        dynamic_label = torch.argmax(dynamic_logits, dim=1)
        domain_label = domain_label.detach().cpu().numpy()
        dynamic_label = dynamic_label.detach().cpu().numpy()
        domains = [self.id2domain[label] for label in domain_label]
        dynamics = [self.id2dynamic[label] for label in dynamic_label]
        
        batch_retrival_results = RunnableLambda(self.retrieve).batch([
            {'query': query, 'interaction_id': interaction_id, 'search_results': search_result}
            for query, interaction_id, search_result in zip(queries, interaction_ids, search_results)
            ]
        )
        
        inputs = [{'query': query, 'query_time': query_time, 'retrival_results': retrival_result, 'domain': domain} 
                  for query, query_time, retrival_result, domain 
                  in zip(queries, query_times, batch_retrival_results, domains)]
        responses = self.rag_chain.batch(inputs)
        return responses

#%%
'''
["I don't know",
 'Four films (with a prequel set to release soon).',
 "I don't know",
 "I don't know.",
 '$34.4825',
 'Justin Bieber is older.',
 'Mt. Mayon',
 'Patrick Stump',
 '$245 billion',
 '$109.71',
 'Frenkie de Jong',
 '0',
 "I don't know",
 "I don't know.",
 "I don't know.",
 "I don't know",
 "I don't know",
 "I don't know.",
 '"I don\'t know"',
 'No.',
 "I don't know",
 'Searching for Sugar Man',
 "I don't know.",
 'Yes.',
 "Robert Smith, Simon Gallup, Roger O'Donnell, Laurence Tolhurst",
 'Coca-Cola',
 "I don't know.",
 '0.00',
 "I don't know",
 'Geno Smith',
 'Aly Raisman',
 "I don't know"]

["i don't know",
 '4',
 "i don't know",
 "i don't know",
 '$34.4825',
 'justin bieber',
 'mt. mayon',
 'patrick stump',
 "i don't know",
 '$109.71',
 'frenkie de jong',
 "i don't know",
 "i don't know",
 "i don't know",
 "i don't know",
 'invalid question',
 "i don't know",
 "i don't know",
 "i don't know",
 "i don't know",
 "i don't know",
 'searching for sugar man',
 "i don't know",
 'yes.',
 "robert smith, simon gallup, roger o'donnell, laurence tolhurst",
 'coca-cola',
 "i don't know",
 '0.00',
 "i don't know",
 "i don't know",
 'aly raisman',
 "i don't know"]
'''
