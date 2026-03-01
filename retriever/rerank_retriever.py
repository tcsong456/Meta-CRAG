import ray
import trafilatura
import html as html_lib
from utils import logger
from bs4 import BeautifulSoup
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt")
_ = stopwords.words("english")

def html2text(html):
    html = str(html)
    if html is None or not html.strip():
        return ''
    
    try:
        text = trafilatura.extract(
            html,
            output_format='txt',
            include_comments=False,
            include_tables=True,
            favor_recall=True
        )
    except Exception as e:
        logger.warn(f'error: {e} encountered, fall back to using beautiful soup')
        text = None
    if not text or not text.strip():
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
    
    if not text:
        return ''
    text = ' '.join(text.split())
    return text

@ray.remote
def extract_text_task(page_html):
    return html2text(page_html)

class Retriever:
    def __init__(self,
                 top_k,
                 top_n,
                 top_preliminary,
                 embedding_model_path,
                 rerank_model_path,
                 rerank=False,
                 device='cuda',
                 timeout=30):
        self.top_k = top_k
        self.top_n = top_n
        self.top_preliminary = top_preliminary
        self.embdding_model = HuggingFaceEmbedding(
            model_name=embedding_model_path,
            device=device
        )
        if rerank:
            self.reranker = SentenceTransformerRerank(
                top_n=self.top_n,
                model=rerank_model_path,
                device=device
            )
        self.rerank = rerank
        self.timeout = timeout
    
    def retrieve(self,
                 query,
                 interaction_id,
                 search_results):
        documents = []
        page_urls = set()
        search_results = [result for result in search_results if not (result['page_url'] in page_urls and page_urls.add(result['page_url']))]
        
        task_refs, snippets = [], []
        for result in search_results:
            page_html = result.get('page_result', '')
            page_snippet = result.get('page_snippet', '')
            snippets.append(html_lib.unescape(page_snippet))
            task_refs.append(extract_text_task.remote(page_html))
        
        ready, not_ready = ray.wait(task_refs, num_returns=len(task_refs), timeout=self.timeout)
        for ref in not_ready:
            logger.warning('Timeout passed, cancel the task')
            ray.cancel(ref, force=True)
        ref_to_text = {ref: ray.get(ref) for ref in ready}
        
        results = []
        for ref, snippet in zip(task_refs, snippets):
            text = ref_to_text.get(ref, '')
            results.append((text, snippet))
        
        for text, snippet in results:
            if text:
                documents.append(Document(text=text))
            if snippet:
                documents.append(Document(text=snippet))
               
        if len(search_results) > 5:
            node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            nodes = node_parser.get_nodes_from_documents(documents)
            if len(nodes) < self.top_preliminary:
                logger.warning(f"Not enough nodes for BM25 retrieval. Using all nodes({len(nodes)}).")
                self.top_preliminary = len(nodes)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.top_preliminary)
            nodes = bm25_retriever.retrieve(query)
            documents = [Document(text=node.get_text().strip()) for node in nodes]
        
        node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, embed_model=self.embdding_model)
        retriever = index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)
        
        if self.rerank:
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                QueryBundle(query_str=query)
            )
            top_candidates = [node.get_text().strip() for node in reranked_nodes]
        else:
            top_candidates = [node.get_text().strip() for node in nodes]
        
        return top_candidates

