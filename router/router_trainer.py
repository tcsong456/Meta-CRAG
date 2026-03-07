import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch import nn
from torch import optim
from utils import load_data, make_indices_split
from sklearn.pipeline import Pipeline
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cnt = 0
        self.sum = 0
        self.average = 0
    
    def update(self, value, n):
        self.cnt += n
        self.sum += value * n
        self.average = self.sum / max(self.cnt, 1)

class AverageMeterV1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.average = 0.0
    
    def update(self, correct, total):
        self.correct += correct
        self.total += total
        self.average = self.correct / self.total
    
class QueryDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 queries,
                 labels,
                 label2id,
                 max_len=30):
        self.tokenizer = tokenizer
        self.queries = queries
        self.labels = labels
        self.label2id = label2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        label = self.labels[idx]
        
        enc = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        label_id = self.label2id[label]
        output = {
            'input_ids': enc['input_ids'].squeeze(),
            'attn_mask': enc['attention_mask'].squeeze(),
            'label': label_id
        }
        return output

def train_tfidf_lr(text, labels, label2id):
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                lowercase=True
            )
        ),
        ('lr', LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                multi_class='auto',
                class_weight='balanced'
            )
                
        )
    ])
    labels = [label2id[label] for label in labels]
    return clf.fit(text, labels)

class BGE3ForClassification(nn.Module):
    def __init__(self,
                 num_labels,
                 freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('BAAI/bge-m3')
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.compressor = nn.Linear(hidden_size, 128)
        self.classifier = nn.Linear(128, num_labels)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
    
    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(1e-9)
        return summed / counts
    
    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        h = self.compressor(pooled)
        logits = self.classifier(h)
        return logits

class Trainer:
    def __init__(self,
                 epochs,
                 queries,
                 labels,
                 model,
                 tokenizer,
                 label2id,
                 id2label):
        self.epochs = epochs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = model.to(device)
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.optimizer = self._build_optimizer(
            lr_encoder=1e-5,
            lr_head=5e-5,
            weight_decay=0.01
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.mode = 'domain' if len(set(labels[0])) == 5 else 'dynamic'
        self.best_loss = np.inf
        self.best_accuracy = 0.0
        self.early_stopping = 1
        
        os.makedirs('checkpoints', exist_ok=True)
        self.checkpoint = f'checkpoints/{self.mode}_best.pth'
    
    def _build_optimizer(self,
                         lr_encoder,
                         lr_head,
                         weight_decay
                         ):
        no_decay = ['bias', 'Layernorm.weight', 'Layernorm.bias']
        head_keywords = ['classifier', 'score', 'pooler']
        
        encoder_decay, encoder_no_decay = [], []
        head_decay, head_no_decay = [], []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_no_decay = any(nd in name for nd in no_decay)
            is_head = any(hk in name for hk in head_keywords)
        
            if is_head:
                (head_no_decay if is_no_decay else head_decay).append(param)
            else:
                (encoder_no_decay if is_no_decay else encoder_decay).append(param)
            
        param_groups = [
            {'params': encoder_decay, 'lr': lr_encoder, 'weight_decay': weight_decay},
            {'params': encoder_no_decay, 'lr': lr_encoder, 'weight_decay': 0.0},
            {'params': head_decay, 'lr': lr_head, 'weight_decay': weight_decay},
            {'params': head_no_decay, 'lr': lr_head, 'weight_decay': 0.0}
        ]
        
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        return optimizer
    
    def train(self):
        scaler = GradScaler(enabled=True)
        
        dataset_params = {
                          'tokenizer': self.tokenizer,
                          'label2id': self.label2id}
        train_dataset = QueryDataset(
            **dataset_params,
            queries=self.queries[0],
            labels=self.labels[0],
            max_len=50)
        val_dataset = QueryDataset(
            **dataset_params,
            queries=self.queries[1],
            labels=self.labels[1],
            max_len=50)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=32
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=16
        )
        
        total_steps = 2 * int(len(self.queries[0]) // 32 + 1)
        warm_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warm_steps,
            num_training_steps=total_steps
        )
        
        for epoch in range(0, self.epochs):
            self.model.train()
            loss_meter_tr = AverageMeter()
            loss_meter_val = AverageMeter()
            acc_meter = AverageMeterV1()
            train_dl = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f'running training on {self.mode} problem'
            )
            for batch in train_dl:
                current_lr = self.optimizer.param_groups[0]['lr']
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                self.optimizer.zero_grad()
                y = batch['label']
                with autocast(enabled=True):
                    logits = self.model(
                        input_ids= batch['input_ids'],
                        attention_mask=batch['attn_mask']
                    )
                    probs = torch.softmax(logits, dim=-1)
                    loss = self.loss_fn(probs, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                bs = y.numel()
                loss_meter_tr.update(loss.item(), bs)
                loss = loss_meter_tr.average
                train_dl.set_postfix({
                    f'epoch {epoch} loss': f'{loss: .5f}',
                    'lr': f'{current_lr: .5f}'
                })
                
            with torch.no_grad():
                self.model.eval()
                val_dl = tqdm(
                    val_dataloader,
                    total=len(val_dataloader),
                    desc=f'running evaluation on {self.mode} problem'
                )
                bge_logits = []
                for batch in val_dl:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(self.device)
                    y = batch['label']
                    with autocast(enabled=True):
                        logits, h = self.model(
                            input_ids= batch['input_ids'],
                            attention_mask=batch['attn_mask']
                        )
                        probs = torch.softmax(logits, dim=-1)
                        val_loss = self.loss_fn(probs, y)
                        y_pred = logits.argmax(dim=-1)
                        bge_logits.append(logits)
                    
                    correct = (y_pred == y).sum()
                    bs = y.numel()
                    loss_meter_val.update(val_loss.item(), bs)
                    acc_meter.update(correct.item(), bs)
                    val_loss = loss_meter_val.average
                    accuracy = acc_meter.average
                    val_dl.set_postfix({
                        f'epoch {epoch} loss': f'{val_loss: .5f}',
                        'accuracy': f'{accuracy: .3f}'
                      })
                
                if val_loss < self.best_loss and accuracy > self.best_accuracy:
                    self.best_loss = val_loss
                    self.best_accuracy = accuracy
                    bad_epoch = 0
                    torch.save({
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_loss': self.best_loss,
                        'accuracy': self.best_accuracy,
                        'epoch': epoch    
                    }, self.checkpoint)
                    best_logits = bge_logits
                else:
                    bad_epoch += 1
                
                if bad_epoch == self.early_stopping or self.best_accuracy == 1.0:
                    print(f'early stopping reaches at epoch: {epoch}')
                    return best_logits
        return best_logits
    
    @torch.no_grad()
    def predict(self, convert_to_text=False):
        dataset = QueryDataset(
            tokenizer=self.tokenizer,
            queries=self.queries[2],
            labels=self.labels[2],
            label2id=self.label2id,
            max_len=50
        )
        test_dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False
        )
        
        self.model.load_state_dict(f'{self.checkpoint}')
        self.model.eval()
        test_dl = tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc=f'predicting on the {self.mode} problem'
        )
        pred_labels = []
        for batch in test_dl:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            with autocast(enabled=True):
                logits, h = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attn_mask']
                )
                
            pred_label = logits.argmax(dim=-1)
            pred_labels.append(pred_label.detach().cpu().numpy())
        
        pred_labels = np.concatenate(pred_labels)
        if convert_to_text:
            pred_labels = [self.id2label[pred_label] for pred_label in pred_labels]
            
        return pred_labels

def parse_args(argv=None):
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, choices=['domain', 'dynamic'], required=True)
    parser.add_argument('--fold', type=int, choices=[0,1,2,3,4], required=True)
    args = parser.parse_args(argv)
    return args

def ensemble_avg_logits(bge_logits, tfidf_scores, alpha=0.5):
    tfidf_scores = torch.tensor(tfidf_scores, dtype=bge_logits.dtype, device=bge_logits.device)
    tfidf_scores = normalize_logits(tfidf_scores)
    bge_logits = normalize_logits(bge_logits)
    fused_logits = alpha * bge_logits + (1 - alpha) * tfidf_scores
    preds = fused_logits.argmax(dim=-1)
    return preds

def normalize_logits(x):
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.field == 'domain':
        label2id = {'finance': 0,
                    'music': 1,
                    'movie': 2,
                    'sports': 3,
                    'open': 4}
        field = 'domain'
    else:
        label2id = {
            'static': 0,
            'slow-changing': 1,
            'fast-changing': 2,
            'real-time': 3
        }
        field = 'static_or_dynamic'
    id2label = {v: k for k, v in label2id.items()}
    
    model = BGE3ForClassification(
        num_labels=len(label2id),
        freeze_encoder=False
    )
    
    data = load_data('crag_task_3_dev_v4')
    train_indices, val_indices, test_indices = make_indices_split(len(data['query']), n_splits=5, pick_fold=args.fold)
    
    query = np.array(data['query'])
    label = np.array(data[field])
    interaction_id = np.array(data['interaction_id'])
    search_results = data['search_results']
    answer = np.array(data['answer'])
    query_time = np.array(data['query_time'])
    sr_train = [search_results[ind] for ind in train_indices]
    sr_val = [search_results[ind] for ind in val_indices]
    sr_test = [search_results[ind] for ind in test_indices]
    
    queries, fields, interaction_ids, results, query_times, answers = [], [], [], [], [], []
    for indices in [train_indices, val_indices, test_indices]:
        queries.append(query[indices])
        fields.append(label[indices])
        interaction_ids.append(interaction_id[indices])
        query_times.append(query_time[indices])
        answers.append(answer[indices])
    results.append(sr_train)
    results.append(sr_val)
    results.append(sr_test)
    
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    
    trainer = Trainer(
        epochs=2,
        queries=queries,
        labels=fields,
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label
    )
    bge_logits = trainer.train()
    bge_logits = torch.cat(bge_logits)
    
    if args.field == 'domain':
        import joblib
        
        tfidf_lr = train_tfidf_lr(queries[0], fields[0], label2id)
        joblib.dump(tfidf_lr, 'models/tfidf_lr.joblib')
        score = tfidf_lr.decision_function(queries[1])
        
        accs = []
        label = torch.tensor([label2id[f] for f in fields[1]], device='cuda:0')
        alphas = np.linspace(0.0, 1.0, 101)
        for alpha in alphas:
            pred = ensemble_avg_logits(bge_logits, score, alpha)
            acc = (pred == label).sum() / label.shape[0]
            accs.append(acc.item())
        max_id = np.argmax(accs)
        best_alpha = alphas[max_id]
        best_acc = accs[max_id]
        logging.info(f'best alpha for fusion model is: {best_alpha: .2f}, best accuracy: {best_acc: .3f}')
        
        with open('best_alpha.pkl', 'wb') as f:
            pickle.dump({"best_alpha": best_alpha}, f)
    
        os.makedirs('artifacts', exist_ok=True)
        np.save('artifacts/test_queries.npy', queries[2])
        np.save('artifacts/test_interaction_id.npy', interaction_ids[2])
        np.save('artifacts/test_query_time.npy', query_times[2])
        np.save('artifacts/test_answers.npy', answers[2])
        with open('artifacts/test_search_results.pkl', 'wb') as f:
            pickle.dump(results[2], f)



