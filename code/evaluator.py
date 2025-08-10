import torch
import numpy as np
from tqdm import tqdm
import time

class Evaluator:
    def __init__(self, model, device, data_loader, args):
        self.model = model
        self.device = device
        self.args = args
        self.entity2id = data_loader.entity2id
        self.rel2id = data_loader.rel2id
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.rel2id)
        
        # 预加载所有真实三元组，用于过滤评估
        self.all_triples = set()
        for split in ['train', 'valid', 'test']:
            triples = data_loader.get_id_triples(split)
            for h, r, t in triples:
                self.all_triples.add((h, r, t))
        
        # 创建实体ID张量（用于批量处理）
        self.all_entity_ids = torch.arange(self.num_entities, device=self.device)
        
        # 缓存过滤矩阵以提高效率
        self.filter_cache = {}

    def get_filter_mask(self, h, r, mode='tail'):
        """生成过滤掩码（高效实现）"""
        cache_key = (h, r, mode) if mode == 'tail' else (r, h, mode)
        
        if cache_key in self.filter_cache:
            return self.filter_cache[cache_key]
        
        if mode == 'tail':
            # 尾实体过滤：找出所有(h, r, ?)的真实三元组
            filter_set = {t for (h_val, r_val, t) in self.all_triples 
                         if h_val == h and r_val == r}
            mask = torch.zeros(self.num_entities, dtype=torch.bool, device=self.device)
            for t_val in filter_set:
                mask[t_val] = True
        else:
            # 头实体过滤：找出所有(?, r, t)的真实三元组
            filter_set = {h for (h, r_val, t_val) in self.all_triples 
                         if r_val == r and t_val == h}  # 注意：这里的h实际上是t
            mask = torch.zeros(self.num_entities, dtype=torch.bool, device=self.device)
            for h_val in filter_set:
                mask[h_val] = True
        
        self.filter_cache[cache_key] = mask
        return mask

    def evaluate_tail_prediction_batch(self, test_triples, batch_size=64):
        """批量评估尾实体预测任务"""
        metrics = {'MR': 0.0, 'MRR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
        num_test = len(test_triples)
        
        # 批量处理测试三元组
        for i in tqdm(range(0, num_test, batch_size), desc="评估尾实体预测"):
            batch_triples = test_triples[i:i+batch_size]
            batch_size_actual = len(batch_triples)
            
            # 创建候选三元组: (h, r, e) for each e in entities
            h_batch = torch.tensor(batch_triples[:, 0], device=self.device)
            r_batch = torch.tensor(batch_triples[:, 1], device=self.device)
            t_true = torch.tensor(batch_triples[:, 2], device=self.device)
            
            # 扩展为 [batch_size * num_entities, 3]
            h_exp = h_batch.repeat_interleave(self.num_entities)
            r_exp = r_batch.repeat_interleave(self.num_entities)
            e_exp = self.all_entity_ids.repeat(batch_size_actual)
            candidate_triples = torch.stack([h_exp, r_exp, e_exp], dim=1)
            
            # 批量计算得分
            with torch.no_grad():
                scores = self.model.predict(candidate_triples)
                scores = scores.view(batch_size_actual, self.num_entities)
            
            # 处理每个三元组
            for j in range(batch_size_actual):
                h, r, t = batch_triples[j]
                t_val = t_true[j].item()
                
                # 应用过滤（排除真实存在的三元组，除了当前目标）
                filter_mask = self.get_filter_mask(h, r, mode='tail')
                scores_j = scores[j].clone()
                scores_j[filter_mask & (self.all_entity_ids != t_val)] = -float('inf')
                
                # 获取排名
                sorted_indices = torch.argsort(scores_j, descending=True)
                rank = (sorted_indices == t_val).nonzero().item() + 1
                
                # 更新指标
                metrics['MR'] += rank
                metrics['MRR'] += 1.0 / rank
                metrics['HITS@1'] += 1 if rank <= 1 else 0
                metrics['HITS@3'] += 1 if rank <= 3 else 0
                metrics['HITS@10'] += 1 if rank <= 10 else 0
        
        # 计算平均值
        for key in metrics:
            metrics[key] /= num_test
        return metrics

    def evaluate_head_prediction_batch(self, test_triples, batch_size=64):
        """批量评估头实体预测任务"""
        metrics = {'MR': 0.0, 'MRR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
        num_test = len(test_triples)
        
        # 批量处理测试三元组
        for i in tqdm(range(0, num_test, batch_size), desc="评估头实体预测"):
            batch_triples = test_triples[i:i+batch_size]
            batch_size_actual = len(batch_triples)
            
            # 创建候选三元组: (e, r, t) for each e in entities
            t_batch = torch.tensor(batch_triples[:, 2], device=self.device)
            r_batch = torch.tensor(batch_triples[:, 1], device=self.device)
            h_true = torch.tensor(batch_triples[:, 0], device=self.device)
            
            # 扩展为 [batch_size * num_entities, 3]
            t_exp = t_batch.repeat_interleave(self.num_entities)
            r_exp = r_batch.repeat_interleave(self.num_entities)
            e_exp = self.all_entity_ids.repeat(batch_size_actual)
            candidate_triples = torch.stack([e_exp, r_exp, t_exp], dim=1)
            
            # 批量计算得分
            with torch.no_grad():
                scores = self.model.predict(candidate_triples)
                scores = scores.view(batch_size_actual, self.num_entities)
            
            # 处理每个三元组
            for j in range(batch_size_actual):
                h, r, t = batch_triples[j]
                h_val = h_true[j].item()
                
                # 应用过滤（排除真实存在的三元组，除了当前目标）
                filter_mask = self.get_filter_mask(t, r, mode='head')  # 注意参数顺序
                scores_j = scores[j].clone()
                scores_j[filter_mask & (self.all_entity_ids != h_val)] = -float('inf')
                
                # 获取排名
                sorted_indices = torch.argsort(scores_j, descending=True)
                rank = (sorted_indices == h_val).nonzero().item() + 1
                
                # 更新指标
                metrics['MR'] += rank
                metrics['MRR'] += 1.0 / rank
                metrics['HITS@1'] += 1 if rank <= 1 else 0
                metrics['HITS@3'] += 1 if rank <= 3 else 0
                metrics['HITS@10'] += 1 if rank <= 10 else 0
        
        # 计算平均值
        for key in metrics:
            metrics[key] /= num_test
        return metrics

    def evaluate(self, test_triples, batch_size=64):
        """综合评估头实体和尾实体预测"""
        print("开始评估...")
        start_time = time.time()
        
        # 评估尾实体预测
        tail_metrics = self.evaluate_tail_prediction_batch(test_triples, batch_size)
        print(f"尾实体预测完成 | 耗时: {time.time()-start_time:.1f}秒")
        
        # 评估头实体预测
        head_metrics = self.evaluate_head_prediction_batch(test_triples, batch_size)
        print(f"头实体预测完成 | 总耗时: {time.time()-start_time:.1f}秒")
        
        # 合并结果
        combined_metrics = {}
        for key in tail_metrics:
            # 头尾实体预测的平均值
            combined_metrics[key] = (tail_metrics[key] + head_metrics[key]) / 2
        
        # 打印结果
        print("\n评估结果:")
        print(f"MR: {combined_metrics['MR']:.1f}")
        print(f"MRR: {combined_metrics['MRR']:.4f}")
        print(f"Hits@1: {combined_metrics['HITS@1']:.4f}")
        print(f"Hits@3: {combined_metrics['HITS@3']:.4f}")
        print(f"Hits@10: {combined_metrics['HITS@10']:.4f}")
        
        return combined_metrics