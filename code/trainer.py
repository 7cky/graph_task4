import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import time
import copy
from evaluator import Evaluator  

class Trainer:
    def __init__(self, model, args, train_triples, valid_triples, test_triples, all_triples):
        self.model = model
        self.args = args
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.all_triples = all_triples
        
        # 优化器配置
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控验证集MRR
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 记录训练状态
        self.best_mrr = -1.0
        self.best_epoch = 0
        self.train_losses = []
        self.valid_mrrs = []
        self.current_epoch = 0
        self.stagnant_epochs = 0  # 用于早停
        self.max_stagnant = 30    # 早停阈值
        
        # 模型保存路径
        self.model_save_dir = "../saved_models"
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.model_save_path = os.path.join(
            self.model_save_dir, 
            f"best_{args.model.lower()}_{args.dataset}.pth"
        )
        
        # 初始化评估器
        self.evaluator = Evaluator(
            model=self.model,
            device=self.args.device,
            data_loader=None,  # 需要从外部传入
            args=self.args
        )
        # 设置评估器的过滤三元组
        self.evaluator.all_triples = self.all_triples
        
        print(f"训练器初始化完成 | 模型: {args.model} | 数据集: {args.dataset}")
        print(f"最佳模型将保存至: {self.model_save_path}")

    def _generate_neg_samples_batch(self, positive_triples):
        """批量生成负样本（高效实现）"""
        batch_size = positive_triples.shape[0]
        neg_samples = torch.zeros(batch_size, self.args.neg_ratio, 3, dtype=torch.long)
        
        for i in range(batch_size):
            h, r, t = positive_triples[i]
            
            for j in range(self.args.neg_ratio):
                if torch.rand(1) < 0.5:  # 替换头实体
                    h_neg = torch.randint(0, self.model.num_entities, (1,))
                    while (h_neg.item(), r, t) in self.all_triples:
                        h_neg = torch.randint(0, self.model.num_entities, (1,))
                    neg_samples[i, j] = torch.tensor([h_neg.item(), r, t])
                else:  # 替换尾实体
                    t_neg = torch.randint(0, self.model.num_entities, (1,))
                    while (h, r, t_neg.item()) in self.all_triples:
                        t_neg = torch.randint(0, self.model.num_entities, (1,))
                    neg_samples[i, j] = torch.tensor([h, r, t_neg.item()])
                    
        return neg_samples.view(-1, 3)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        batch_size = self.args.batch_size
        
        # 打乱训练数据
        indices = np.random.permutation(len(self.train_triples))
        num_batches = len(indices) // batch_size
        
        # 使用tqdm显示进度条
        batch_iter = tqdm(range(num_batches), desc=f"Epoch {self.current_epoch} 训练中", position=0, leave=True)
        
        for i in batch_iter:
            self.optimizer.zero_grad()
            
            # 获取当前批次的正样本
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            pos_triples = self.train_triples[batch_indices]
            
            # 生成负样本
            neg_triples = self._generate_neg_samples_batch(pos_triples)
            
            # 移动到设备
            pos_triples = torch.LongTensor(pos_triples).to(self.args.device)
            neg_triples = neg_triples.to(self.args.device)
            
            # 前向传播和损失计算
            loss = self.model(pos_triples, neg_triples)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            batch_iter.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate_on_validation(self):
        """在验证集上评估模型"""
        print(f"Epoch {self.current_epoch} - 在验证集上评估模型...")
        start_time = time.time()
        
        # 使用评估器进行批量评估
        metrics = self.evaluator.evaluate(self.valid_triples, batch_size=256)
        
        # 记录验证集MRR
        valid_mrr = metrics['MRR']
        self.valid_mrrs.append(valid_mrr)
        
        print(f"验证集评估完成 | 耗时: {time.time()-start_time:.1f}秒")
        print(f"验证集指标: MRR={valid_mrr:.4f}, Hits@1={metrics['HITS@1']:.4f}, Hits@10={metrics['HITS@10']:.4f}")
        
        return valid_mrr

    def should_stop(self):
        """检查是否应该停止训练（早停机制）"""
        # 如果当前MRR比最佳MRR低，增加停滞计数器
        if self.valid_mrrs[-1] <= self.best_mrr:
            self.stagnant_epochs += 1
        else:
            self.stagnant_epochs = 0
            
        # 如果连续多个epoch没有提升，则停止训练
        return self.stagnant_epochs >= self.max_stagnant

    def save_model(self, epoch, mrr):
        """保存当前模型"""
        # 保存模型状态
        torch.save(self.model.state_dict(), self.model_save_path)
        
        # 保存最佳模型信息
        self.best_mrr = mrr
        self.best_epoch = epoch
        
        print(f"保存最佳模型（Epoch {epoch}, MRR={mrr:.4f}）至 {self.model_save_path}")

    def train(self):
        """完整的训练流程"""
        print(f"\n开始训练 {self.args.model} 模型...")
        print(f"训练配置: Epochs={self.args.epochs}, Batch Size={self.args.batch_size}, LR={self.args.lr}")
        print(f"负样本比例: {self.args.neg_ratio}, 设备: {self.args.device}")
        
        # 训练前先评估初始模型
        init_valid_mrr = self.evaluate_on_validation()
        self.best_mrr = init_valid_mrr
        self.save_model(0, init_valid_mrr)
        
        # 训练循环
        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            
            # 训练一个epoch
            start_time = time.time()
            epoch_loss = self.train_epoch()
            epoch_time = time.time() - start_time
            
            print(f"训练完成 | 耗时: {epoch_time:.1f}秒 | 平均损失: {epoch_loss:.6f}")
            
            # 每N个epoch或在最后几个epoch评估一次
            eval_freq = 5 if epoch < 50 else 10
            if epoch % eval_freq == 0 or epoch == self.args.epochs or epoch <= 10:
                valid_mrr = self.evaluate_on_validation()
                
                # 更新学习率
                self.scheduler.step(valid_mrr)
                
                # 保存最佳模型
                if valid_mrr > self.best_mrr:
                    self.save_model(epoch, valid_mrr)
            
            # 检查早停条件
            if self.should_stop():
                print(f"\n早停触发! 连续 {self.max_stagnant} 个epoch验证集MRR未提升")
                print(f"最佳模型在Epoch {self.best_epoch}, MRR={self.best_mrr:.4f}")
                break
        
        # 最终评估
        print("\n训练完成! 加载最佳模型进行最终评估...")
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.args.device))
        print(f"加载最佳模型（Epoch {self.best_epoch}, MRR={self.best_mrr:.4f}）")
        
        # 在测试集上评估
        test_metrics = self.evaluator.evaluate(self.test_triples, batch_size=256)
        
        print("\n最终测试集指标:")
        print(f"MR: {test_metrics['MR']:.1f}")
        print(f"MRR: {test_metrics['MRR']:.4f}")
        print(f"Hits@1: {test_metrics['HITS@1']:.4f}")
        print(f"Hits@3: {test_metrics['HITS@3']:.4f}")
        print(f"Hits@10: {test_metrics['HITS@10']:.4f}")
        
        # 保存训练结果
        results = {
            'model': self.args.model,
            'dataset': self.args.dataset,
            'best_epoch': self.best_epoch,
            'best_mrr': self.best_mrr,
            'test_metrics': test_metrics,
            'train_losses': self.train_losses,
            'valid_mrrs': self.valid_mrrs,
            'hyperparameters': vars(self.args)
        }
        
        return results

    def plot_training_curve(self, save_path=None):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, 'b-', label='训练损失')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 验证集MRR曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.valid_mrrs, 'r-', label='验证集MRR')
        plt.title('验证集MRR曲线')
        plt.xlabel('Epoch')
        plt.ylabel('MRR')
        plt.axvline(x=self.best_epoch, color='g', linestyle='--', label=f'最佳epoch {self.best_epoch}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"训练曲线保存至: {save_path}")
        else:
            plt.show()