import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_loader import DataLoader
from model import TransE, RotatE, ConvE
from trainer import Trainer
from evaluator import Evaluator
import json
import time

def save_results(results, model_name, dataset, save_dir='../results'):
    """保存评估结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{model_name}_{dataset}_{timestamp}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return save_path

def set_model_specific_params(args):
    """为不同模型设置专属超参数默认值"""
    if args.model == 'RotatE':
        # RotatE最佳实践参数
        if args.emb_dim == 100:
            args.emb_dim = 200
        if args.gamma == 12.0:
            args.gamma = 24.0  # 关键修复：增大gamma值
        if args.lr == 0.001:
            args.lr = 0.0001
        if args.neg_ratio < 100:
            args.neg_ratio = 256  # RotatE需要更多负样本
        args.negative_adversarial_sampling = True
        
    elif args.model == 'ConvE':
        # ConvE最佳实践参数
        if args.emb_dim == 100:
            args.emb_dim = 200
        if args.lr == 0.001:
            args.lr = 0.001
        if args.batch_size > 256:
            args.batch_size = 128  # ConvE需要较小的批次大小
            
    elif args.model == 'TransE':
        # TransE最佳实践参数
        if args.gamma == 12.0:
            args.gamma = 10.0
        if args.lr == 0.001:
            args.lr = 0.001
        if args.neg_ratio < 32:
            args.neg_ratio = 64
    
    # 确保ConvE的嵌入维度是完全平方数
    if args.model == 'ConvE':
        root = int(args.emb_dim ** 0.5)
        if root * root != args.emb_dim:
            # 找到最接近的完全平方数
            new_dim = root * root
            print(f"警告: ConvE要求嵌入维度是完全平方数，{args.emb_dim}不是完全平方数。")
            print(f"自动将嵌入维度调整为: {new_dim}")
            args.emb_dim = new_dim
    
    return args

def main():
    parser = argparse.ArgumentParser(description="知识图谱嵌入模型训练与评估")
    
    # 模型与数据集配置
    parser.add_argument('--model', type=str, default='TransE', choices=['TransE', 'RotatE', 'ConvE'])
    parser.add_argument('--dataset', type=str, default='FB15k-237', choices=['FB15k-237', 'WN18RR', 'YAGO3-10'])
    parser.add_argument('--emb_dim', type=int, default=100, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=4608, help='ConvE隐藏层维度(自动计算)')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--gamma', type=float, default=12.0, help='损失函数中的边际参数')
    parser.add_argument('--neg_ratio', type=int, default=64, help='每个正样本的负样本数量')
    parser.add_argument('--regularization', type=float, default=1e-5, help='L3正则化系数')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW权重衰减系数')
    
    # 负采样配置
    parser.add_argument('--negative_adversarial_sampling', action='store_true', help='启用负对抗采样')
    parser.add_argument('--adversarial_temperature', type=float, default=1.0, help='对抗采样温度')
    
    # 设备与日志配置
    parser.add_argument('--cuda', action='store_true', help='使用GPU')
    parser.add_argument('--log_steps', type=int, default=10, help='每多少轮打印一次损失')
    parser.add_argument('--eval_freq', type=int, default=5, help='每多少轮评估一次验证集')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 根据模型类型自动调整超参数
    args = set_model_specific_params(args)
    
    # 设备选择
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    args.device = device
    
    print("\n" + "="*50)
    print(f"开始 {args.model} 在 {args.dataset} 上的训练")
    print(f"使用设备: {device}")
    print(f"超参数配置:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # 数据加载
    start_time = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data', args.dataset)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")
    
    print(f"加载数据集: {data_path}")
    data_loader = DataLoader(data_path)
    
    train_triples = data_loader.get_id_triples('train')
    valid_triples = data_loader.get_id_triples('valid')
    test_triples = data_loader.get_id_triples('test')
    
    # 创建所有真实三元组的集合（用于负样本过滤）
    all_triples = set()
    for triple in np.vstack([train_triples, valid_triples, test_triples]):
        all_triples.add(tuple(triple))
    
    print(f"数据集统计: "
          f"训练集={len(train_triples):,} | "
          f"验证集={len(valid_triples):,} | "
          f"测试集={len(test_triples):,}")
    print(f"实体数={len(data_loader.entity2id):,} | "
          f"关系数={len(data_loader.rel2id):,}")
    print(f"数据加载完成 | 耗时: {time.time()-start_time:.1f}秒\n")

    # 模型初始化
    print(f"初始化 {args.model} 模型...")
    start_time = time.time()
    num_entities = len(data_loader.entity2id)
    num_relations = len(data_loader.rel2id)
    
    if args.model == 'TransE':
        model = TransE(num_entities, num_relations, args)
    elif args.model == 'RotatE':
        model = RotatE(num_entities, num_relations, args)
    elif args.model == 'ConvE':
        model = ConvE(num_entities, num_relations, args)
    
    model = model.to(device)
    print(f"模型初始化完成 | 耗时: {time.time()-start_time:.1f}秒")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练器初始化
    print("\n初始化训练器...")
    trainer = Trainer(
        model=model,
        args=args,
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        all_triples=all_triples
    )
    
    # 设置评估器的数据加载器信息
    trainer.evaluator.entity2id = data_loader.entity2id
    trainer.evaluator.rel2id = data_loader.rel2id
    trainer.evaluator.num_entities = num_entities
    trainer.evaluator.num_relations = num_relations
    
    # 开始训练
    results = trainer.train()
    
    # 保存结果
    save_path = save_results(results, args.model, args.dataset)
    print(f"训练结果保存至: {save_path}")
    
    # 保存训练曲线
    curve_path = f"../results/{args.model}_{args.dataset}_training_curve.png"
    trainer.plot_training_curve(curve_path)
    
    print("\n训练完成!")

if __name__ == '__main__':
    main()