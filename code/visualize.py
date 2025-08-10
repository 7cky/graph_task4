import matplotlib.pyplot as plt
import json
import os

def plot_comparison(results_paths, metric='MRR'):
    """绘制不同模型的指标对比图"""
    models = []
    metrics = []
    for path in results_paths:
        with open(path, 'r') as f:
            res = json.load(f)
        models.append(res['model'])
        metrics.append(res[metric])
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, metrics)
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.savefig(f'../results/{metric}_comparison.png')
    plt.show()