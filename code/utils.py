import json
import os
from datetime import datetime

def save_results(results, model_name, dataset):
    """保存性能数据到JSON文件"""
    if not os.path.exists('../results'):
        os.makedirs('../results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'../results/{model_name}_{dataset}_{timestamp}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    return path