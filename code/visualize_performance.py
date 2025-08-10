import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_model_results(results_dir):
    """加载所有模型的评估结果"""
    results = {}
    
    # 查找目录中所有JSON结果文件
    result_files = glob(os.path.join(results_dir, "*.json"))
    
    for file in result_files:
        # 从文件名提取模型名称和数据集
        filename = os.path.basename(file)
        parts = filename.split("_")
        if len(parts) >= 2:
            model_name = parts[0]
            dataset = parts[1]
            key = f"{model_name} ({dataset})"
        else:
            key = filename.split(".")[0]  #  fallback
        
        # 读取JSON文件
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取评估指标
        if 'metrics' in data:
            results[key] = {
                'MRR': data['metrics'].get('MRR', 0),
                'Hit@1': data['metrics'].get('HITS@1', 0),
                'Hit@3': data['metrics'].get('HITS@3', 0),
                'Hit@10': data['metrics'].get('HITS@10', 0)
            }
    
    return results

def plot_performance_comparison(results, output_file, title="comparison"):
    """生成模型性能比较条形图"""
    # 设置中文字体
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    # 准备数据
    models = list(results.keys())
    metrics = ['MRR', 'Hit@1', 'Hit@3', 'Hit@10']
    
    # 设置图形大小
    x = np.arange(len(models))  # 模型位置
    width = 0.2  # 条形宽度
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每个指标创建一组条形
    rects = []
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        rect = ax.bar(x + i*width, values, width, label=metric)
        rects.append(rect)
    
    # 添加标签、标题和自定义x轴刻度
    ax.set_ylabel('Metrics')
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    
    # 在条形上方添加数值标签
    def autolabel(rects):
        """为每个条形添加数值标签"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0, fontsize=8)
    
    for rect in rects:
        autolabel(rect)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"性能比较图已保存至: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="模型性能比较可视化工具")
    parser.add_argument('--results_dir', type=str, default='../results', 
                      help='存放评估结果JSON文件的目录')
    parser.add_argument('--output', type=str, default='../results/model_comparison.png', 
                      help='输出图像文件路径')
    parser.add_argument('--title', type=str, default='不同模型在知识图谱补全任务上的性能比较', 
                      help='图表标题')
    args = parser.parse_args()
    
    # 加载结果
    print(f"从 {args.results_dir} 加载模型评估结果...")
    results = load_model_results(args.results_dir)
    
    if not results:
        print("未找到任何评估结果文件，请检查目录是否正确")
        return
    
    print(f"找到 {len(results)} 个模型的评估结果")
    
    # 生成可视化图表
    plot_performance_comparison(results, args.output, args.title)

if __name__ == '__main__':
    main()
    