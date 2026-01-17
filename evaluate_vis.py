"""
Evaluation script for UMSF-Net with comprehensive visualization
包含混淆矩阵、分类报告、样本可视化等功能
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

from models import build_moco_model
from datasets import WHUOptSARDataset, get_val_transforms
from utils.metrics import ClusteringMetrics


def setup_chinese_fonts():
    """设置中文字体"""
    font_candidates = [
        'SimHei', 'WenQuanYi Zen Hei', 'Heiti TC',
        'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans'
    ]
    
    for font in font_candidates:
        try:
            plt.rcParams["font.family"] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
        except:
            continue
    
    print("警告：未找到中文字体，使用默认字体")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate UMSF-Net')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=str, default=None,
                        help='Specific sample indices to visualize (comma-separated)')
    
    return parser.parse_args()


# 类别名称和颜色映射
CLASS_NAMES = ['农田', '城市', '村庄', '水体', '森林', '道路', '其他']
CLASS_NAMES_EN = ['Farmland', 'City', 'Village', 'Water', 'Forest', 'Road', 'Others']

COLOR_MAP = {
    0: [0, 128, 0],      # 农田 - 绿色
    1: [255, 0, 0],      # 城市 - 红色
    2: [255, 165, 0],    # 村庄 - 橙色
    3: [0, 0, 255],      # 水体 - 蓝色
    4: [0, 100, 0],      # 森林 - 深绿
    5: [255, 255, 255],  # 道路 - 白色
    6: [128, 128, 128]   # 其他 - 灰色
}


def create_color_mask(mask, num_classes=7):
    """将标签mask转换为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id in range(num_classes):
        color_mask[mask == cls_id] = COLOR_MAP.get(cls_id, [0, 0, 0])
    
    return color_mask


def plot_confusion_matrix(cm, classes, save_path, normalize=False):
    """绘制并保存混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def visualize_sample(optical, sar, label, pred, sample_idx, save_dir, version='v1'):
    """可视化单个样本的结果"""
    sample_dir = os.path.join(save_dir, f'sample_{sample_idx:04d}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. 保存光学图像
    opt_img = optical.cpu().numpy()
    if opt_img.ndim == 3:
        opt_img = opt_img.transpose(1, 2, 0)  # CHW -> HWC
    if opt_img.shape[-1] > 3:
        opt_img = opt_img[..., :3]  # 只取RGB通道
    opt_img = ((opt_img - opt_img.min()) / (opt_img.max() - opt_img.min() + 1e-8) * 255).astype(np.uint8)
    opt_path = os.path.join(sample_dir, f'optical_{sample_idx:04d}_{version}.png')
    cv2.imwrite(opt_path, cv2.cvtColor(opt_img, cv2.COLOR_RGB2BGR))
    
    # 2. 保存SAR图像
    sar_img = sar.cpu().numpy()
    if sar_img.ndim == 3:
        sar_img = sar_img[0]  # 取第一个通道
    sar_img = ((sar_img - sar_img.min()) / (sar_img.max() - sar_img.min() + 1e-8) * 255).astype(np.uint8)
    sar_path = os.path.join(sample_dir, f'sar_{sample_idx:04d}_{version}.png')
    cv2.imwrite(sar_path, sar_img)
    
    # 3. 保存真实标签
    label_mask = label.cpu().numpy()
    label_color = create_color_mask(label_mask)
    label_path = os.path.join(sample_dir, f'label_{sample_idx:04d}_{version}.png')
    cv2.imwrite(label_path, cv2.cvtColor(label_color, cv2.COLOR_RGB2BGR))
    
    # 4. 保存预测结果
    pred_mask = pred.cpu().numpy()
    pred_color = create_color_mask(pred_mask)
    pred_path = os.path.join(sample_dir, f'pred_{sample_idx:04d}_{version}.png')
    cv2.imwrite(pred_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    
    # 5. 保存对比图（四合一）
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(opt_img)
    axes[0, 0].set_title('Optical Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sar_img, cmap='gray')
    axes[0, 1].set_title('SAR Image', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(label_color)
    axes[1, 0].set_title('Ground Truth', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_color)
    axes[1, 1].set_title('Prediction', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Sample {sample_idx}', fontsize=16)
    plt.tight_layout()
    
    compare_path = os.path.join(sample_dir, f'comparison_{sample_idx:04d}_{version}.png')
    plt.savefig(compare_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  样本 {sample_idx} 已保存到: {sample_dir}")


def plot_class_legend(save_path, class_names, color_map):
    """绘制类别图例"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for i, (name, color) in enumerate(zip(class_names, [color_map[j] for j in range(len(class_names))])):
        color_normalized = [c/255 for c in color]
        ax.barh(i, 1, color=color_normalized, edgecolor='black')
        ax.text(1.1, i, name, va='center', fontsize=12)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Class Legend', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"类别图例已保存: {save_path}")


def evaluate(args):
    """主评估函数"""
    setup_chinese_fonts()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'eval_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    print(f"结果将保存到: {output_dir}")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_classes = config['model']['num_classes']
    
    # 构建模型
    print("加载模型...")
    model = build_moco_model(config['model'])
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 尝试直接加载
        try:
            model.load_state_dict(checkpoint)
        except:
            # 尝试加载到encoder_q
            model.encoder_q.load_state_dict(checkpoint, strict=False)
    
    model = model.cuda()
    model.eval()
    print("✓ 模型加载成功")
    
    # 构建数据加载器
    _, opt_transform, sar_transform = get_val_transforms(
        size=config['data'].get('image_size', 256)
    )
    
    test_dataset = WHUOptSARDataset(
        data_root=args.data_root,
        split='test',
        optical_transform=opt_transform,
        sar_transform=sar_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 确定要可视化的样本索引
    if args.sample_indices:
        vis_indices = [int(i) for i in args.sample_indices.split(',')]
    else:
        # 均匀选择样本
        total = len(test_dataset)
        step = max(1, total // args.num_samples)
        vis_indices = list(range(0, total, step))[:args.num_samples]
    
    print(f"将可视化样本索引: {vis_indices}")
    
    # 评估
    all_preds = []
    all_labels = []
    all_features = []
    samples_to_visualize = {}
    
    current_idx = 0
    
    print("\n开始评估...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估进度'):
            optical = batch['optical'].cuda()
            sar = batch['sar'].cuda()
            
            # 前向传播
            z, p, features = model.encoder_q(optical, sar, return_features=True)
            preds = p.argmax(dim=1)
            
            batch_size = optical.size(0)
            
            # 收集要可视化的样本
            for i in range(batch_size):
                global_idx = current_idx + i
                if global_idx in vis_indices:
                    samples_to_visualize[global_idx] = {
                        'optical': optical[i],
                        'sar': sar[i],
                        'pred': preds[i]
                    }
            
            current_idx += batch_size
            
            # 收集预测和特征
            all_preds.append(preds.cpu())
            all_features.append(features['fused'].cpu())
            
            if 'label' in batch:
                all_labels.append(batch['label'])
                
                # 为可视化样本添加标签
                for i in range(batch_size):
                    global_idx = current_idx - batch_size + i
                    if global_idx in samples_to_visualize:
                        samples_to_visualize[global_idx]['label'] = batch['label'][i]
    
    # 合并结果
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_features = torch.cat(all_features, dim=0).numpy()
    
    has_labels = len(all_labels) > 0
    if has_labels:
        all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # 计算指标
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    
    if has_labels:
        # 展平用于像素级评估
        labels_flat = all_labels.flatten()
        preds_flat = all_preds.flatten()
        
        # 计算准确率
        accuracy = (labels_flat == preds_flat).sum() / len(labels_flat) * 100
        print(f"总体准确率 (OA): {accuracy:.2f}%")
        
        # 聚类指标
        metrics = ClusteringMetrics()
        acc = metrics.clustering_accuracy(labels_flat, preds_flat)
        nmi = metrics.nmi(labels_flat, preds_flat)
        ari = metrics.ari(labels_flat, preds_flat)
        
        print(f"聚类准确率 (ACC): {acc:.4f}")
        print(f"标准化互信息 (NMI): {nmi:.4f}")
        print(f"调整兰德指数 (ARI): {ari:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(labels_flat, preds_flat, labels=range(num_classes))
        plot_confusion_matrix(
            cm, CLASS_NAMES_EN[:num_classes],
            output_dir / 'confusion_matrix.png'
        )
        
        # 归一化混淆矩阵
        plot_confusion_matrix(
            cm, CLASS_NAMES_EN[:num_classes],
            output_dir / 'confusion_matrix_normalized.png',
            normalize=True
        )
        
        # 分类报告
        report = classification_report(
            labels_flat, preds_flat,
            target_names=CLASS_NAMES_EN[:num_classes],
            digits=4,
            zero_division=0
        )
        print("\n分类报告:")
        print(report)
        
        # 保存分类报告
        report_path = output_dir / 'classification_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"UMSF-Net Evaluation Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Overall Accuracy (OA): {accuracy:.2f}%\n")
            f.write(f"Clustering Accuracy (ACC): {acc:.4f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n")
            f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n\n")
            f.write(f"Classification Report:\n")
            f.write(report)
        print(f"分类报告已保存: {report_path}")
    
    # 可视化样本
    print(f"\n生成 {len(samples_to_visualize)} 个样本的可视化结果...")
    for idx in sorted(samples_to_visualize.keys()):
        sample = samples_to_visualize[idx]
        if 'label' in sample:
            visualize_sample(
                sample['optical'],
                sample['sar'],
                sample['label'],
                sample['pred'],
                idx,
                str(vis_dir),
                version='umsf'
            )
    
    # 保存类别图例
    plot_class_legend(
        output_dir / 'class_legend.png',
        CLASS_NAMES_EN[:num_classes],
        COLOR_MAP
    )
    
    # t-SNE可视化（可选）
    try:
        from sklearn.manifold import TSNE
        
        print("\n生成 t-SNE 可视化...")
        # 采样部分特征进行可视化
        n_samples = min(5000, len(all_features))
        indices = np.random.choice(len(all_features), n_samples, replace=False)
        features_sample = all_features.reshape(all_features.shape[0], -1)[indices]
        
        if has_labels:
            labels_sample = all_labels.flatten()[indices]
        else:
            labels_sample = all_preds.flatten()[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_sample[:, :256])  # 只用前256维
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels_sample, cmap='tab10', alpha=0.6, s=5
        )
        plt.colorbar(scatter)
        plt.title('t-SNE Feature Visualization', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        
        tsne_path = output_dir / 'tsne_visualization.png'
        plt.savefig(tsne_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"t-SNE可视化已保存: {tsne_path}")
        
    except Exception as e:
        print(f"t-SNE可视化失败: {e}")
    
    print(f"\n{'='*50}")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*50}")
    
    # 打印文件列表
    print("\n生成的文件:")
    for f in sorted(output_dir.glob('*')):
        if f.is_file():
            print(f"  - {f.name}")
    print(f"  - visualizations/ ({len(samples_to_visualize)} 个样本)")


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
