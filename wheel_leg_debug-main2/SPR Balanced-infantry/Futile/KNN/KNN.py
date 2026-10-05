import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==================== 配置区 ====================
CSV_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "3.csv")  # 你的二分类数据
OUTPUT_H = "knn_model.h"           # 输出头文件
N_CLUSTERS_PER_CLASS = 20          # 每类保留的聚类中心数（总共 2*N 个）
K = 5                              # KNN 近邻数（仅用于头文件信息）
EVALUATE = True                    # 是否在训练集上测试准确率
VISUALIZE = True                   # 是否显示聚类散点图
# ================================================

def main():
    # 1. 读取数据
    print(f"读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if df.shape[1] != 11:
        raise ValueError("数据列数应为 11（10 特征 + 1 标签）")
    features = df.iloc[:, :10].values.astype(np.float32)
    labels = df.iloc[:, 10].values.astype(np.int32)
    print(f"样本总数: {len(labels)}")
    print(f"标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

    # 2. 全局标准化（保留参数，导出给单片机）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    mean_vec = scaler.mean_.astype(np.float32)
    std_vec  = scaler.scale_.astype(np.float32)

    # 3. 分别对两类聚类
    centers_list, labels_list = [], []
    for cls in [0, 1]:
        mask = labels == cls
        cls_data = features_scaled[mask]
        n_clusters = min(N_CLUSTERS_PER_CLASS, cls_data.shape[0])
        if n_clusters == 0:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(cls_data)
        centers_list.append(kmeans.cluster_centers_)
        labels_list.append(np.full(n_clusters, cls, dtype=np.int32))
        print(f"类别 {cls}: {cls_data.shape[0]} 样本 -> {n_clusters} 个中心")

    all_centers = np.vstack(centers_list).astype(np.float32)
    all_labels = np.concatenate(labels_list).astype(np.int32)
    print(f"总聚类中心数: {len(all_labels)}")

    # 4. （可选）评估准确率
    if EVALUATE:
        # 用原始数据（标准化后的）测试 KNN
        correct = 0
        for i in range(len(features_scaled)):
            sample = features_scaled[i]
            # 计算到所有中心的平方距离
            dists = np.sum((all_centers - sample) ** 2, axis=1)
            # 找到最近的 K 个中心的索引
            k_idx = np.argpartition(dists, K)[:K]
            k_labs = all_labels[k_idx]
            # 投票：正样本比例 >= 0.5 则为 1
            pred = 1 if np.mean(k_labs) >= 0.5 else 0
            if pred == labels[i]:
                correct += 1
        acc = correct / len(features_scaled)
        print(f"训练集自评准确率 (K={K}): {acc:.4f}")

    # 5. 可视化
    if VISUALIZE:
        # 使用 PCA 降到 2D
        pca = PCA(n_components=2, random_state=42)
        all_2d = pca.fit_transform(features_scaled)
        centers_2d = pca.transform(all_centers)

        plt.figure(figsize=(10, 6))
        # 画原始样本（按标签颜色区分）
        for cls, color, label in [(0, 'blue', 'Not Liftoff'), (1, 'red', 'Liftoff')]:
            mask = labels == cls
            plt.scatter(all_2d[mask, 0], all_2d[mask, 1],
                        c=color, alpha=0.3, s=5, label=label)
        # 画聚类中心（用黑色叉号）
        for cls, marker, color in [(0, 'x', 'darkblue'), (1, '+', 'darkred')]:
            mask_center = all_labels == cls
            plt.scatter(centers_2d[mask_center, 0], centers_2d[mask_center, 1],
                        marker=marker, c=color, s=100,
                        label=f'Center (class {cls})')
        plt.title(f'Binary KNN Clusters ({len(all_labels)} centers)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 6. 导出头文件
    N = len(all_labels)
    with open(OUTPUT_H, 'w') as f:
        f.write('// 二分类 KNN 模型（基于聚类中心）\n')
        f.write(f'#define KNN_SAMPLES {N}\n')
        f.write(f'#define KNN_FEATURES 10\n')
        f.write(f'#define KNN_K {K}\n\n')

        f.write('// 特征均值（标准化用）\n')
        f.write('const float knn_mean[KNN_FEATURES] = {')
        f.write(', '.join(f'{v:.6f}f' for v in mean_vec))
        f.write('};\n\n')

        f.write('// 特征标准差\n')
        f.write('const float knn_std[KNN_FEATURES] = {')
        f.write(', '.join(f'{v:.6f}f' for v in std_vec))
        f.write('};\n\n')

        f.write('// 训练数据：每行 [标准化特征(10), 标签(0/1)]\n')
        f.write(f'const float knn_train_data[KNN_SAMPLES][KNN_FEATURES+1] = {{\n')
        for i in range(N):
            row = np.append(all_centers[i], all_labels[i]).astype(np.float32)
            vals = ', '.join(f'{v:.6f}f' for v in row)
            line = f'    {{{vals}}}'
            if i < N - 1:
                line += ','
            line += '\n'
            f.write(line)
        f.write('};\n')
    print(f"头文件已生成: {OUTPUT_H}")

if __name__ == '__main__':
    main()