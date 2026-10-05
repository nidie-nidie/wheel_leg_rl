"""
train_impact_cluster_vis.py
基于正常状态聚类的台阶撞击检测 —— 仅使用前6列特征，最后1列为标签
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# ================== 参数配置 ==================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
CSV_FILES = [
    os.path.join(desktop, "512_two.csv"),
    os.path.join(desktop, "511err.csv")
]
WINDOW_SIZE = 8
N_CLUSTERS = 7

# 修改：只保留前6个有效特征名，移除后面无意义的占位
FEATURE_NAMES = [
    "v_body", "v_wheel_avg", "diff_body_wheel",
    "diff_LR"#, "tau_L", "tau_R"
]
N_FEATURES = len(FEATURE_NAMES)      # 现在为 6
VEC_LEN = WINDOW_SIZE * N_FEATURES   # 展开后维度

# ================== 数据加载 ==================
def load_data(path):
    # skiprows=1 跳过标题行
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    # 修改：CSV可能有11列（10特征+1标签），但只取前N_FEATURES列作为特征
    if data.shape[1] < N_FEATURES + 1:
        raise ValueError(f"{path} 每行至少需要 {N_FEATURES} 个特征 + 1 个标签")
    # 前 N_FEATURES 列是特征，最后一列是标签（索引 -1）
    features = data[:, :N_FEATURES]  # 只取前6列
    labels = data[:, -1]             # 最后一列作为标签
    return features, labels

# 合并多个文件
all_features, all_labels = [], []
for fpath in CSV_FILES:
    print(f"加载数据: {fpath}")
    feats, labs = load_data(fpath)
    print(f"  帧数: {len(feats)}, 撞击帧: {np.sum(labs)}")
    all_features.append(feats)
    all_labels.append(labs)

features = np.vstack(all_features)
labels = np.hstack(all_labels)
print(f"合并后总帧数: {len(features)}, 撞击帧: {np.sum(labels)}")

# ================== 滑窗生成样本 (只保留正常样本) ==================
def make_samples(features, labels, window):
    X_norm = []
    for i in range(window - 1, len(features)):
        if labels[i] == 0:   # 只取正常行走窗口
            sample = features[i - window + 1 : i + 1, :].flatten()
            X_norm.append(sample)
    return np.array(X_norm)

X_train = make_samples(features, labels, WINDOW_SIZE)
print(f"正常样本数: {X_train.shape[0]}")

# ================== 标准化 ==================
mean = np.mean(X_train, axis=0)
std  = np.std(X_train, axis=0)
std[std == 0] = 1e-6
X_norm = (X_train - mean) / std

# ================== K-Means 聚类 ==================
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X_norm)
centroids = kmeans.cluster_centers_

# 计算距离及阈值
distances = np.min(np.sum((X_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1)
threshold = np.percentile(distances, 95)
max_dist = np.max(distances)
print(f"正常样本到最近中心距离：max={max_dist:.3f}, 95%分位={threshold:.3f}")

# ================== 可视化 ==================
# PCA 降至 2 维用于绘图
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_norm)
centroids_2d = pca.transform(centroids)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：样本分布与聚类中心
ax1.scatter(X_2d[:, 0], X_2d[:, 1], c='lightblue', alpha=0.5, s=10, label='Normal samples')
ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=100, edgecolors='k', label='Centroids')
ax1.set_title('Normal Samples and Cluster Centers (PCA)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：距离分布直方图与阈值
ax2.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (95%): {threshold:.2f}')
ax2.set_title('Distance to Nearest Cluster Center')
ax2.set_xlabel('Squared Euclidean Distance')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('impact_cluster_visualization.png', dpi=150)
plt.show()
print("可视化图片已保存为 impact_cluster_visualization.png")

# ================== 导出 C 头文件 ==================
lines = []
lines.append("// Step Impact Detector based on Normal-State Clustering")
lines.append(f"// Window: {WINDOW_SIZE}, Features: " + ", ".join(FEATURE_NAMES))
lines.append(f"// Clusters: {N_CLUSTERS}, Threshold: {threshold:.3f}")
lines.append("")
lines.append("#ifndef KNN_IMPACT_MODEL_H")
lines.append("#define KNN_IMPACT_MODEL_H")
lines.append("")
lines.append(f"#define IMPACT_WINDOW_SIZE {WINDOW_SIZE}")
lines.append(f"#define IMPACT_N_FEATURES {N_FEATURES}")
lines.append(f"#define IMPACT_N_CLUSTERS {N_CLUSTERS}")
lines.append(f"#define IMPACT_VEC_LEN (IMPACT_WINDOW_SIZE * IMPACT_N_FEATURES)")
lines.append("")
lines.append("// Normalization parameters")
lines.append("static const float impact_norm_mean[IMPACT_VEC_LEN] = {")
vals = ", ".join(f"{v:.6f}f" for v in mean)
lines.append(f"    {vals}")
lines.append("};")
lines.append("static const float impact_norm_std[IMPACT_VEC_LEN] = {")
vals = ", ".join(f"{v:.6f}f" for v in std)
lines.append(f"    {vals}")
lines.append("};")
lines.append("")
lines.append("// Cluster centres (normal space)")
lines.append(f"static const float impact_centroids[IMPACT_N_CLUSTERS][IMPACT_VEC_LEN] = {{")
for i, c in enumerate(centroids):
    line = "    {" + ", ".join(f"{v:.6f}f" for v in c) + "}"
    if i < N_CLUSTERS - 1:
        line += ","
    lines.append(line)
lines.append("};")
lines.append("")
lines.append("// Distance threshold for anomaly (squared)")
lines.append(f"#define IMPACT_DIST_THRESHOLD {threshold:.6f}f")
lines.append("")
lines.append("#endif // KNN_IMPACT_MODEL_H")

with open("knn_impact_model.h", "w") as f:
    f.write("\n".join(lines))
print("模型头文件已生成: knn_impact_model.h")