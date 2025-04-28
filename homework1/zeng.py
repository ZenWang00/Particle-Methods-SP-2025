import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import pickle

# --------------------- 1. 读取 CSV 数据 ---------------------
# 请修改下面的文件路径为实际路径
data_file = '/Users/zitian/Desktop/zeng.csv'
df = pd.read_csv(data_file, header=0)

# 查看数据结构和前5行数据
print("数据结构：")
print(df.info())
print("前5行数据：")
print(df.head())

# --------------------- 2. 数据预处理 ---------------------
# 删除第一列（数字序号）
df = df.iloc[:, 1:]

# 检查是否存在 "class" 列
if 'class' not in df.columns:
    raise ValueError("数据中没有找到 'class' 列，请检查数据结构！")
else:
    # 将 "class" 列转换为类别变量
    df['class'] = df['class'].astype('category')

# 分离特征和目标变量
X = df.drop(columns=['class'])
y = df['class']

print("\n特征矩阵形状:", X.shape)
print("目标变量形状:", y.shape)

# --------------------- 3. 尝试 k 从 1 到 11 的全部情况 ---------------------
k_values = list(range(1, 12))
mean_accuracies = []
mean_errors = []

# 使用 10 折交叉验证
cv = KFold(n_splits=10, shuffle=True, random_state=42)

print("\n不同 k 值下的交叉验证结果：")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 计算交叉验证下的准确率
    scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
    mean_acc = np.mean(scores)
    mean_accuracies.append(mean_acc)
    mean_errors.append(1 - mean_acc)
    print(f"k = {k}, Accuracy = {mean_acc:.4f}, Misclassification Error = {1-mean_acc:.4f}")

# 选择最佳 k 值（最高准确率）
best_index = np.argmax(mean_accuracies)
best_k = k_values[best_index]
best_accuracy = mean_accuracies[best_index]
print(f"\n最佳 k 值为: {best_k}，对应的平均准确率为: {best_accuracy:.4f}")

# --------------------- 4. 绘制图表展示结果 ---------------------
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_accuracies, marker='o', label='Accuracy')
plt.plot(k_values, mean_errors, marker='x', label='Misclassification Error')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN model performance in different k value')
plt.legend()
plt.grid(True)
plt.show()

# --------------------- 5. 训练最终模型 ---------------------
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X, y)

# --------------------- 6. 保存最终模型 ---------------------
model_file = 'finalKnnModelmuli.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n最终模型已保存至: {model_file}")
