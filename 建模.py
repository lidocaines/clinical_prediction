
import numpy as np
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # 新增配置
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

# ==================== 全局配置 ====================
model_name = "Gradient_Boosting"  # 切换模型名称
NUM_FEATURES = 9  # 特征数量

# 新相对路径（相对于 streamlit_app.py）
#本地运行网页
#RESULTS_DIR = f"./clinical_prediction_app/"
#DATA_PATH = "./clinical_prediction_app/mimic_knn.csv"
#importance_file = f"./clinical_prediction_app/results_step2_importance/{model_name}/feature_importance_{model_name}.csv"
#网页客户端
RESULTS_DIR = f"./"
DATA_PATH = "./mimic_knn.csv"
importance_file =f"./results_step2_importance/{model_name}/feature_importance_{model_name}.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)  # 确保结果目录存在


# 定义模型
models = {
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1),
    "Logistic_Regression": LogisticRegression(random_state=42, max_iter=10000),
    "Random_Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42),
    "LDA": LDA()
}

# ==================== 加载特征函数 ====================
def load_features():
    feature_df = pd.read_csv(importance_file)
    features = feature_df['feature'].head(NUM_FEATURES).tolist()
    return features


# ==================== 删除旧模型文件 ====================
def delete_old_model_files():
    files_to_delete = [
        os.path.join(RESULTS_DIR, 'clinical_model.pkl'),
        os.path.join(RESULTS_DIR, 'scaler.pkl'),
        os.path.join(RESULTS_DIR, 'features.csv'),
        os.path.join(RESULTS_DIR, 'x_train.pkl')
    ]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)


# ==================== 模型训练和保存 ====================
def train_and_save_model():
    # 删除旧模型文件
    delete_old_model_files()

    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df[df['aki_stage'] > 0]

    # 加载特征
    features = load_features()

    # 数据准备
    X = df[features]
    y = df['crrt']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 训练模型
    model = models[model_name]
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    # 计算训练集和测试集的预测值
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算全部数据的性能指标
    y_all_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_all = y  # 使用原始的y数据

    from sklearn.utils import resample
    # 定义一个函数来计算并输出各项指标
    def print_metrics(y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba >= threshold).astype(int)
        auc = roc_auc_score(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        # 使用自助法计算AUC的95%置信区间
        n_bootstraps = 1000
        rng_seed = 42  # 可复现结果
        bootstrapped_scores = []
        for i in range(n_bootstraps):
            indices = resample(np.arange(len(y_true)), replace=True, random_state=rng_seed + i)
            y_true_bootstrap = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
            y_pred_proba_bootstrap = y_pred_proba[indices]
            auc_bootstrap = roc_auc_score(y_true_bootstrap, y_pred_proba_bootstrap)
            bootstrapped_scores.append(auc_bootstrap)
        lower_ci, upper_ci = np.percentile(bootstrapped_scores, [2.5, 97.5])

        return auc, lower_ci, upper_ci, accuracy, f1, recall, precision

    # 训练集性能
    train_metrics = print_metrics(y_train, y_train_pred_proba)
    print(
        f"Training set metrics: AUC: {train_metrics[0]:.4f} (95% CI: {train_metrics[1]:.4f} - {train_metrics[2]:.4f}), "
        f"Accuracy: {train_metrics[3]:.4f}, F1-score: {train_metrics[4]:.4f}, "
        f"Recall: {train_metrics[5]:.4f}, Precision: {train_metrics[6]:.4f}")

    # 测试集性能
    test_metrics = print_metrics(y_test, y_test_pred_proba)
    print(
        f"Testing set metrics: AUC: {test_metrics[0]:.4f} (95% CI: {test_metrics[1]:.4f} - {test_metrics[2]:.4f}), "
        f"Accuracy: {test_metrics[3]:.4f}, F1-score: {test_metrics[4]:.4f}, "
        f"Recall: {test_metrics[5]:.4f}, Precision: {test_metrics[6]:.4f}")

    # 全部数据性能
    all_metrics = print_metrics(y_all, y_all_pred_proba)
    print(
        f"All data metrics: AUC: {all_metrics[0]:.4f} (95% CI: {all_metrics[1]:.4f} - {all_metrics[2]:.4f}), "
        f"Accuracy: {all_metrics[3]:.4f}, F1-score: {all_metrics[4]:.4f}, "
        f"Recall: {all_metrics[5]:.4f}, Precision: {all_metrics[6]:.4f}")
    # 保存路径配置
    model_path = os.path.join(RESULTS_DIR, 'clinical_model.pkl')
    scaler_path = os.path.join(RESULTS_DIR, 'scaler.pkl')
    features_path = os.path.join(RESULTS_DIR, 'features.csv')
    x_train_path = os.path.join(RESULTS_DIR, 'x_train.pkl')  # 新增训练数据保存路径

    # 保存资源
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X_train, x_train_path)  # 新增：保存训练数据
    pd.DataFrame({'feature': features}).to_csv(features_path, index=False)

    print(f"模型已保存至：{RESULTS_DIR}")
    return model, scaler, features


if __name__ == "__main__":
    train_and_save_model()