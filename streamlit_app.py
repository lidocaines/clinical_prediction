# clinical_prediction_app.py
import numpy as np
import joblib
import streamlit as st
import shap
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, confusion_matrix
# ==================== 全局配置 ====================
model_name = "LDA"
NUM_FEATURES = 7
# 新相对路径（相对于 streamlit_app.py）
DATA_PATH = "./data/mimic_knn.csv"
importance_file = "./results_step2_importance/LDA/feature_importance_LDA.csv"
MODEL_DIR = "./models/LDA_7_features/"  # 新增模型目录变量
RESULTS_DIR = f"./clinical_prediction_app"
os.makedirs(RESULTS_DIR, exist_ok=True)  # 确保结果目录存在

# 特征配置（根据实际特征调整）
FEATURE_CONFIG = {
    'age': {'min': 18.0, 'max': 88.0, 'step': 7.0, 'format': '%.0f', 'required': False},
    'weight': {'min': 30.0, 'max': 150.0, 'step': 10.0, 'format': '%.0f', 'required': False},
    'sofa': {'min': 0.0, 'max': 24.0, 'step': 2.0, 'format': '%.0f', 'required': False},
    'creatinine_max': {'min': 0.0, 'max': 900.0, 'step': 100.0, 'format': '%.1f', 'required': False},
    'lactate_max': {'min': 0.0, 'max': 20.0, 'step': 2.0, 'format': '%.1f', 'required': False},
    'urineoutput': {'min': 0, 'max': 4000, 'step': 500, 'format': '%.0f', 'required': False},
    'temperature_min': {'min': 34.0, 'max': 42.0, 'step': 1.0, 'format': '%.1f', 'required': False},
    'rdw_max': {'min': 10.0, 'max': 45.0, 'step': 5.0, 'format': '%.0f', 'required': False},
    'spo2_min': {'min': 50.0, 'max': 100.0, 'step': 10.0, 'format': '%.2f', 'required': False},
    'aki_stage': {'min': 0.0, 'max': 3.0, 'step': 1.0, 'format': '%.0f', 'required': False},
}


# ==================== 模型训练和保存 ====================
def train_and_save_model():
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df[df['aki_stage'] > 0]

    # 加载特征
    feature_df = pd.read_csv(importance_file)
    features = feature_df['feature'].head(NUM_FEATURES).tolist()

    # 数据准备
    X = df[features]
    y = df['crrt']

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 训练模型
    model = LDA()
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


# ==================== 动态生成输入组件 ====================
def generate_inputs(features):
    """根据特征动态生成输入表单"""
    input_values = {}

    # 核心参数
    with st.container():
        st.header("核心参数")
        cols = st.columns(2)
        main_features = [f for f in features if FEATURE_CONFIG[f]['required']]

        for i, feat in enumerate(main_features):
            with cols[i % 2]:
                cfg = FEATURE_CONFIG[feat]
                # 统一数值类型
                min_val = cfg['min']
                max_val = cfg['max']
                step_val = cfg['step']
                default_val = (min_val + max_val) / 2

                # 根据格式判断数值类型
                if '%.0f' in cfg['format']:  # 整数类型
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(default_val),
                        step=int(step_val),
                        format=cfg['format']
                    )
                else:  # 浮点数类型
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        step=float(step_val),
                        format=cfg['format']
                    )

        # 高级参数 (修改方式同上)
        with st.expander("高级参数"):
            adv_features = [f for f in features if not FEATURE_CONFIG[f]['required']]
            cols = st.columns(2)

            for i, feat in enumerate(adv_features):
                with cols[i % 2]:
                    cfg = FEATURE_CONFIG[feat]
                    min_val = cfg['min']
                    max_val = cfg['max']
                    step_val = cfg['step']
                    default_val = (min_val + max_val) / 2

                    if '%.0f' in cfg['format']:
                        input_values[feat] = st.number_input(
                            label=feat,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(default_val),
                            step=int(step_val),
                            format=cfg['format']
                        )
                    else:
                        input_values[feat] = st.number_input(
                            label=feat,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default_val),
                            step=float(step_val),
                            format=cfg['format']
                        )

        return input_values

# ==================== Streamlit界面 ====================
def main():
    st.set_page_config(page_title="临床预测系统", layout="wide")

    # 加载路径配置
    model_path = os.path.join(RESULTS_DIR, 'clinical_model.pkl')
    scaler_path = os.path.join(RESULTS_DIR, 'scaler.pkl')
    features_path = os.path.join(RESULTS_DIR, 'features.csv')

    try:
        # 加载模型资源
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = pd.read_csv(features_path)['feature'].tolist()
        X_train = joblib.load(os.path.join(RESULTS_DIR, 'x_train.pkl'))  # 新增：加载训练数据

        # 验证特征配置
        missing_features = [f for f in features if f not in FEATURE_CONFIG]
        if missing_features:
            st.error(f"缺失特征配置：{', '.join(missing_features)}")
            st.stop()

    except FileNotFoundError as e:
        st.warning(f"模型文件未找到({e})，正在训练新模型...")
        model, scaler, features = train_and_save_model()

    # 界面标题
    st.title("CRRT治疗需求预测系统")
    st.markdown("---")

    # 生成输入表单
    input_values = generate_inputs(features)

    # 预测按钮
    if st.button("开始预测", use_container_width=True):
        try:
            # 准备输入数据
            input_data = pd.DataFrame([[input_values[f] for f in features]], columns=features)

            # 标准化
            scaled_data = scaler.transform(input_data)

            # 预测概率
            proba = model.predict_proba(scaled_data)[0][1]
            prediction = 1 if proba > 0.5 else 0

            # 显示结果
            st.markdown("---")
            result_container = st.container()
            with result_container:
                st.subheader("预测结果")
                if prediction == 1:
                    st.error(f"高风险 (概率: {proba * 100:.1f}%)")
                    st.markdown("""
                    **临床建议：**
                    - 立即联系重症监护团队
                    - 评估血流动力学状态
                    - 准备CRRT治疗设备
                    """)
                else:
                    st.success(f"低风险 (概率: {proba * 100:.1f}%)")
                    st.markdown("""
                    **临床建议：**
                    - 持续监测生命体征
                    - 每4小时评估肾功能
                    - 维持液体平衡
                    """)

                    # SHAP解释
                    with st.container():
                        st.subheader("预测解释")

                        # 初始化解释器
                        # 在SHAP解释部分可以改为：
                        try:
                            explainer = shap.LinearExplainer(model, X_train,
                            feature_perturbation="interventional")
                        except:
                            explainer = shap.KernelExplainer(model.predict_proba, X_train,
                            feature_perturbation="interventional")

                        # 计算SHAP值
                        shap_values = explainer.shap_values(scaled_data)

                        # 瀑布图
                        st.markdown("#### 瀑布图解释")
                        fig1, ax1 = plt.subplots()
                        shap.plots._waterfall.waterfall_legacy(
                            explainer.expected_value,
                            shap_values[0],
                            feature_names=features,
                            max_display=10,
                            show=False
                        )
                        st.pyplot(fig1)

                        # 力图
                        st.markdown("#### 特征贡献力图")
                        fig2, ax2 = plt.subplots()
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            input_data.iloc[0],
                            matplotlib=True,
                            show=False,
                            text_rotation=15
                        )
                        st.pyplot(fig2)

                        # 决策图
                        st.markdown("#### 决策过程图")
                        fig3, ax3 = plt.subplots()
                        shap.decision_plot(
                            explainer.expected_value,
                            shap_values,
                            features,
                            feature_order='importance',
                            show=False
                        )
                        st.pyplot(fig3)
        except Exception as e:
            st.error(f"预测失败：{str(e)}")


# ==================== 执行程序 ====================
if __name__ == "__main__":
    # 自动检查并训练模型
    if not os.path.exists(os.path.join(RESULTS_DIR, 'clinical_model.pkl')):
        train_and_save_model()

    # 运行应用
    main()