import shap
import numpy as np
import joblib
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
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
NUM_FEATURES = 8  # 特征数量



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

# 特征配置（根据实际特征调整）
FEATURE_CONFIG = {
    'age': {'min': 18.0, 'max': 88.0, 'step': 7.0, 'format': '%.0f', 'required': False},
    'weight': {'min': 30.0, 'max': 150.0, 'step': 10.0, 'format': '%.0f', 'required': False},
    'sofa': {'min': 0.0, 'max': 24.0, 'step': 2.0, 'format': '%.0f', 'required': False},
    'creatinine_max': {'min': 0.0, 'max': 100.0, 'step': 10.0, 'format': '%.1f', 'required': False},
    'lactate_max': {'min': 0.0, 'max': 20.0, 'step': 2.0, 'format': '%.1f', 'required': False},
    'urineoutput': {'min': 0, 'max': 4000, 'step': 500, 'format': '%.0f', 'required': False},
    'temperature_min': {'min': 34.0, 'max': 42.0, 'step': 1.0, 'format': '%.1f', 'required': False},
    'rdw_max': {'min': 10.0, 'max': 45.0, 'step': 5.0, 'format': '%.0f', 'required': False},
    'spo2_min': {'min': 50.0, 'max': 100.0, 'step': 10.0, 'format': '%.0f', 'required': False},
    'pao2fio2ratio_min': {'min': 20, 'max': 500, 'step': 10.0, 'format': '%.1f', 'required': False},
    'aki_stage': {'min': 0.0, 'max': 3.0, 'step': 1.0, 'format': '%.0f', 'required': False},
    'mbp_min': {'min': 40.0, 'max': 120.0, 'step': 5.0, 'format': '%.0f', 'required': False},
    'mchc_min': {'min': 20.0, 'max': 40.0, 'step': 0.5, 'format': '%.1f', 'required': False},
    'potassium_max': {'min': 2.0, 'max': 7.0, 'step': 0.5, 'format': '%.1f', 'required': False},
    'bmi': {'min': 10.0, 'max': 50.0, 'step': 1.0, 'format': '%.1f', 'required': False},
    'platelets_min': {'min': 0.0, 'max': 500.0, 'step': 50.0, 'format': '%.0f', 'required': False},
    'gender': {'min': 0, 'max': 1, 'step': 1, 'format': '%.0f', 'required': False},
    'height': {'min': 60.0, 'max': 240.0, 'step': 10.0, 'format': '%.0f', 'required': False},
    'po2_min': {'min': 20.0, 'max': 400.0, 'step': 20.0, 'format': '%.0f', 'required': False},
    'hemoglobin_min': {'min': 5.0, 'max': 20.0, 'step': 1.0, 'format': '%.1f', 'required': False},
    'ph_min': {'min': 6.8, 'max': 7.8, 'step': 0.1, 'format': '%.1f', 'required': False},
    'ph_max': {'min': 6.8, 'max': 7.8, 'step': 0.1, 'format': '%.1f', 'required': False},
}



# ==================== 动态生成输入组件 ====================
def generate_inputs(features):
    """根据特征动态生成输入表单"""
    input_values = {}

    # 核心参数
    with st.container():
        st.header("Key Parameters")
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

                # 在generate_inputs函数中统一修改
                if '%.0f' in cfg['format']:
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=float(cfg['min']),
                        max_value=float(cfg['max']),
                        value=float((cfg['min'] + cfg['max']) / 2),
                        step=float(cfg['step']),
                        format=cfg['format']
                    )
                else:
                    # 原有浮点数处理逻辑
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
    st.set_page_config(page_title="CRRT Need Prediction System", layout="wide")

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
        st.warning(f"模型文件未找到({e})请先运行建模脚本训练模型")

    # 界面标题
    st.title("CRRT Need Prediction System")         #CRRT治疗需求预测系统
    st.markdown("---")

    # 生成输入表单
    input_values = generate_inputs(features)

    # 预测按钮
    if st.button("Start Prediction", use_container_width=True):         #开始预测
        try:
            # 准备输入数据
            input_data = pd.DataFrame([[input_values[f] for f in features]], columns=features)
            scaled_data = scaler.transform(input_data)

            # 预测结果
            proba = model.predict_proba(scaled_data)[0][1]
            prediction = 1 if proba > 0.5 else 0

            # 显示预测结果
            # ... [保持原有结果显示逻辑不变] ...

            # SHAP可视化
            with st.container():
                st.subheader("Interpretation")  # shap解释

                try:
                    # 初始化解释器
                    if isinstance(model, (LDA, LogisticRegression)):
                        explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
                    elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier,
                                            xgb.XGBClassifier, lgb.LGBMClassifier)):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.KernelExplainer(model.predict_proba, X_train)

                    # 计算SHAP值
                    shap_values = explainer(scaled_data)

                    # 处理多分类输出
                    if isinstance(shap_values, list):
                        expected_value = explainer.expected_value[1]
                        shap_value = shap_values[1][0]
                    else:
                        expected_value = explainer.expected_value
                        shap_value = shap_values[0]

                    # 创建特征标签（特征名 + 原始值）
                    raw_values = [input_values[f] for f in features]
                    formatted_features = [
                        f"{name}\n({value:.1f})"
                        for name, value in zip(features, raw_values)
                    ]

                    # ========== 修改后的瀑布图部分 ==========
                    st.markdown("#### Waterfall Plot")          # 瀑布图解释
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # 确保 base_value 是标量
                    base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else \
                    explainer.expected_value[0]
                    # 创建Explanation对象
                    explanation = shap.Explanation(
                        values=shap_value.values,
                        base_values=base_value,
                        data=scaled_data[0],  # 使用标准化后的数据
                        feature_names=formatted_features
                    )

                    # 绘制瀑布图
                    shap.plots.waterfall(explanation, show=False)
                    plt.title(f"Probability:{proba * 100:.1f}%", fontsize=12)
                    plt.tight_layout()

                    # 显示图形
                    st.pyplot(fig)
                    plt.close(fig)

                    # ========== 决策图部分 ==========
                    st.markdown("Decision Plot")            # 决策过程图
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    shap.decision_plot(
                        expected_value,
                        shap_value.values,
                        formatted_features,
                        feature_order='importance',
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                except Exception as e:
                    st.error(f"SHAP解释生成失败：{str(e)}")

        except Exception as e:
            st.error(f"预测失败：{str(e)}")


# ==================== 执行程序 ====================
if __name__ == "__main__":
    main()