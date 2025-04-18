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

# 新相对路径（相对于 streamlit_app.py）
#本地运行网页
#RESULTS_DIR = f"./clinical_prediction_app/"
#DATA_PATH = "./clinical_prediction_app/mimic_knn.csv"
#网页客户端
RESULTS_DIR = f"./"
DATA_PATH = "./mimic_knn.csv"


os.makedirs(RESULTS_DIR, exist_ok=True)  # 确保结果目录存在

# 特征配置（根据实际特征调整）
FEATURE_CONFIG = {
    'age':    {'min': 18, 'max': 88, 'step': 1, 'format': '%d', 'default': 50, 'required': False},  # 正常成年人年龄
    'sofa':   {'min': 0, 'max': 24, 'step': 1, 'format': '%d', 'default': 0, 'required': False},    # 正常SOFA评分0分
    'urineoutput': {'min': 0, 'max': 4000, 'step': 100, 'format': '%d', 'default': 1500, 'required': False},  # 正常尿量1500ml/天
    'aki_stage': {'min': 0, 'max': 3, 'step': 1, 'format': '%d', 'default': 0, 'required': False},  # 默认无AKI
    'gender': {'min': 0, 'max': 1, 'step': 1, 'format': '%d', 'default': 0, 'required': False},     # 0-女 1-男
    # 肾功能指标
    'creatinine_max': {'min': 0.0, 'max': 20.0, 'step': 0.1, 'format': '%.1f', 'default': 0.8, 'required': False},  # 正常肌酐(女0.5-1.1,男0.6-1.2)
    # 代谢指标
    'lactate_max': {'min': 0.0, 'max': 20.0, 'step': 0.5, 'format': '%.1f', 'default': 1.5, 'required': False},     # 正常乳酸0.5-2.2mmol/L
    # 生命体征
    'temperature_min': {'min': 34.0, 'max': 42.0, 'step': 0.1, 'format': '%.1f', 'default': 36.5, 'required': False}, # 正常体温36.5-37.2℃
    'mbp_min': {'min': 40.0, 'max': 120.0, 'step': 1.0, 'format': '%.1f', 'default': 85.0, 'required': False},      # 正常平均动脉压70-105mmHg
    'spo2_min': {'min': 50.0, 'max': 100.0, 'step': 1.0, 'format': '%.1f', 'default': 98.0, 'required': False},     # 正常血氧饱和度95-100%
    # 血气分析
    'ph_min': {'min': 6.00, 'max': 8.00, 'step': 0.1, 'format': '%.2f', 'default': 7.40, 'required': False},          # 正常pH7.35-7.45
    'po2_min': {'min': 20.0, 'max': 400.0, 'step': 10.0, 'format': '%.1f', 'default': 80.0, 'required': False},     # 正常PaO2 80-100mmHg
    # 血液指标
    'hemoglobin_min': {'min': 5.0, 'max': 20.0, 'step': 1.0, 'format': '%.1f', 'default': 12.0, 'required': False}, # 正常Hb(女12-16g/dL)
    'platelets_min': {'min': 0.0, 'max': 400.0, 'step': 10.0, 'format': '%.1f', 'default': 200.0, 'required': False}, # 正常血小板150-400×10^9/L
    # 其他指标
    'weight': {'min': 30.0, 'max': 150.0, 'step': 1.0, 'format': '%.1f', 'default': 65.0, 'required': False},       # 正常体重示例
    'height': {'min': 60.0, 'max': 240.0, 'step': 1.0, 'format': '%.1f', 'default': 170.0, 'required': False},      # 正常身高示例
    'bmi': {'min': 10.0, 'max': 50.0, 'step': 0.1, 'format': '%.1f', 'default': 22.5, 'required': False},          # 正常BMI18.5-24.9
    # 特殊配置
    'pao2fio2ratio_min': {'min': 20, 'max': 500, 'step': 10.0, 'format': '%.1f', 'default': 300.0, 'required': False}, # 正常>300
    'rdw_max': {'min': 10.0, 'max': 45.0, 'step': 0.5, 'format': '%.1f', 'default': 13.0, 'required': False},       # 正常RDW11-14%
    'potassium_max': {'min': 0.0, 'max': 10.0, 'step': 0.1, 'format': '%.1f', 'default': 4.0, 'required': False},   # 正常血钾3.5-5.0mmol/L
    'mchc_min': {'min': 20.0, 'max': 40.0, 'step': 0.1, 'format': '%.1f', 'default': 33.0, 'required': False},      # 正常MCHC32-36g/dL
}



# ==================== 动态生成输入组件 ====================
def generate_inputs(features):
    """根据特征动态生成输入表单"""
    input_values = {}

    # 参数输入区域
    with st.container():
        st.header("Key Parameters")  # 统一标题
        cols = st.columns(2)  # 两列布局

        for i, feat in enumerate(features):
            with cols[i % 2]:  # 交替分配参数到左、右两列
                cfg = FEATURE_CONFIG[feat]

                # 检查是否为整数格式
                if cfg['format'] == '%d':  # 直接匹配%d
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=int(cfg['min']),
                        max_value=int(cfg['max']),
                        value=cfg.get('default', int((cfg['min'] + cfg['max']) // 2)),
                        step=int(cfg['step']),
                        format=cfg['format']  # 现在为%d
                    )
                else:
                    # 处理浮点数
                    input_values[feat] = st.number_input(
                        label=feat,
                        min_value=float(cfg['min']),
                        max_value=float(cfg['max']),
                        value=cfg.get('default', float((cfg['min'] + cfg['max']) / 2) ),
                        step=float(cfg['step']),
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