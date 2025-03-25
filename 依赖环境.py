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
import subprocess
import sys  # 新增


# ==================== 新增：生成依赖文件函数 ====================
def save_requirements(filename='requirements.txt'):
    """自动生成项目依赖文件"""
    try:
        st.write("正在生成依赖文件...")

        # 获取当前环境已安装的包
        output = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'],
            stderr=subprocess.STDOUT
        )
        requirements = output.decode('utf-8').splitlines()

        # 过滤项目相关的核心包（可根据需要增减）
        core_packages = {
            'numpy', 'pandas', 'scikit-learn',
            'streamlit', 'joblib', 'shap',
            'matplotlib', 'python-dateutil'
        }
        filtered = [req for req in requirements if any(pkg in req for pkg in core_packages)]

        # 写入文件
        with open(filename, 'w') as f:
            f.write('\n'.join(filtered))

        st.success(f"依赖文件已生成：{os.path.abspath(filename)}")
        st.markdown("""