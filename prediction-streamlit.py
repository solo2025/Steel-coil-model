import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="钢卷性能预测平台",
    page_icon="📊",
    layout="wide"
)

# 加载CSS样式
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 初始化模型列表
@st.cache_resource
def load_models():
    model_dir = 'models'
    models = {}
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                model_path = os.path.join(model_dir, file)
                models[file] = model_path
    return models

# 安全归一化函数
def safe_normalize(df):
    """处理可能产生NaN的归一化"""
    normalized_df = df.copy()
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        # 避免除以0
        if col_max != col_min:
            normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0  # 所有值相同则设为0
    return normalized_df

# 数据验证函数
def validate_data(df, required_cols):
    """检查数据有效性"""
    # 检查缺失列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要特征列：{missing_cols}")
    
    # 检查无效值
    if df.isnull().any().any():
        raise ValueError("数据包含空值")
    
    # 检查无穷大值
    if np.isinf(df.values).any():
        raise ValueError("数据包含无穷大值")

# 主函数
def main():
    st.title("📊 钢卷性能机器学习预测平台")
    st.markdown("### 使用训练好的模型进行预测")
    
    # 侧边栏配置
    st.sidebar.header("模型配置")
    
    # 加载模型列表
    models_dict = load_models()
    
    if not models_dict:
        st.error("未找到任何模型文件！请确认output目录下存在pkl模型文件")
        return
    
    # 模型选择器
    selected_model = st.sidebar.selectbox(
        "选择模型文件",
        options=list(models_dict.keys()),
        help="选择要使用的训练模型"
    )
    
    # 加载选中的模型
    try:
        model = joblib.load(models_dict[selected_model])
        st.sidebar.success("模型加载成功")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {str(e)}")
        return
    
    # 文件上传区域
    st.markdown("### 📁 数据上传")
    uploaded_file = st.file_uploader(
        "上传CSV测试数据文件",
        type=["csv"],
        help="请上传包含特征列的CSV文件（需与训练数据格式一致）"
    )
    
    if uploaded_file is not None:
        try:
            # 读取数据
            df = pd.read_csv(uploaded_file)
            
            # 定义特征列（根据实际模型调整）
            feature_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 
                             'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 
                             'x18', 'x19']
            
            # 验证数据
            validate_data(df, feature_columns)
            
            # 提取特征数据
            X = df[feature_columns]

            # 数据预览（限制显示行数）
            st.markdown("#### 数据预览（前5行）")
            st.dataframe(X.head(), use_container_width=True, height=200)

            # 安全归一化
            X_normalized = safe_normalize(X)

            # 预测按钮
            if st.button("🔮 开始预测", type="primary"):
                with st.spinner("正在预测..."):
                    # 进行预测
                    predictions = model.predict(X_normalized)
                    
                    # 创建结果DataFrame
                    result_df = pd.DataFrame({
                        '预测结果': predictions
                    })
                    
                    # 合并原始数据和预测结果
                    final_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
                    
                    # 显示预测结果（分页显示）
                    st.markdown("### 📌 预测结果（前100行）")
                    st.dataframe(final_df.head(100), use_container_width=True)
                    
                    # 提供下载链接
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载预测结果",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"数据处理错误: {str(e)}")
            st.exception(e)  # 显示完整错误堆栈
    
    # 模型信息显示
    st.markdown("### ℹ️ 模型信息")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        st.json(params, expanded=False)
    else:
        st.info("无法获取模型参数信息")

if __name__ == "__main__":
    main()