import streamlit as st
import pandas as pd
import joblib
import os

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
            
            # 特征列定义（根据训练数据保持一致）
            feature_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 
                              'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 
                              'x18', 'x19']
            
            # 验证特征列
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                st.error(f"数据中缺少必要特征列：{missing_cols}")
                return
            
            # 提取特征数据
            X = df[feature_columns]
            
            # 显示数据预览
            st.markdown("#### 数据预览")
            st.dataframe(X.head(), use_container_width=True)
            
            # 预测按钮
            if st.button("🔮 开始预测", type="primary"):
                with st.spinner("正在预测..."):
                    # 进行预测
                    predictions = model.predict(X)
                    
                    # 创建结果DataFrame
                    result_df = pd.DataFrame({
                        '预测结果': predictions
                    })
                    
                    # 合并原始数据和预测结果
                    final_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
                    
                    # 显示预测结果
                    st.markdown("### 📌 预测结果")
                    st.dataframe(final_df, use_container_width=True)
                    
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
    
    # 模型信息显示
    st.markdown("### ℹ️ 模型信息")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        st.json(params, expanded=False)
    else:
        st.info("无法获取模型参数信息")

if __name__ == "__main__":
    main()