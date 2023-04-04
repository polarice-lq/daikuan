# 导入所需的库
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
#import seaborn as sns
#%matplotlib inline

# 导入数据
full_data = pd.read_csv("D:\\ST\\train_ctrUa4K.csv")
#full_data = pd.read_csv("D:\\ST\\train_ctrUa4K.csv")
full_data.shape

#对于数值变量：使用均值或中位数进行插补 对于分类变量：使用常见众数进行插补，这里主要使用众数进行插补空值
full_data['Gender'].fillna(full_data['Gender'].value_counts().idxmax(), inplace=True)
full_data['Married'].fillna(full_data['Married'].value_counts().idxmax(), inplace=True)
full_data['Dependents'].fillna(full_data['Dependents'].value_counts().idxmax(), inplace=True)
full_data['Self_Employed'].fillna(full_data['Self_Employed'].value_counts().idxmax(), inplace=True)
full_data["LoanAmount"].fillna(full_data["LoanAmount"].mean(skipna=True), inplace=True)
full_data['Loan_Amount_Term'].fillna(full_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
full_data['Credit_History'].fillna(full_data['Credit_History'].value_counts().idxmax(), inplace=True)

#对于异常值需要进行处理，这里采用对数log转化处理，消除异常值的影响，让数据回归正态分布
full_data['LoanAmount_log'] = np.log(full_data['LoanAmount'])
full_data['LoanAmount_log'].hist(bins=20)
full_data['ApplicantIncomeLog'] = np.log(full_data['ApplicantIncome'])
full_data['ApplicantIncomeLog'].hist(bins=20)

#类别特征值转化为数值
#教育
full_data['Education']=full_data['Education'].map({'Not Graduate':0,'Graduate':1})
#性别
full_data['Gender']=full_data['Gender'].map({'Male':0,'Female':1})
#结婚
full_data['Married']= full_data['Married'].map({'Yes':1,'No':0})
#工作
full_data['Self_Employed']= full_data['Self_Employed'].map({'Yes':0,'No':1})
#贷款状态
full_data['Loan_Status']=full_data['Loan_Status'].map({'Y':1,'N':0})
#信用历史
#full_data['Credit_History']=full_data['Credit_History'].map({'Y ':1,'N':0})

predictors = ['Education', 'ApplicantIncome', 'Married', 'LoanAmount','Credit_History','Loan_Amount_Term', ]
#predictors = ['Education', 'ApplicantIncome', 'Married', 'LoanAmount','Credit_History','Loan_Amount_Term']
#选择目标变量：
outcome = 'Loan_Status'

#选择预测模型：
model = LogisticRegression()

#用数据训练模型：
full_1=full_data[full_data['Loan_Status'].notnull()]
#full_1[predictors] = full_1[predictors].values
#full_1[outcome] = full_1[outcome].values
model.fit(full_1[predictors], full_1[outcome])

#用生成模型生成预测值
predicted = model.predict(full_1[predictors])

#比较预测值与实际值，得到预测准确度
accuracy = metrics.accuracy_score(predicted,full_1[outcome])
print("Accuracy :  %s" % "{0:.3%}" .format(accuracy))
# 设置全局样式
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            max-width: none;
            margin: 0 auto;
            padding: 0;
        }
        .stButton {
            background-color: #2a9d8f;
            color: #fff;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: bold;
            border: none;
            transition: all 0.2s ease-in-out;
        }
        .stButton:hover {
            background-color: #21867a;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# 设置输入框的样式
st.markdown(
    """
    <style>
        .stSelectbox {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: bold;
            width: 100%;
        }
        .stSlider {
            margin-top: 1rem;
        }
        .stTextInput {
            border-radius: 5px;
            border: 1px solid #ccc;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: bold;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True
)
#项目背景
with open('project_background.txt', 'r', encoding='utf-8') as f:
    project_background = f.read()

show_project_background = st.expander('点击显示项目背景', expanded=False)
with show_project_background:
    st.markdown(project_background)
# 创建 Streamlit 应用程序
# 定义SessionState类，用于存储用户信息
class SessionState:
    def __init__(self):
        self.user = None

# 获取SessionState实例
def get_session():
    return st.session_state.my_session

# 创建SessionState实例
if 'my_session' not in st.session_state:
    st.session_state.my_session = SessionState()

# 登录页面
def page_login():
    st.write("登录页面")
    password = st.text_input("请输入密码", type="password")
    if password == "123":
        get_session().user = "admin"
        st.success("登录成功")
    else:
        st.error("密码错误")

# 预测页面
def page_prediction():
    st.write("欢迎使用，%s" % get_session().user)
    st.write(
        f"<div style='background-color:#E5F0E8; padding:10px; border-radius:10px'>"
        f"<h1 style='text-align:center; font-family:Papyrus;'>贷款信息预测</h1>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<h1 style='font-weight: bold; margin-top: 0'>请输入您的个人信息</h1>", unsafe_allow_html=True)

    with st.sidebar:
        education = st.selectbox("教育程度", ["大学已毕业", "大学未毕业"])

    education = 1 if education == "大学已毕业" else 0
    print("education:",education)

    with st.sidebar:
        Credit_History = st.selectbox("行用记录", ["未有失信记录", "有失信记录"])
    Credit_History = 1 if Credit_History == "未有失信记录" else 0

    with st.sidebar:
        married = st.selectbox("婚姻状况", ["已婚", "未婚"])
    married = 1 if married == "已婚" else 0
    #with st.sidebar:
    income = st.slider("申请人收入/年（￥）", 0, 80000, 40000)
    amount = st.slider("贷款金额/月（￥）", 0, 800, 400)
    Loan_Amount_Term=st.slider("贷款期限/天", 0, 500, 250)
    #print("married:",married)
    #print("Credit_History:",Credit_History)
    X_new = [[education,income,married,amount,Credit_History,Loan_Amount_Term]]
    print(X_new)

    if st.button("获得结果"):
        # 运行模型并显示结果
        result = model.predict(X_new)[0]
        if result == 1:
            success("恭喜您，可以贷款！")
            st.write("🎉 成功！")
            st.balloons()
        else:
            warning("很抱歉，您无法获得贷款。")
            st.write("💔 很遗憾，期待你的下一次使用。")


# 设置输出结果的样式
st.markdown(
    """
    <style>
        .success {
            color: #2a9d8f;
            font-weight: bold;
            font-size: 1.5rem;
            text-align: center;
        }
        .error {
            color: #dc143c;
            font-weight: bold;
            font-size: 1.5rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
import streamlit as st
from streamlit import success, warning
from PIL import Image

# 在头部添加CSS样式
st.write(
    f"""
    <style>
        div.stButton > button:first-child {{
            color: black !important;
            font-family: cursive !important;
            font-size: 20px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 创建菜单
menu = ["登录", "数据预测"]
choice = st.sidebar.selectbox("选择一个选项", menu)

# 根据用户选择的选项调用相应的页面
if choice == "登录":
    page_login()
elif choice == "数据预测":
    if get_session().user is not None:
        page_prediction()
    else:
        st.warning("请先登录")
        page_login()