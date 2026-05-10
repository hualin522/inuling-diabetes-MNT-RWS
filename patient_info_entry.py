import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import os

# ===== 依赖自检 =====
missing_pkgs = []
required_pkgs = {
    "gspread": "gspread",
    "google.oauth2": "google-auth",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_huggingface": "langchain-huggingface",
    "langchain_deepseek": "langchain-deepseek",
    "langchain_core": "langchain-core",
    "langchain.chains": "langchain",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "pypdf": "pypdf",
}
for mod, pkg in required_pkgs.items():
    try:
        __import__(mod)
    except ImportError:
        missing_pkgs.append(pkg)

if missing_pkgs:
    st.error(
        f"❌ 缺少必要的 Python 包，请在 requirements.txt 中添加以下依赖:\n\n"
        + "\n".join(missing_pkgs)
        + "\n\n然后重新部署应用。"
    )
    st.stop()

# 动态导入（确保检查通过后再导入）
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import gspread
from google.oauth2 import service_account

# ============================================
# 自动计算函数
# ============================================
def calculate_age(birth_date):
    if birth_date is None:
        return None
    today = date.today()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age

def calculate_disease_years(diagnosis_date):
    if diagnosis_date is None:
        return None
    today = date.today()
    days = (today - diagnosis_date).days
    if days < 0:
        return 0.0
    return round(days / 365.25, 1)

def calculate_bmi(height_cm, weight_kg):
    if not height_cm or not weight_kg or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)

def calculate_symptom_total(symptom_dict):
    """体感总分：任一子项为 None 则返回 None，否则求和"""
    if any(v is None for v in symptom_dict.values()):
        return None
    return sum(symptom_dict.values())

def calculate_duration(start_date, end_date):
    if start_date and end_date:
        delta = end_date - start_date
        return delta.days
    return None

def parse_other_meds(meds_text):
    """解析其他药物多行文本，返回列表"""
    meds_list = []
    if meds_text and meds_text.strip():
        for line in meds_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.replace('，', ',').split(',')
            if len(parts) >= 3:
                name = parts[0].strip()
                try:
                    times = float(parts[1].strip())
                    dose = float(parts[2].strip())
                except ValueError:
                    times, dose = 0, 0
                meds_list.append({"药名": name, "每天次数": times, "每次剂量": dose})
            else:
                st.warning(f"其他药物格式有误：{line}，已忽略")
    return meds_list

def plot_glucose_curve(glucose_values, title):
    if not all(glucose_values):
        return None, None
    times = [0, 0.5, 1, 2, 3]
    auc = 0
    for i in range(len(times)-1):
        auc += (glucose_values[i] + glucose_values[i+1]) / 2 * (times[i+1] - times[i])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=glucose_values, mode='lines+markers', name=title))
    fig.update_layout(
        title=title,
        xaxis_title='时间 (小时)',
        yaxis_title='血糖 (mmol/L)',
        xaxis=dict(tickmode='array', tickvals=times, ticktext=['空腹','0.5h','1h','2h','3h']),
        yaxis=dict(range=[0, max(glucose_values) * 1.05])
    )
    return fig, round(auc, 2)

# ============================================
# 异常值提醒辅助函数
# ============================================
def warn_range(label, value, min_val, max_val, unit=""):
    if value is not None and value != 0:
        if value < min_val or value > max_val:
            st.warning(f"⚠️ {label}：{value}{unit} 超出合理范围（{min_val}-{max_val}{unit}）")

def warn_date_logic(label, target_date, reference_date=None, allow_future=False):
    if target_date is None:
        return
    today = date.today()
    if not allow_future and target_date > today:
        st.warning(f"⚠️ {label} {target_date} 不能晚于今天")
    if reference_date and target_date < reference_date:
        st.warning(f"⚠️ {label} {target_date} 不应早于参考日期 {reference_date}")

# ============================================
# Google Sheets 辅助函数
# ============================================
def flatten_dict(d, parent_key='', sep='_'):
    """递归展开字典，处理不可序列化的类型"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        elif isinstance(v, (date, datetime)):
            items.append((new_key, v.isoformat()))
        elif v is None:
            items.append((new_key, ""))
        else:
            items.append((new_key, v))
    return dict(items)

def save_to_google_sheets(patient_dict):
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(st.secrets["google_sheets"]["spreadsheet_id"]).sheet1

        flat = flatten_dict(patient_dict)

        header_row = sheet.row_values(1)
        if not header_row or all(cell == '' for cell in header_row):
            header_to_write = list(flat.keys())
            sheet.append_row(header_to_write)
            header_row = header_to_write

        row_data = [flat.get(col, "") for col in header_row]
        sheet.append_row(row_data)
        st.success("✅ 数据已同步至 Google Sheets")
    except Exception as e:
        st.warning(f"⚠️ Google Sheets 写入失败（数据已保存在本地列表中）: {e}")

# ============================================
# DeepSeek + 本地知识库问答模块
# ============================================

# ---------- 1. 加载本地知识库（缓存） ----------
@st.cache_resource
def load_knowledge_base(pdf_dir="pdf_data"):
    """加载 PDF 并创建 FAISS 向量数据库"""
    if not os.path.exists(pdf_dir):
        st.error(f"知识库目录 {pdf_dir} 不存在，请创建并放入 PDF 文件")
        return None
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    if not docs:
        st.warning("未检测到任何 PDF 文档，知识库为空")
        return None
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    
    # 向量化
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(chunks, embedding)
    return vectordb

# ---------- 2. 构建 RAG 问答链 ----------
def build_rag_chain(vectordb):
    """创建检索 + 生成链"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    template = """
你是一位资深的临床营养师，专攻应用菊粉类益生元等营养补充方式进行糖尿病营养管理。请根据以下资料回答问题：
1. 本地专业文档
2. 当前患者的具体干预前数据

如果信息不足，你可以结合公认的医学知识给出建议，但必须注明依据来源。

【本地文档内容】
{context}

【患者干预前数据】
身高：{height} cm
体重：{weight} kg
BMI：{bmi}
腰围：{waist} cm
高压：{sbp} mmHg
低压：{dbp} mmHg
空腹血糖：{fpg} mmol/L
餐后2h血糖：{pg2h} mmol/L
糖化血红蛋白：{hba1c}%
体感总分：{symptom_total}
其他慢病：{chronic}
并发症：{complications}

【用户问题】
{input}

请分点列出个体化的营养治疗方案和建议，并解释预期效果。
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # API Key 保护
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("❌ 请在 Streamlit Secrets 中设置 DEEPSEEK_API_KEY")
        st.stop()
    
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.3
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

# ---------- 3. 生成方案建议 ----------
def generate_plan(patient_data: dict) -> str:
    """根据患者数据调用 RAG 生成营养方案"""
    vectordb = load_knowledge_base()
    if vectordb is None:
        return "❌ 知识库未加载，请检查 PDF 文件"
    
    rag_chain = build_rag_chain(vectordb)
    
    # 构造输入（缺失则填“未知”）
    input_data = {
        "height": patient_data.get("干预前身高", "未知"),
        "weight": patient_data.get("干预前体重", "未知"),
        "bmi": patient_data.get("干预前BMI", "未知"),
        "waist": patient_data.get("干预前腰围", "未知"),
        "sbp": patient_data.get("干预前高压", "未知"),
        "dbp": patient_data.get("干预前低压", "未知"),
        "fpg": patient_data.get("干预前FPG", "未知"),
        "pg2h": patient_data.get("干预前PG120", "未知"),
        "hba1c": patient_data.get("干预前糖化", "未知"),
        "symptom_total": patient_data.get("干预前体感总分", "未知"),
        "chronic": patient_data.get("其他慢病", "无"),
        "complications": patient_data.get("并发症", "无"),
    }
    
    result = rag_chain.invoke({
        "input": "请为这位糖尿病患者制定个体化的营养治疗方案，并预测可能的效果",
        "height": input_data["height"],
        "weight": input_data["weight"],
        "bmi": input_data["bmi"],
        "waist": input_data["waist"],
        "sbp": input_data["sbp"],
        "dbp": input_data["dbp"],
        "fpg": input_data["fpg"],
        "pg2h": input_data["pg2h"],
        "hba1c": input_data["hba1c"],
        "symptom_total": input_data["symptom_total"],
        "chronic": input_data["chronic"],
        "complications": input_data["complications"],
    })
    return result["answer"]


# ============================================
# 信息录入主界面
# ============================================
def patient_info_entry():
    st.header("📋 英纽林糖尿病营养治疗真实世界研究案例收集")

    if "patients" not in st.session_state:
        st.session_state.patients = []

    # ---- 将年龄/病史输入方式移到表单外 ----
    st.subheader("基本信息输入方式")
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        age_mode = st.radio("年龄输入方式", ["自动计算", "手动输入"], horizontal=True, key="age_mode_radio")
    with col_mode2:
        disease_mode = st.radio("病史年输入方式", ["自动计算", "手动输入"], horizontal=True, key="disease_mode_radio")
    st.markdown("---")

    # 手动输入时年龄和病史的 session_state 初始化
    if "age_manual" not in st.session_state:
        st.session_state.age_manual = 0
    if "disease_manual" not in st.session_state:
        st.session_state.disease_manual = 0.0

    with st.form(key="patient_form", clear_on_submit=False, enter_to_submit=False):
        # ===== 1. 用户基本信息 =====
        with st.expander("1️⃣ 用户基本信息", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                name = st.text_input("患者姓名 *")
                gender = st.selectbox("性别", ["男", "女"])
                phone = st.text_input("联系电话")
            with col2:
                birth_date = st.date_input("出生日期", value=None, min_value=date(1900, 1, 1), format="YYYY-MM-DD")
                auto_age = calculate_age(birth_date)
                if age_mode == "自动计算":
                    age_disabled = True
                    age_value = auto_age if auto_age is not None else 0
                else:
                    age_disabled = False
                    age_value = st.session_state.age_manual

                age_input = st.number_input(
                    "年龄（岁）",
                    min_value=0, max_value=120, step=1,
                    value=age_value,
                    disabled=age_disabled,
                    key="age_manual"
                )
                if age_mode == "自动计算":
                    age_input = auto_age
                if age_input is not None and age_input != 0:
                    warn_range("年龄", age_input, 0, 120, "岁")
            with col3:
                diagnosis_date = st.date_input("确诊日期/年月日", value=None, min_value=date(1900, 1, 1), format="YYYY-MM-DD")
                auto_disease = calculate_disease_years(diagnosis_date)
                if disease_mode == "自动计算":
                    disease_disabled = True
                    disease_value = auto_disease if auto_disease is not None else 0.0
                else:
                    disease_disabled = False
                    disease_value = st.session_state.disease_manual

                disease_input = st.number_input(
                    "病史/年",
                    min_value=0.0, max_value=80.0, step=0.5,
                    value=disease_value,
                    disabled=disease_disabled,
                    key="disease_manual"
                )
                if disease_mode == "自动计算":
                    disease_input = auto_disease
                if disease_input is not None and disease_input != 0.0:
                    warn_range("病史年", disease_input, 0, 80, "年")
                warn_date_logic("确诊日期", diagnosis_date, allow_future=False)
            with col4:
                location = st.text_input("所在地/省/市/区")
                complications = st.text_input("并发症 (若无填无)")
                other_chronic = st.text_input("其他慢病")

        # ===== 2. 干预前基本指标 =====
        with st.expander("2️⃣ 干预前基本指标", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                pre_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=None, step=0.1, key="pre_h")
                pre_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=None, step=0.1, key="pre_w")
                pre_bmi = calculate_bmi(pre_height, pre_weight)
                st.text_input("BMI (自动计算)", value=f"{pre_bmi}" if pre_bmi else "待填写", disabled=True, key="pre_bmi_display")
                warn_range("身高(前)", pre_height, 50, 250, "cm")
                warn_range("体重(前)", pre_weight, 10, 150, "kg")
            with col2:
                pre_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1)
                pre_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1)
                pre_sbp = st.number_input("高压 (mmHg)", min_value=50, max_value=250, value=None, step=1)
                pre_dbp = st.number_input("低压 (mmHg)", min_value=30, max_value=150, value=None, step=1)
                warn_range("腰围(前)", pre_waist, 50, 200, "cm")
                warn_range("臀围(前)", pre_hip, 50, 200, "cm")
                warn_range("高压(前)", pre_sbp, 70, 200, "mmHg")
                warn_range("低压(前)", pre_dbp, 40, 120, "mmHg")

        # ===== 3. 干预后基本指标 =====
        with st.expander("3️⃣ 干预后基本指标", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                post_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=None, step=0.1, key="post_h")
                post_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=None, step=0.1, key="post_w")
                post_bmi = calculate_bmi(post_height, post_weight)
                st.text_input("BMI (自动计算)", value=f"{post_bmi}" if post_bmi else "待填写", disabled=True, key="post_bmi_display")
                warn_range("身高(后)", post_height, 50, 250, "cm")
                warn_range("体重(后)", post_weight, 10, 150, "kg")
            with col2:
                post_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1, key="post_wc")
                post_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1, key="post_hc")
                post_sbp = st.number_input("高压 (mmHg)", min_value=50, max_value=250, value=None, step=1, key="post_sbp")
                post_dbp = st.number_input("低压 (mmHg)", min_value=30, max_value=150, value=None, step=1, key="post_dbp")
                warn_range("腰围(后)", post_waist, 50, 200, "cm")
                warn_range("臀围(后)", post_hip, 50, 200, "cm")
                warn_range("高压(后)", post_sbp, 70, 200, "mmHg")
                warn_range("低压(后)", post_dbp, 40, 120, "mmHg")

        # ===== 4. 干预前体感指标 =====
        with st.expander("4️⃣ 干预前体感指标", expanded=False):
            pre_symptom_date = st.date_input("干预前录入日期", value=None, min_value=date(1900,1,1), key="symptom_pre_date")
            warn_date_logic("干预前体感日期", pre_symptom_date, allow_future=False)
            col1, col2, col3 = st.columns(3)
            with col1:
                pre_halitosis = st.number_input("口臭", 1, 10, value=None, key="pre_hal")
                pre_defecation = st.number_input("排便情况", 1, 10, value=None, key="pre_def")
                pre_gi = st.number_input("胃肠道", 1, 10, value=None, key="pre_gi")
                pre_numbness = st.number_input("四肢麻木", 1, 10, value=None, key="pre_num")
            with col2:
                pre_pruritus = st.number_input("皮肤瘙痒", 1, 10, value=None, key="pre_pru")
                pre_sleep = st.number_input("睡眠", 1, 10, value=None, key="pre_sleep")
                pre_vision = st.number_input("视物", 1, 10, value=None, key="pre_vis")
                pre_fatigue = st.number_input("乏力", 1, 10, value=None, key="pre_fat")
            with col3:
                pre_polydipsia = st.number_input("多饮", 1, 10, value=None, key="pre_polyd")
                pre_polyphagia = st.number_input("多食", 1, 10, value=None, key="pre_polyp")
                pre_polyuria = st.number_input("多尿", 1, 10, value=None, key="pre_polyu")
                pre_lumbago = st.number_input("腰膝酸软", 1, 10, value=None, key="pre_lumb")
            col1, col2 = st.columns(2)
            with col1:
                pre_night_sweat = st.number_input("盗汗情况", 1, 10, value=None, key="pre_night")
            with col2:
                pre_mood = st.number_input("情绪状况", 1, 10, value=None, key="pre_mood")
            pre_symptom_scores = {
                "口臭": pre_halitosis, "排便情况": pre_defecation, "胃肠道": pre_gi,
                "四肢麻木": pre_numbness, "皮肤瘙痒": pre_pruritus, "睡眠": pre_sleep,
                "视物": pre_vision, "乏力": pre_fatigue, "多饮": pre_polydipsia,
                "多食": pre_polyphagia, "多尿": pre_polyuria, "腰膝酸软": pre_lumbago,
                "盗汗情况": pre_night_sweat, "情绪状况": pre_mood
            }
            pre_total = calculate_symptom_total(pre_symptom_scores)
            st.text_input("体感总分 (自动计算)", value=str(pre_total) if pre_total is not None else "待填写", disabled=True, key="pre_total_display")

        # ===== 5. 干预后体感指标 =====
        with st.expander("5️⃣ 干预后体感指标", expanded=False):
            post_symptom_date = st.date_input("干预后录入日期", value=None, min_value=date(1900,1,1), key="symptom_post_date")
            warn_date_logic("干预后体感日期", post_symptom_date, allow_future=False)
            col1, col2, col3 = st.columns(3)
            with col1:
                post_halitosis = st.number_input("口臭", 1, 10, value=None, key="post_hal")
                post_defecation = st.number_input("排便情况", 1, 10, value=None, key="post_def")
                post_gi = st.number_input("胃肠道", 1, 10, value=None, key="post_gi")
                post_numbness = st.number_input("四肢麻木", 1, 10, value=None, key="post_num")
            with col2:
                post_pruritus = st.number_input("皮肤瘙痒", 1, 10, value=None, key="post_pru")
                post_sleep = st.number_input("睡眠", 1, 10, value=None, key="post_sleep")
                post_vision = st.number_input("视物", 1, 10, value=None, key="post_vis")
                post_fatigue = st.number_input("乏力", 1, 10, value=None, key="post_fat")
            with col3:
                post_polydipsia = st.number_input("多饮", 1, 10, value=None, key="post_polyd")
                post_polyphagia = st.number_input("多食", 1, 10, value=None, key="post_polyp")
                post_polyuria = st.number_input("多尿", 1, 10, value=None, key="post_polyu")
                post_lumbago = st.number_input("腰膝酸软", 1, 10, value=None, key="post_lumb")
            col1, col2 = st.columns(2)
            with col1:
                post_night_sweat = st.number_input("盗汗情况", 1, 10, value=None, key="post_night")
            with col2:
                post_mood = st.number_input("情绪状况", 1, 10, value=None, key="post_mood")
            post_symptom_scores = {
                "口臭": post_halitosis, "排便情况": post_defecation, "胃肠道": post_gi,
                "四肢麻木": post_numbness, "皮肤瘙痒": post_pruritus, "睡眠": post_sleep,
                "视物": post_vision, "乏力": post_fatigue, "多饮": post_polydipsia,
                "多食": post_polyphagia, "多尿": post_polyuria, "腰膝酸软": post_lumbago,
                "盗汗情况": post_night_sweat, "情绪状况": post_mood
            }
            post_total = calculate_symptom_total(post_symptom_scores)
            st.text_input("体感总分 (自动计算)", value=str(post_total) if post_total is not None else "待填写", disabled=True, key="post_total_display")

        # ===== 6. 干预前糖尿病药物 =====
        with st.expander("6️⃣ 干预前糖尿病药物", expanded=False):
            st.subheader("胰岛素")
            col1, col2 = st.columns(2)
            with col1:
                pre_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0, step=1, value=None, key="pre_ins_times")
                warn_range("胰岛素次数(前)", pre_insulin_times, 0, 10, "次/天")
            with col2:
                pre_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0, value=None, key="pre_ins_dose")
                warn_range("胰岛素剂量(前)", pre_insulin_dose, 0, 100, "IU")
            st.subheader("口服药")
            col1, col2, col3 = st.columns(3)
            with col1:
                pre_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0, step=1, value=None, key="pre_met_times")
                pre_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0, step=250, value=None, key="pre_met_dose")
                warn_range("二甲双胍剂量(前)", pre_metformin_dose, 0, 2000, "mg")
            with col2:
                pre_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0, step=1, value=None, key="pre_acb_times")
                pre_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0, step=50, value=None, key="pre_acb_dose")
                warn_range("阿卡波糖剂量(前)", pre_acarbose_dose, 0, 300, "mg")
            with col3:
                pre_other_meds = st.text_area(
                    "其他药物（每行一种，格式：药名，每天次数，每次剂量）",
                    placeholder="例如：\n格列美脲，1，2\n达格列净，1，10",
                    key="pre_other_meds",
                    help="每行一种药物，用英文逗号或中文逗号分隔，依次填写：药名，每天次数，每次剂量"
                )

        # ===== 7. 干预后糖尿病药物 =====
        with st.expander("7️⃣ 干预后糖尿病药物", expanded=False):
            st.subheader("胰岛素")
            col1, col2 = st.columns(2)
            with col1:
                post_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0, step=1, value=None, key="post_ins_times")
                warn_range("胰岛素次数(后)", post_insulin_times, 0, 10, "次/天")
            with col2:
                post_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0, value=None, key="post_ins_dose")
                warn_range("胰岛素剂量(后)", post_insulin_dose, 0, 100, "IU")
            st.subheader("口服药")
            col1, col2, col3 = st.columns(3)
            with col1:
                post_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0, step=1, value=None, key="post_met_times")
                post_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0, step=250, value=None, key="post_met_dose")
                warn_range("二甲双胍剂量(后)", post_metformin_dose, 0, 2000, "mg")
            with col2:
                post_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0, step=1, value=None, key="post_acb_times")
                post_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0, step=50, value=None, key="post_acb_dose")
                warn_range("阿卡波糖剂量(后)", post_acarbose_dose, 0, 300, "mg")
            with col3:
                post_other_meds = st.text_area(
                    "其他药物（每行一种，格式：药名，每天次数，每次剂量）",
                    placeholder="例如：\n格列美脲，1，2\n达格列净，1，10",
                    key="post_other_meds",
                    help="每行一种药物，用英文逗号或中文逗号分隔，依次填写：药名，每天次数，每次剂量"
                )

        # ===== 8. 用药调整情况 =====
        with st.expander("8️⃣ 用药调整情况", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                drug_pre_date = st.date_input("干预前日期", value=None, min_value=date(1900,1,1), key="drug_pre_date")
                drug_pre_med = st.text_area("干预前用药 (可简述)", key="drug_pre_med")
            with col2:
                drug_post_date = st.date_input("干预后日期", value=None, min_value=date(1900,1,1), key="drug_post_date")
                drug_post_med = st.text_area("干预后用药 (可简述)", key="drug_post_med")
            drug_reduction = st.selectbox("减药/停药", ["无变化", "减剂量", "减种类", "停用所有口服", "其他"], key="drug_reduction")

        # ===== 9. 干预前生化指标 =====
        with st.expander("9️⃣ 干预前生化指标", expanded=False):
            pre_bio_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="pre_bio_date")
            warn_date_logic("干预前生化日期", pre_bio_date, allow_future=False)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pre_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1, value=None, key="pre_hba1c")
                pre_tc = st.number_input("TC (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_tc")
                warn_range("糖化(前)", pre_hba1c, 3, 15, "%")
                warn_range("总胆固醇(前)", pre_tc, 1, 20, "mmol/L")
            with col2:
                pre_tg = st.number_input("TG (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_tg")
                pre_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_ldl")
                warn_range("甘油三酯(前)", pre_tg, 0.2, 15, "mmol/L")
                warn_range("LDL-C(前)", pre_ldl, 0.5, 10, "mmol/L")
            with col3:
                pre_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_hdl")
                pre_alt = st.number_input("ALT (U/L)", min_value=0, step=1, value=None, key="pre_alt")
                warn_range("HDL-C(前)", pre_hdl, 0.2, 3, "mmol/L")
                warn_range("ALT(前)", pre_alt, 0, 1000, "U/L")
            with col4:
                pre_ast = st.number_input("AST (U/L)", min_value=0, step=1, value=None, key="pre_ast")
                warn_range("AST(前)", pre_ast, 0, 1000, "U/L")

        # ===== 10. 干预后生化指标 =====
        with st.expander("🔟 干预后生化指标", expanded=False):
            post_bio_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_bio_date")
            warn_date_logic("干预后生化日期", post_bio_date, allow_future=False)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                post_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1, value=None, key="post_hba1c")
                post_tc = st.number_input("TC (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_tc")
                warn_range("糖化(后)", post_hba1c, 3, 15, "%")
                warn_range("总胆固醇(后)", post_tc, 1, 20, "mmol/L")
            with col2:
                post_tg = st.number_input("TG (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_tg")
                post_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_ldl")
                warn_range("甘油三酯(后)", post_tg, 0.2, 15, "mmol/L")
                warn_range("LDL-C(后)", post_ldl, 0.5, 10, "mmol/L")
            with col3:
                post_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_hdl")
                post_alt = st.number_input("ALT (U/L)", min_value=0, step=1, value=None, key="post_alt")
                warn_range("HDL-C(后)", post_hdl, 0.2, 3, "mmol/L")
                warn_range("ALT(后)", post_alt, 0, 1000, "U/L")
            with col4:
                post_ast = st.number_input("AST (U/L)", min_value=0, step=1, value=None, key="post_ast")
                warn_range("AST(后)", post_ast, 0, 1000, "U/L")

        # ===== 11. 干预前5点血糖指标 =====
        with st.expander("1️⃣1️⃣ 干预前5点血糖指标", expanded=False):
            pre_glyc_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="pre_glyc_date")
            warn_date_logic("干预前5点血糖日期", pre_glyc_date, allow_future=False)
            pre_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_fpg")
            warn_range("空腹血糖(前)", pre_fpg, 2, 30, "mmol/L")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pre_pg30 = st.number_input("PG 30min (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_pg30")
                warn_range("PG30(前)", pre_pg30, 2, 30, "mmol/L")
            with col2:
                pre_pg60 = st.number_input("PG 60min (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_pg60")
                warn_range("PG60(前)", pre_pg60, 2, 30, "mmol/L")
            with col3:
                pre_pg120 = st.number_input("PG 120min (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_pg120")
                warn_range("PG120(前)", pre_pg120, 2, 30, "mmol/L")
            with col4:
                pre_pg180 = st.number_input("PG 180min (mmol/L)", min_value=0.0, step=0.1, value=None, key="pre_pg180")
                warn_range("PG180(前)", pre_pg180, 2, 30, "mmol/L")

        # ===== 12. 干预后5点血糖指标 =====
        with st.expander("1️⃣2️⃣ 干预后5点血糖指标", expanded=False):
            post_glyc_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_glyc_date")
            warn_date_logic("干预后5点血糖日期", post_glyc_date, allow_future=False)
            post_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_fpg")
            warn_range("空腹血糖(后)", post_fpg, 2, 30, "mmol/L")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                post_pg30 = st.number_input("PG 30min (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_30")
                warn_range("PG30(后)", post_pg30, 2, 30, "mmol/L")
            with col2:
                post_pg60 = st.number_input("PG 60min (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_60")
                warn_range("PG60(后)", post_pg60, 2, 30, "mmol/L")
            with col3:
                post_pg120 = st.number_input("PG 120min (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_120")
                warn_range("PG120(后)", post_pg120, 2, 30, "mmol/L")
            with col4:
                post_pg180 = st.number_input("PG 180min (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_180")
                warn_range("PG180(后)", post_pg180, 2, 30, "mmol/L")

        # ===== 13. 干预方案 =====
        with st.expander("1️⃣3️⃣ 干预方案", expanded=False):
            intervention_type = st.selectbox("营养治疗方案", ["畅快", "纽畅", "纽畅B", "其他营养治疗"], key="intervention_type")
            intervention_detail = st.text_area(
                "方案细节（用量/用法/周期等）",
                placeholder="例如：纽畅B 每日2次，每次1包，餐后服用",
                key="intervention_detail"
            )

        # ===== 14. 干预前日常7点血糖 =====
        with st.expander("1️⃣4️⃣ 干预前日常7点血糖", expanded=False):
            pre_7_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="pre_7_date")
            warn_date_logic("干预前7点血糖日期", pre_7_date, allow_future=False)
            cols = st.columns(7)
            with cols[0]:
                pre_bf_before = st.number_input("早餐前 (mmol/L)", step=0.1, value=None, key="pre_bf_before")
                warn_range("早餐前(前)", pre_bf_before, 2, 30, "mmol/L")
            with cols[1]:
                pre_bf_after = st.number_input("早餐后2h (mmol/L)", step=0.1, value=None, key="pre_bf_after")
                warn_range("早餐后2h(前)", pre_bf_after, 2, 30, "mmol/L")
            with cols[2]:
                pre_lunch_before = st.number_input("午餐前 (mmol/L)", step=0.1, value=None, key="pre_lunch_before")
                warn_range("午餐前(前)", pre_lunch_before, 2, 30, "mmol/L")
            with cols[3]:
                pre_lunch_after = st.number_input("午餐后2h (mmol/L)", step=0.1, value=None, key="pre_lunch_after")
                warn_range("午餐后2h(前)", pre_lunch_after, 2, 30, "mmol/L")
            with cols[4]:
                pre_dinner_before = st.number_input("晚餐前 (mmol/L)", step=0.1, value=None, key="pre_dinner_before")
                warn_range("晚餐前(前)", pre_dinner_before, 2, 30, "mmol/L")
            with cols[5]:
                pre_dinner_after = st.number_input("晚餐后2h (mmol/L)", step=0.1, value=None, key="pre_dinner_after")
                warn_range("晚餐后2h(前)", pre_dinner_after, 2, 30, "mmol/L")
            with cols[6]:
                pre_bed = st.number_input("睡前 (mmol/L)", step=0.1, value=None, key="pre_bed")
                warn_range("睡前(前)", pre_bed, 2, 30, "mmol/L")

        # ===== 15. 干预后日常7点血糖 =====
        with st.expander("1️⃣5️⃣ 干预后日常7点血糖", expanded=False):
            post_7_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_7_date")
            warn_date_logic("干预后7点血糖日期", post_7_date, allow_future=False)
            cols = st.columns(7)
            with cols[0]:
                post_bf_before = st.number_input("早餐前 (mmol/L)", step=0.1, value=None, key="post_bf_before")
                warn_range("早餐前(后)", post_bf_before, 2, 30, "mmol/L")
            with cols[1]:
                post_bf_after = st.number_input("早餐后2h (mmol/L)", step=0.1, value=None, key="post_bf_after")
                warn_range("早餐后2h(后)", post_bf_after, 2, 30, "mmol/L")
            with cols[2]:
                post_lunch_before = st.number_input("午餐前 (mmol/L)", step=0.1, value=None, key="post_lunch_before")
                warn_range("午餐前(后)", post_lunch_before, 2, 30, "mmol/L")
            with cols[3]:
                post_lunch_after = st.number_input("午餐后2h (mmol/L)", step=0.1, value=None, key="post_lunch_after")
                warn_range("午餐后2h(后)", post_lunch_after, 2, 30, "mmol/L")
            with cols[4]:
                post_dinner_before = st.number_input("晚餐前 (mmol/L)", step=0.1, value=None, key="post_dinner_before")
                warn_range("晚餐前(后)", post_dinner_before, 2, 30, "mmol/L")
            with cols[5]:
                post_dinner_after = st.number_input("晚餐后2h (mmol/L)", step=0.1, value=None, key="post_dinner_after")
                warn_range("晚餐后2h(后)", post_dinner_after, 2, 30, "mmol/L")
            with cols[6]:
                post_bed = st.number_input("睡前 (mmol/L)", step=0.1, value=None, key="post_bed")
                warn_range("睡前(后)", post_bed, 2, 30, "mmol/L")

        # ===== 16. 案例来源与备注 =====
        with st.expander("1️⃣6️⃣ 案例来源 & 备注", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                project_region = st.text_input("项目/医疗地区")
                health_coach = st.text_input("健管师")
                doctor = st.text_input("医生")
                clinic_name = st.text_input("诊所/门店名称")
            with col2:
                submitter = st.text_input("提交人")
                supervisor = st.text_input("指导健管师")
            remarks = st.text_area("备注信息")

        # ===== 提交按钮 =====
        submitted = st.form_submit_button("✅ 提交并保存患者信息")

        if submitted:
            if not name:
                st.error("患者姓名不能为空")
                st.stop()

            # 解析其他药物
            pre_other_list = parse_other_meds(pre_other_meds)
            post_other_list = parse_other_meds(post_other_meds)

            # 计算干预时长
            duration_days = calculate_duration(pre_glyc_date, post_glyc_date)

            # 组装完整患者数据字典
            patient_data = {
                "录入时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "患者姓名": name,
                "联系电话": phone,
                "所在地": location,
                "性别": gender,
                "出生日期": birth_date,
                "年龄": age_input,
                "确诊日期": diagnosis_date,
                "病史年": disease_input,
                "并发症": complications,
                "其他慢病": other_chronic,
                "干预前身高": pre_height,
                "干预前体重": pre_weight,
                "干预前BMI": pre_bmi,
                "干预前腰围": pre_waist,
                "干预前臀围": pre_hip,
                "干预前高压": pre_sbp,
                "干预前低压": pre_dbp,
                "干预后身高": post_height,
                "干预后体重": post_weight,
                "干预后BMI": post_bmi,
                "干预后腰围": post_waist,
                "干预后臀围": post_hip,
                "干预后高压": post_sbp,
                "干预后低压": post_dbp,
                "干预前体感日期": pre_symptom_date,
                "干预前体感子项": pre_symptom_scores,
                "干预前体感总分": pre_total,
                "干预后体感日期": post_symptom_date,
                "干预后体感子项": post_symptom_scores,
                "干预后体感总分": post_total,
                "干预前胰岛素次/天": pre_insulin_times,
                "干预前胰岛素剂量/次": pre_insulin_dose,
                "干预前二甲双胍天/次": pre_metformin_times,
                "干预前二甲双胍剂量/次": pre_metformin_dose,
                "干预前阿卡波糖天/次": pre_acarbose_times,
                "干预前阿卡波糖剂量/次": pre_acarbose_dose,
                "干预后胰岛素次/天": post_insulin_times,
                "干预后胰岛素剂量/次": post_insulin_dose,
                "干预后二甲双胍天/次": post_metformin_times,
                "干预后二甲双胍剂量/次": post_metformin_dose,
                "干预后阿卡波糖天/次": post_acarbose_times,
                "干预后阿卡波糖剂量/次": post_acarbose_dose,
                "干预前其他药物": pre_other_list,
                "干预后其他药物": post_other_list,
                "用药调整干预前日期": drug_pre_date,
                "用药调整干预前用药": drug_pre_med,
                "用药调整干预后日期": drug_post_date,
                "用药调整干预后用药": drug_post_med,
                "减药/停药情况": drug_reduction,
                "干预前生化日期": pre_bio_date,
                "干预前糖化": pre_hba1c,
                "干预前TC": pre_tc,
                "干预前TG": pre_tg,
                "干预前LDL": pre_ldl,
                "干预前HDL": pre_hdl,
                "干预前ALT": pre_alt,
                "干预前AST": pre_ast,
                "干预后生化日期": post_bio_date,
                "干预后糖化": post_hba1c,
                "干预后TC": post_tc,
                "干预后TG": post_tg,
                "干预后LDL": post_ldl,
                "干预后HDL": post_hdl,
                "干预后ALT": post_alt,
                "干预后AST": post_ast,
                "干预前5点日期": pre_glyc_date,
                "干预前FPG": pre_fpg,
                "干预前PG30": pre_pg30,
                "干预前PG60": pre_pg60,
                "干预前PG120": pre_pg120,
                "干预前PG180": pre_pg180,
                "干预后5点日期": post_glyc_date,
                "干预后FPG": post_fpg,
                "干预后PG30": post_pg30,
                "干预后PG60": post_pg60,
                "干预后PG120": post_pg120,
                "干预后PG180": post_pg180,
                "干预方案": intervention_type,
                "干预方案细节": intervention_detail,
                "干预时长(天)": duration_days,
                "干预前7点日期": pre_7_date,
                "干预前早餐前": pre_bf_before,
                "干预前早餐后2h": pre_bf_after,
                "干预前午餐前": pre_lunch_before,
                "干预前午餐后2h": pre_lunch_after,
                "干预前晚餐前": pre_dinner_before,
                "干预前晚餐后2h": pre_dinner_after,
                "干预前睡前": pre_bed,
                "干预后7点日期": post_7_date,
                "干预后早餐前": post_bf_before,
                "干预后早餐后2h": post_bf_after,
                "干预后午餐前": post_lunch_before,
                "干预后午餐后2h": post_lunch_after,
                "干预后晚餐前": post_dinner_before,
                "干预后晚餐后2h": post_dinner_after,
                "干预后睡前": post_bed,
                "项目/医疗地区": project_region,
                "健管师": health_coach,
                "医生": doctor,
                "诊所/门店名称": clinic_name,
                "提交人": submitter,
                "指导健管师": supervisor,
                "备注": remarks
            }

            st.session_state.patients.append(patient_data)
            st.session_state.last_patient = patient_data   # 保存最后一次提交的患者
            st.success(f"✅ 患者 {name} 的信息已成功录入！")

            # 同步至 Google Sheets
            if "gcp_service_account" in st.secrets and "google_sheets" in st.secrets:
                save_to_google_sheets(patient_data)
            else:
                st.info("💡 提示：配置 Google Sheets 后数据将自动云端汇总")

            st.balloons()

    # ===== AI 方案建议（移到表单外部，通过 last_patient 触发） =====
    if st.session_state.get("last_patient"):
        st.markdown("---")
        st.subheader("🤖 AI 智能方案建议")
        patient_for_plan = st.session_state.last_patient
        st.write(f"当前患者：**{patient_for_plan.get('患者姓名', '未知')}**")

        if st.button("生成个体化营养治疗方案", key="gen_plan_btn"):
            with st.spinner("正在分析患者数据并检索知识库..."):
                try:
                    plan = generate_plan(patient_for_plan)
                    st.session_state.ai_plan = plan
                except Exception as e:
                    st.session_state.ai_plan = f"❌ 生成失败：{str(e)}"

        if st.session_state.get("ai_plan"):
            st.text_area("AI 建议", value=st.session_state.ai_plan, height=400)

    # ===== 显示已录入患者列表 =====
    st.subheader("📋 已录入患者列表")
    if len(st.session_state.patients) == 0:
        st.info("暂无患者数据，请使用上方表单录入。")
    else:
        df = pd.DataFrame(st.session_state.patients)
        display_cols = ["患者姓名", "性别", "年龄", "干预前BMI", "干预后BMI", "干预方案", "干预时长(天)", "录入时间"]
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出全部数据为 CSV", data=csv, file_name="patients_data.csv", mime="text/csv")

    # ===== 血糖曲线分析 =====
    st.subheader("📈 患者血糖曲线分析")
    if len(st.session_state.patients) > 0:
        selected_patient_name = st.selectbox(
            "选择患者查看血糖曲线",
            [p["患者姓名"] for p in st.session_state.patients],
            key="glucose_analysis"
        )
        patient = next(p for p in st.session_state.patients if p["患者姓名"] == selected_patient_name)

        pre_values = [
            patient.get("干预前FPG"), patient.get("干预前PG30"),
            patient.get("干预前PG60"), patient.get("干预前PG120"),
            patient.get("干预前PG180")
        ]
        pre_fig, pre_auc = plot_glucose_curve(pre_values, f"{selected_patient_name} - 干预前")

        post_values = [
            patient.get("干预后FPG"), patient.get("干预后PG30"),
            patient.get("干预后PG60"), patient.get("干预后PG120"),
            patient.get("干预后PG180")
        ]
        post_fig, post_auc = plot_glucose_curve(post_values, f"{selected_patient_name} - 干预后")

        col1, col2 = st.columns(2)
        with col1:
            if pre_fig:
                st.plotly_chart(pre_fig, use_container_width=True)
                st.metric("干预前 AUC (mmol/L·h)", pre_auc)
            else:
                st.info("无干预前5点血糖数据")
        with col2:
            if post_fig:
                st.plotly_chart(post_fig, use_container_width=True)
                st.metric("干预后 AUC (mmol/L·h)", post_auc)
            else:
                st.info("无干预后5点血糖数据")
    else:
        st.info("请先录入患者数据，然后即可查看血糖曲线和AUC。")

# ============================================
# 主入口
# ============================================
if __name__ == "__main__":
    patient_info_entry()