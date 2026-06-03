import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import os
import json
import uuid

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
    "langchain_classic": "langchain-classic",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "pypdf": "pypdf",
    "torchvision": "torchvision",
}
for mod, pkg in required_pkgs.items():
    try:
        __import__(mod)
    except ImportError:
        missing_pkgs.append(pkg)

if missing_pkgs:
    st.error(
        "❌ 缺少必要的 Python 包，请在 requirements.txt 中添加以下依赖:\n\n"
        + "\n".join(missing_pkgs)
        + "\n\n然后重新部署应用。"
    )
    st.stop()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import gspread
from google.oauth2 import service_account

# ============================================
# 辅助安全转换函数
# ============================================
def safe_date(val):
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return datetime.strptime(val.strip(), "%Y-%m-%d").date()
        except:
            return None
    return None

def safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

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
    if any(v is None for v in symptom_dict.values()):
        return None
    return sum(symptom_dict.values())

def calculate_duration(start_date, end_date):
    if start_date and end_date:
        delta = end_date - start_date
        return delta.days
    return None

def parse_other_meds(meds_text):
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

def plot_combined_glucose_curve(pre_glucose_values, post_glucose_values, title):
    if not pre_glucose_values or not post_glucose_values:
        return None, None, None
    times = [0, 0.5, 1, 2, 3]
    def compute_auc(vals):
        if not all(vals):
            return None
        auc = 0
        for i in range(len(times)-1):
            auc += (vals[i] + vals[i+1]) / 2 * (times[i+1] - times[i])
        return round(auc, 2)
    pre_auc = compute_auc(pre_glucose_values)
    post_auc = compute_auc(post_glucose_values)
    if pre_auc is None or post_auc is None:
        return None, None, None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=pre_glucose_values,
                             mode='lines+markers', name='干预前', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=times, y=post_glucose_values,
                             mode='lines+markers', name='干预后', line=dict(color='red')))
    all_vals = [v for v in pre_glucose_values + post_glucose_values if v is not None]
    max_y = max(all_vals) if all_vals else 10
    fig.update_layout(
        title=title,
        xaxis_title='时间 (小时)',
        yaxis_title='血糖 (mmol/L)',
        xaxis=dict(tickmode='array', tickvals=times, ticktext=['空腹','0.5h','1h','2h','3h']),
        yaxis=dict(range=[0, max_y * 1.05])
    )
    return fig, pre_auc, post_auc

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

def smart_data_warnings(data_dict):
    w = []
    if data_dict.get("干预前体重"):
        wt = safe_float(data_dict["干预前体重"])
        if wt is not None and wt > 100:
            w.append(f"⚠️ 干预前体重：{wt} kg，已超过 100 kg，请确认单位为公斤（不是斤）。")
    if data_dict.get("干预后体重"):
        wt = safe_float(data_dict["干预后体重"])
        if wt is not None and wt > 100:
            w.append(f"⚠️ 干预后体重：{wt} kg，已超过 100 kg，请确认单位为公斤。")
    if data_dict.get("干预前身高"):
        ht = safe_float(data_dict["干预前身高"])
        if ht is not None and (ht < 100 or ht > 220):
            w.append(f"⚠️ 干预前身高：{ht} cm，数值异常，请确认单位为厘米。")
    if data_dict.get("干预前腰围"):
        wc = safe_float(data_dict["干预前腰围"])
        if wc is not None and wc > 150:
            w.append(f"⚠️ 干预前腰围：{wc} cm，数值偏大，请确认单位。")
    if data_dict.get("干预后腰围"):
        wc = safe_float(data_dict["干预后腰围"])
        if wc is not None and wc > 150:
            w.append(f"⚠️ 干预后腰围：{wc} cm，数值偏大，请确认单位。")
    if data_dict.get("干预前FPG"):
        fpg = safe_float(data_dict["干预前FPG"])
        if fpg is not None and (fpg < 2.0 or fpg > 25.0):
            w.append(f"⚠️ 干预前空腹血糖：{fpg} mmol/L，超出常见范围（2-25 mmol/L），请核实。")
    if data_dict.get("干预后FPG"):
        fpg = safe_float(data_dict["干预后FPG"])
        if fpg is not None and (fpg < 2.0 or fpg > 25.0):
            w.append(f"⚠️ 干预后空腹血糖：{fpg} mmol/L，超出常见范围，请核实。")
    if data_dict.get("干预前高压"):
        sbp = safe_float(data_dict["干预前高压"])
        if sbp is not None and (sbp < 70 or sbp > 250):
            w.append(f"⚠️ 干预前收缩压：{sbp} mmHg，数值异常，请确认。")
    if data_dict.get("干预前低压"):
        dbp = safe_float(data_dict["干预前低压"])
        if dbp is not None and (dbp < 30 or dbp > 150):
            w.append(f"⚠️ 干预前舒张压：{dbp} mmHg，数值异常，请确认。")
    if data_dict.get("干预前ALT"):
        alt = safe_float(data_dict["干预前ALT"])
        if alt is not None and alt > 500:
            w.append(f"⚠️ 干预前 ALT：{alt} U/L，显著升高，请核实。")
    if data_dict.get("干预前AST"):
        ast = safe_float(data_dict["干预前AST"])
        if ast is not None and ast > 500:
            w.append(f"⚠️ 干预前 AST：{ast} U/L，显著升高，请核实。")
    return w

def flatten_dict(d, parent_key='', sep='_'):
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

# ===== Google Sheets 操作（含删除与更新） =====
def get_sheet():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(st.secrets["google_sheets"]["spreadsheet_id"]).sheet1
    return sheet

def find_patient_row_by_id(patient_id):
    sheet = get_sheet()
    all_values = sheet.get_all_values()
    if not all_values:
        return None
    headers = all_values[0]
    try:
        id_col = headers.index("患者ID") + 1
    except ValueError:
        return None
    for i, row in enumerate(all_values[1:], start=2):
        if len(row) >= id_col and row[id_col-1] == patient_id:
            return i
    return None

def delete_patient_row_by_id(patient_id):
    row_num = find_patient_row_by_id(patient_id)
    if row_num is not None:
        sheet = get_sheet()
        sheet.delete_rows(row_num)
        return True
    return False

def save_to_google_sheets(patient_dict):
    try:
        sheet = get_sheet()
        flat = flatten_dict(patient_dict)
        essential_cols = ["提交者ID", "患者姓名", "随访记录"]
        for col in essential_cols:
            if col not in flat:
                flat[col] = ""
        header_row = sheet.row_values(1)
        if not header_row or all(cell == '' for cell in header_row):
            header_to_write = list(flat.keys())
            if len(header_to_write) > 130:
                st.warning("字段过多，将截断部分字段")
                header_to_write = header_to_write[:130]
            sheet.append_row(header_to_write)
            row_data = [flat.get(col, "") for col in header_to_write]
            sheet.append_row(row_data)
        else:
            if "随访记录" not in header_row:
                last_col = len(header_row) + 1
                sheet.update_cell(1, last_col, "随访记录")
                header_row = sheet.row_values(1)
            row_data = [flat.get(col, "") for col in header_row]
            sheet.append_row(row_data)
        st.success("✅ 数据已同步至云端")
    except Exception as e:
        st.warning(f"⚠️ 云端写入失败（数据已保存在本地列表中）: {e}")

def update_patient_in_sheets(patient_dict):
    patient_id = patient_dict.get("患者ID")
    if not patient_id:
        st.error("无法更新：患者ID缺失")
        return False
    deleted = delete_patient_row_by_id(patient_id)
    if not deleted:
        st.info("未找到旧记录，将直接追加新记录")
    save_to_google_sheets(patient_dict)
    return True

def load_patients_from_sheets(submitter_id=None):
    try:
        sheet = get_sheet()
        all_data = sheet.get_all_records()
        if not all_data:
            return []
        if submitter_id:
            all_data = [row for row in all_data if str(row.get("提交者ID", "")) == str(submitter_id)]
        date_fields = [
            "出生日期", "确诊日期", "干预前体感日期", "干预后体感日期",
            "干预前生化日期", "干预后生化日期", "干预前5点日期", "干预后5点日期",
            "干预前7点日期", "干预后7点日期", "用药调整干预前日期", "用药调整干预后日期"
        ]
        numeric_fields = [
            "干预前身高", "干预前体重", "干预前BMI", "干预前腰围", "干预前臀围",
            "干预前高压", "干预前低压", "干预后身高", "干预后体重", "干预后BMI",
            "干预后腰围", "干预后臀围", "干预后高压", "干预后低压",
            "干预前FPG", "干预前PG30", "干预前PG60", "干预前PG120", "干预前PG180",
            "干预后FPG", "干预后PG30", "干预后PG60", "干预后PG120", "干预后PG180",
            "干预前糖化", "干预前TC", "干预前TG", "干预前LDL", "干预前HDL",
            "干预前ALT", "干预前AST", "干预后糖化", "干预后TC", "干预后TG",
            "干预后LDL", "干预后HDL", "干预后ALT", "干预后AST",
            "干预前胰岛素次/天", "干预前胰岛素剂量/次", "干预前二甲双胍天/次",
            "干预前二甲双胍剂量/次", "干预前阿卡波糖天/次", "干预前阿卡波糖剂量/次",
            "干预后胰岛素次/天", "干预后胰岛素剂量/次", "干预后二甲双胍天/次",
            "干预后二甲双胍剂量/次", "干预后阿卡波糖天/次", "干预后阿卡波糖剂量/次",
            "干预前早餐前", "干预前早餐后2h", "干预前午餐前", "干预前午餐后2h",
            "干预前晚餐前", "干预前晚餐后2h", "干预前睡前",
            "干预后早餐前", "干预后早餐后2h", "干预后午餐前", "干预后午餐后2h",
            "干预后晚餐前", "干预后晚餐后2h", "干预后睡前",
            "年龄", "病史年"
        ]
        symptom_items = [
            "口臭", "排便情况", "胃肠道", "四肢麻木", "皮肤瘙痒", "睡眠",
            "视物", "乏力", "多饮", "多食", "多尿", "腰膝酸软", "盗汗情况", "情绪状况"
        ]
        filtered_data = []
        for row in all_data:
            if not row.get("提交者ID"):
                continue
            for nf in numeric_fields:
                if nf in row:
                    row[nf] = safe_float(row[nf])
            for df in date_fields:
                if df in row:
                    row[df] = safe_date(row[df])
            if "干预前体感子项" in row and isinstance(row["干预前体感子项"], str):
                try:
                    row["干预前体感子项"] = json.loads(row["干预前体感子项"])
                except:
                    row["干预前体感子项"] = {}
            if not isinstance(row.get("干预前体感子项"), dict):
                rebuilt = {}
                for item in symptom_items:
                    flat_key = f"干预前体感子项_{item}"
                    if flat_key in row:
                        rebuilt[item] = safe_float(row[flat_key])
                row["干预前体感子项"] = rebuilt if rebuilt else {}
            if "随访记录" in row and isinstance(row["随访记录"], str):
                try:
                    row["随访记录"] = json.loads(row["随访记录"])
                except:
                    row["随访记录"] = []
            if "随访记录" in row and isinstance(row["随访记录"], list):
                for record in row["随访记录"]:
                    for df in date_fields:
                        if df in record:
                            record[df] = safe_date(record[df])
                    for nf in numeric_fields:
                        if nf in record:
                            record[nf] = safe_float(record[nf])
            filtered_data.append(row)
        patient_map = {}
        for row in filtered_data:
            key = (str(row.get("提交者ID", "")), str(row.get("患者姓名", "")))
            if key not in patient_map:
                patient_map[key] = row.copy()
                if not isinstance(patient_map[key].get("随访记录"), list):
                    patient_map[key]["随访记录"] = []
            else:
                existing = patient_map[key]
                new_followups = row.get("随访记录", [])
                if isinstance(new_followups, list):
                    existing_followups = existing.get("随访记录", [])
                    existing_times = {f.get("随访时间") for f in existing_followups}
                    for f in new_followups:
                        if f.get("随访时间") not in existing_times:
                            existing_followups.append(f)
        return list(patient_map.values())
    except Exception as e:
        st.error(f"从云端加载数据失败：{e}")
        return []

# ============================================
# AI 模块（加密加载版，与原始相同）
# ============================================
import pyzipper
import tempfile
import importlib.util

@st.cache_resource(ttl=10800)
def load_encrypted_assets():
    password = st.secrets.get("ASSETS_PASSWORD")
    if not password:
        st.error("❌ 请在 Streamlit Secrets 中设置 ASSETS_PASSWORD")
        st.stop()
    zip_path = os.path.join(os.path.dirname(__file__), "assets.enc.zip")
    if not os.path.exists(zip_path):
        st.error("❌ 未找到加密文件 assets.enc.zip")
        st.stop()
    current_mtime = os.path.getmtime(zip_path)
    if "asset_mtime" not in st.session_state:
        st.session_state.asset_mtime = current_mtime
    elif abs(st.session_state.asset_mtime - current_mtime) > 1e-6:
        st.cache_resource.clear()
        st.session_state.asset_mtime = current_mtime
    tmp_dir = tempfile.mkdtemp()
    try:
        with pyzipper.AESZipFile(zip_path, 'r') as zf:
            zf.setpassword(password.encode())
            zf.extractall(tmp_dir)
    except Exception as e:
        st.error(f"❌ 解密失败，请检查密码或文件格式：{e}")
        st.stop()
    return tmp_dir

def load_prompts():
    tmp_dir = load_encrypted_assets()
    prompts_path = os.path.join(tmp_dir, "prompts.py")
    spec = importlib.util.spec_from_file_location("prompts", prompts_path)
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    return prompts

prompts = load_prompts()
pre_templates = prompts.pre_templates
post_templates = prompts.post_templates

@st.cache_resource
def load_knowledge_base():
    tmp_dir = load_encrypted_assets()
    pdf_dir = os.path.join(tmp_dir, "pdf_data")
    if not os.path.exists(pdf_dir):
        st.error(f"知识库目录 {pdf_dir} 不存在，请检查加密文件")
        return None
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    if not docs:
        st.warning("未检测到任何 PDF 文档，知识库为空")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding)
    return vectordb

def build_rag_chain(vectordb, mode="pre", source=None):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    if mode == "pre":
        template_dict = pre_templates
    else:
        template_dict = post_templates
    template = template_dict.get(source) or template_dict["default"]
    prompt = ChatPromptTemplate.from_template(template)
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("❌ 请在 Streamlit Secrets 中设置 DEEPSEEK_API_KEY")
        st.stop()
    llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key, temperature=0.3)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def generate_plan(patient_combined_data: dict) -> str:
    vectordb = load_knowledge_base()
    if vectordb is None:
        return "❌ 知识库未加载，请检查 PDF 文件"
    source = patient_combined_data.get("项目/医疗地区", "").strip()
    has_post = False
    followups = patient_combined_data.get("随访记录", [])
    if isinstance(followups, list) and followups:
        for fu in followups:
            if any([
                fu.get("干预后FPG"), fu.get("干预后PG120"),
                fu.get("干预后糖化"), fu.get("干预后体重")
            ]):
                has_post = True
                break
    else:
        has_post = any([
            patient_combined_data.get("干预后FPG"),
            patient_combined_data.get("干预后PG120"),
            patient_combined_data.get("干预后糖化"),
            patient_combined_data.get("干预后体重")
        ])
    mode = "post" if has_post else "pre"
    input_text = "请为这位糖尿病患者依据英纽林产品说明制定个体化的营养治疗方案，并预测可能的效果" if mode == "pre" else "请为这位糖尿病患者进行干预前后对比分析，并给出下一阶段的营养建议。"
    rag_chain = build_rag_chain(vectordb, mode, source=source)

    base_data = {
        "height": patient_combined_data.get("干预前身高", "未知"),
        "chronic": patient_combined_data.get("其他慢病", "无"),
        "complications": patient_combined_data.get("并发症", "无"),
    }
    def symptom_dict_to_str(symptom_dict):
        if not symptom_dict or not isinstance(symptom_dict, dict):
            return "无数据"
        items = [f"{name}：{value}分" if value is not None else f"{name}：未知" for name, value in symptom_dict.items()]
        return "；".join(items) if items else "无数据"

    pre_symptom_detail = symptom_dict_to_str(patient_combined_data.get("干预前体感子项", {}))
    pre_data = {
        "pre_weight": patient_combined_data.get("干预前体重", "未知"),
        "pre_bmi": patient_combined_data.get("干预前BMI", "未知"),
        "pre_waist": patient_combined_data.get("干预前腰围", "未知"),
        "pre_sbp": patient_combined_data.get("干预前高压", "未知"),
        "pre_dbp": patient_combined_data.get("干预前低压", "未知"),
        "pre_fpg": patient_combined_data.get("干预前FPG", "未知"),
        "pre_pg2h": patient_combined_data.get("干预前PG120", "未知"),
        "pre_hba1c": patient_combined_data.get("干预前糖化", "未知"),
        "pre_symptom_detail": pre_symptom_detail,
    }
    post_data = {}
    if mode == "post":
        if followups and isinstance(followups, list):
            latest_followup = followups[-1]
            post_symptom = latest_followup.get("干预后体感子项", {})
        else:
            latest_followup = patient_combined_data
            post_symptom = patient_combined_data.get("干预后体感子项", {})
        post_symptom_detail = symptom_dict_to_str(post_symptom)
        post_data = {
            "post_weight": latest_followup.get("干预后体重", "未知"),
            "post_bmi": latest_followup.get("干预后BMI", "未知"),
            "post_waist": latest_followup.get("干预后腰围", "未知"),
            "post_sbp": latest_followup.get("干预后高压", "未知"),
            "post_dbp": latest_followup.get("干预后低压", "未知"),
            "post_fpg": latest_followup.get("干预后FPG", "未知"),
            "post_pg2h": latest_followup.get("干预后PG120", "未知"),
            "post_hba1c": latest_followup.get("干预后糖化", "未知"),
            "post_symptom_detail": post_symptom_detail,
        }
    history_text = ""
    if mode == "post" and followups and isinstance(followups, list):
        lines = []
        for i, fu in enumerate(followups, start=1):
            fu_time = fu.get("随访时间", "")[:10]
            weight = fu.get("干预后体重", "?")
            fpg = fu.get("干预后FPG", "?")
            pg120 = fu.get("干预后PG120", "?")
            hba1c = fu.get("干预后糖化", "?")
            symp_dict = fu.get("干预后体感子项", {})
            if symp_dict:
                symp_items = [f"{k}{v}" for k, v in symp_dict.items() if v is not None]
                symp_str = "，".join(symp_items) if symp_items else "无"
            else:
                symp_str = "无"
            lines.append(f"随访{i} ({fu_time}): 体重{weight}kg, FPG{fpg}, PG120{pg120}, 糖化{hba1c}%, 体感({symp_str})")
        if lines:
            history_text = "；".join(lines)
        else:
            history_text = "暂无历史随访数据"
    else:
        history_text = ""

    fb_symptoms = patient_combined_data.get("使用反馈症状", [])
    fb_symptoms_str = ", ".join(fb_symptoms) if fb_symptoms else "无"
    fb_notes = patient_combined_data.get("使用反馈备注", "") or "无"
    selected_products = patient_combined_data.get("干预方案产品文本", "未指定")
    intervention_detail = patient_combined_data.get("干预方案细节", "未填写")

    invoke_input = {
        "input": input_text,
        **base_data,
        **pre_data,
        **post_data,
        "feedback_symptoms": fb_symptoms_str,
        "feedback_notes": fb_notes,
        "selected_products": selected_products,
        "intervention_detail": intervention_detail,
        "history_followups": history_text,
    }
    result = rag_chain.invoke(invoke_input)
    return result["answer"]

# ============================================
# 主界面
# ============================================
def patient_info_entry():
    st.header("📋 《英纽林糖尿病医学营养治疗真实世界研究》案例收集")

    query_params = st.query_params
    submitter_id = query_params.get("submitter_id", None)
    admin_token = query_params.get("admin_token", None)

    is_admin = False
    if admin_token and "ADMIN_TOKEN" in st.secrets:
        is_admin = (admin_token == st.secrets["ADMIN_TOKEN"])

    if is_admin:
        st.info("🔑 管理员模式：可查看所有提交者的数据")
    elif submitter_id:
        st.success(f"🆔 当前提交者ID：{submitter_id}")
    else:
        st.error("❌ 请使用正确的链接访问（需要提交者ID或管理员令牌）")
        st.stop()

    # 初始化 session 状态
    if "patients" not in st.session_state:
        st.session_state.patients = []
    if "loaded_from_cloud" not in st.session_state:
        st.session_state.loaded_from_cloud = False
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = None          # 'baseline' 或 'followup'
    if "edit_patient_id" not in st.session_state:
        st.session_state.edit_patient_id = None
    if "edit_followup_idx" not in st.session_state:
        st.session_state.edit_followup_idx = None
    if "edit_target_patient" not in st.session_state:
        st.session_state.edit_target_patient = None
    if "edit_target_followup" not in st.session_state:
        st.session_state.edit_target_followup = None

    # 加载云端数据
    if not st.session_state.loaded_from_cloud and "gcp_service_account" in st.secrets:
        with st.spinner("正在从云端加载您的历史数据..."):
            cloud_patients = load_patients_from_sheets() if is_admin else load_patients_from_sheets(submitter_id)
            existing_keys = {(p.get("提交者ID", ""), p["患者姓名"]) for p in st.session_state.patients}
            for cp in cloud_patients:
                key = (cp.get("提交者ID", ""), cp["患者姓名"])
                if key not in existing_keys:
                    st.session_state.patients.append(cp)
                    existing_keys.add(key)
            st.session_state.loaded_from_cloud = True
            st.success(f"已加载 {len(cloud_patients)} 条历史记录")

    if st.button("🔄 重新加载云端数据"):
        st.session_state.loaded_from_cloud = False
        st.rerun()

    # 患者选择（按提交者ID过滤）
    if is_admin:
        all_patient_names = list({p["患者姓名"] for p in st.session_state.patients})
    else:
        all_patient_names = list({p["患者姓名"] for p in st.session_state.patients if p.get("提交者ID") == submitter_id})
    patient_names = ["+ 新增患者"] + all_patient_names

    col_sel, col_edit_btn = st.columns([3,1])
    with col_sel:
        selected_patient_name = st.selectbox("选择已有患者（可自动填充干预前数据）", patient_names, key="selected_patient")
    with col_edit_btn:
        if selected_patient_name != "+ 新增患者" and st.button("✏️ 编辑基线信息"):
            patient = next((p for p in st.session_state.patients if p["患者姓名"] == selected_patient_name), None)
            if patient:
                st.session_state.edit_mode = "baseline"
                st.session_state.edit_patient_id = patient.get("患者ID")
                st.session_state.edit_target_patient = patient.copy()
                st.session_state.edit_target_followup = None
                st.rerun()

    selected_patient_data = None
    if selected_patient_name != "+ 新增患者":
        for p in st.session_state.patients:
            if p["患者姓名"] == selected_patient_name:
                if is_admin or p.get("提交者ID") == submitter_id:
                    selected_patient_data = p
                    break

    # 表单默认值来源（优先级：编辑数据 > 已有患者数据 > 空）
    default_patient = None
    if st.session_state.edit_mode == "baseline" and st.session_state.edit_target_patient:
        default_patient = st.session_state.edit_target_patient
    else:
        default_patient = selected_patient_data

    st.subheader("基本信息输入方式")
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        age_mode = st.radio("年龄输入方式", ["自动计算", "手动输入"], horizontal=True, key="age_mode_radio")
    with col_mode2:
        disease_mode = st.radio("病史年输入方式", ["自动计算", "手动输入"], horizontal=True, key="disease_mode_radio")
    st.markdown("---")

    if "age_manual" not in st.session_state:
        st.session_state.age_manual = 0
    if "disease_manual" not in st.session_state:
        st.session_state.disease_manual = 0.0

    with st.form(key="patient_form", clear_on_submit=True, enter_to_submit=False):
        # 知情同意书
        with st.expander("📜 知情同意书（请阅读后勾选同意）", expanded=True):
            st.markdown("""
            **《英纽林糖尿病医学营养治疗真实世界研究》案例收集项目**

            尊敬的参与者：

            本研究旨在收集真实世界中接受英纽林系列营养产品干预的糖尿病患者案例，用于学术分析和产品优化。  
            所有收集的数据将严格**匿名化处理**，仅用于统计分析，**不会泄露任何个人隐私信息**。

            - 参与本研究完全**自愿**，您可以随时退出，不会影响您获得正常的医疗服务。
            - 数据提交后，研究团队将妥善保管，并仅用于与糖尿病营养治疗相关的科研用途。
            - 如有任何疑问，请联系您的主治医生或研究团队。

            点击下方的复选框，即表示您已**阅读并理解**上述内容，同意将个案数据用于本研究。
            """)
            consent_given = st.checkbox("我已阅读并同意知情同意书", key="consent_checkbox")

        # 1. 用户基本信息
        with st.expander("1️⃣ 用户基本信息", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                name = st.text_input("患者姓名 *", value=default_patient["患者姓名"] if default_patient else "")
                gender = st.selectbox("性别", ["男", "女"], index=["男", "女"].index(default_patient["性别"]) if default_patient and default_patient.get("性别") in ["男", "女"] else 0)
                phone = st.text_input("联系电话", value=default_patient.get("联系电话", "") if default_patient else "")
            with col2:
                default_birth = safe_date(default_patient.get("出生日期")) if default_patient else None
                birth_date = st.date_input("出生日期", value=default_birth, min_value=date(1900,1,1), key="birth")
                auto_age = calculate_age(birth_date)
                if age_mode == "自动计算":
                    age_disabled = True
                    age_value = auto_age if auto_age is not None else 0
                else:
                    age_disabled = False
                    age_value = st.session_state.get("age_manual", 0)
                age_input = st.number_input("年龄（岁）", min_value=0.0, max_value=120.0, step=1.0, value=float(age_value) if age_value is not None else 0.0, disabled=age_disabled, key="age_manual")
                if age_mode == "自动计算":
                    age_input = auto_age
            with col3:
                default_diag = safe_date(default_patient.get("确诊日期")) if default_patient else None
                diagnosis_date = st.date_input("确诊日期/年月日", value=default_diag, min_value=date(1900,1,1), key="diag")
                auto_disease = calculate_disease_years(diagnosis_date)
                if disease_mode == "自动计算":
                    disease_disabled = True
                    disease_value = auto_disease if auto_disease is not None else 0.0
                else:
                    disease_disabled = False
                    disease_value = st.session_state.get("disease_manual", 0.0)
                disease_input = st.number_input("病史/年", min_value=0.0, max_value=80.0, step=0.5, value=float(disease_value) if disease_value is not None else 0.0, disabled=disease_disabled, key="disease_manual")
                if disease_mode == "自动计算":
                    disease_input = auto_disease
            with col4:
                location = st.text_input("所在地/省/市/区", value=default_patient.get("所在地", "") if default_patient else "")
            complications = st.text_input("并发症", value=default_patient.get("并发症", "") if default_patient else "")
            other_chronic = st.text_input("其他慢病", value=default_patient.get("其他慢病", "") if default_patient else "")

        # 案例来源与备注
        with st.expander("2️⃣ 案例来源 & 备注", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                predefined_options = ["医疗", "合作项目", "其他"]
                current_val = default_patient.get("项目/医疗地区", "") if default_patient else ""
                if current_val in predefined_options:
                    dropdown_val = current_val
                    custom_text = ""
                elif current_val:
                    dropdown_val = "其他"
                    custom_text = current_val
                else:
                    dropdown_val = "医疗"
                    custom_text = ""
                project_dropdown = st.selectbox("项目类型", options=predefined_options, index=predefined_options.index(dropdown_val), key="project_dropdown")
                project_custom = st.text_input("如选择“其他”，请填写具体项目名称", value=custom_text, key="project_custom")
                if project_dropdown == "其他":
                    project_region = project_custom.strip()
                else:
                    project_region = project_dropdown
                health_coach = st.text_input("健管师", value=default_patient.get("健管师", "") if default_patient else "", key="health_coach")
                doctor = st.text_input("医生", value=default_patient.get("医生", "") if default_patient else "", key="doctor")
                clinic_name = st.text_input("诊所/门店名称", value=default_patient.get("诊所/门店名称", "") if default_patient else "", key="clinic_name")
            with col2:
                submitter = st.text_input("提交人", value=default_patient.get("提交人", "") if default_patient else "", key="submitter")
                supervisor = st.text_input("指导健管师", value=default_patient.get("指导健管师", "") if default_patient else "", key="supervisor")
            remarks = st.text_area("备注信息", value=default_patient.get("备注", "") if default_patient else "", key="remarks")
        # 干预前数据（编辑基线模式下字段可编辑，否则若已有患者则禁用）
        pre_disabled = (selected_patient_data is not None and st.session_state.edit_mode != "baseline")
        with st.expander("3️⃣ 干预前数据（基本指标、五点血糖、体感、药物、生化、7点血糖）", expanded=False):
            with st.expander("基本指标", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    pre_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0,
                                                value=safe_float(default_patient.get("干预前身高")) if default_patient else None,
                                                step=0.1, key="pre_h", disabled=pre_disabled)
                    pre_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0,
                                                value=safe_float(default_patient.get("干预前体重")) if default_patient else None,
                                                step=0.1, key="pre_w", disabled=pre_disabled)
                with col2:
                    pre_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0,
                                                value=safe_float(default_patient.get("干预前腰围")) if default_patient else None,
                                                step=0.1, disabled=pre_disabled)
                    pre_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0,
                                            value=safe_float(default_patient.get("干预前臀围")) if default_patient else None,
                                            step=0.1, disabled=pre_disabled)
                col1, col2 = st.columns(2)
                with col1:
                    pre_sbp = st.number_input("高压 (mmHg)", min_value=50.0, max_value=250.0,
                                            value=safe_float(default_patient.get("干预前高压")) if default_patient else None,
                                            step=1.0, disabled=pre_disabled)
                with col2:
                    pre_dbp = st.number_input("低压 (mmHg)", min_value=30.0, max_value=150.0,
                                            value=safe_float(default_patient.get("干预前低压")) if default_patient else None,
                                            step=1.0, disabled=pre_disabled)
                pre_bmi = calculate_bmi(pre_height, pre_weight)

            with st.expander("5点血糖", expanded=False):
                pre_glyc_date = st.date_input("检测日期", value=safe_date(default_patient.get("干预前5点日期")) if default_patient else None, min_value=date(1900,1,1), key="pre_glyc_date", disabled=pre_disabled)
                pre_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1,
                                        value=safe_float(default_patient.get("干预前FPG")) if default_patient else None,
                                        key="pre_fpg", disabled=pre_disabled)
                col1, col2 = st.columns(2)
                with col1:
                    pre_pg30 = st.number_input("PG 30min (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前PG30")) if default_patient else None,
                                            key="pre_pg30", disabled=pre_disabled)
                with col2:
                    pre_pg60 = st.number_input("PG 60min (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前PG60")) if default_patient else None,
                                            key="pre_pg60", disabled=pre_disabled)
                col3, col4 = st.columns(2)
                with col3:
                    pre_pg120 = st.number_input("PG 120min (mmol/L)", min_value=0.0, step=0.1,
                                                value=safe_float(default_patient.get("干预前PG120")) if default_patient else None,
                                                key="pre_pg120", disabled=pre_disabled)
                with col4:
                    pre_pg180 = st.number_input("PG 180min (mmol/L)", min_value=0.0, step=0.1,
                                                value=safe_float(default_patient.get("干预前PG180")) if default_patient else None,
                                                key="pre_pg180", disabled=pre_disabled)

            with st.expander("体感指标", expanded=False):
                st.caption("评分标准：0分为最差，10分为最好（即无该症状）")
                pre_symptom_date = st.date_input("录入日期", value=safe_date(default_patient.get("干预前体感日期")) if default_patient else None, min_value=date(1900,1,1), key="symptom_pre_date", disabled=pre_disabled)
                st.caption("如果与五点血糖检测日期相同，可不填")
                pre_scores = default_patient.get("干预前体感子项", {}) if default_patient else {}
                col1, col2, col3 = st.columns(3)
                with col1:
                    pre_halitosis = st.number_input("口臭", 1, 10, value=pre_scores.get("口臭"), key="pre_hal", disabled=pre_disabled)
                    pre_defecation = st.number_input("排便情况", 1, 10, value=pre_scores.get("排便情况"), key="pre_def", disabled=pre_disabled)
                    pre_gi = st.number_input("胃肠道", 1, 10, value=pre_scores.get("胃肠道"), key="pre_gi", disabled=pre_disabled)
                    pre_numbness = st.number_input("四肢麻木", 1, 10, value=pre_scores.get("四肢麻木"), key="pre_num", disabled=pre_disabled)
                with col2:
                    pre_pruritus = st.number_input("皮肤瘙痒", 1, 10, value=pre_scores.get("皮肤瘙痒"), key="pre_pru", disabled=pre_disabled)
                    pre_sleep = st.number_input("睡眠", 1, 10, value=pre_scores.get("睡眠"), key="pre_sleep", disabled=pre_disabled)
                    pre_vision = st.number_input("视物", 1, 10, value=pre_scores.get("视物"), key="pre_vis", disabled=pre_disabled)
                    pre_fatigue = st.number_input("乏力", 1, 10, value=pre_scores.get("乏力"), key="pre_fat", disabled=pre_disabled)
                with col3:
                    pre_polydipsia = st.number_input("多饮", 1, 10, value=pre_scores.get("多饮"), key="pre_polyd", disabled=pre_disabled)
                    pre_polyphagia = st.number_input("多食", 1, 10, value=pre_scores.get("多食"), key="pre_polyp", disabled=pre_disabled)
                    pre_polyuria = st.number_input("多尿", 1, 10, value=pre_scores.get("多尿"), key="pre_polyu", disabled=pre_disabled)
                    pre_lumbago = st.number_input("腰膝酸软", 1, 10, value=pre_scores.get("腰膝酸软"), key="pre_lumb", disabled=pre_disabled)
                col1, col2 = st.columns(2)
                with col1:
                    pre_night_sweat = st.number_input("盗汗情况", 1, 10, value=pre_scores.get("盗汗情况"), key="pre_night", disabled=pre_disabled)
                with col2:
                    pre_mood = st.number_input("情绪状况", 1, 10, value=pre_scores.get("情绪状况"), key="pre_mood", disabled=pre_disabled)
                pre_symptom_scores = {
                    "口臭": pre_halitosis, "排便情况": pre_defecation, "胃肠道": pre_gi,
                    "四肢麻木": pre_numbness, "皮肤瘙痒": pre_pruritus, "睡眠": pre_sleep,
                    "视物": pre_vision, "乏力": pre_fatigue, "多饮": pre_polydipsia,
                    "多食": pre_polyphagia, "多尿": pre_polyuria, "腰膝酸软": pre_lumbago,
                    "盗汗情况": pre_night_sweat, "情绪状况": pre_mood
                }
                pre_total = calculate_symptom_total(pre_symptom_scores)

            with st.expander("糖尿病药物", expanded=False):
                st.subheader("胰岛素")
                col1, col2 = st.columns(2)
                with col1:
                    pre_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_patient.get("干预前胰岛素次/天")) if default_patient else None,
                                                        key="pre_ins_times", disabled=pre_disabled)
                    pre_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_patient.get("干预前胰岛素剂量/次")) if default_patient else None,
                                                        key="pre_ins_dose", disabled=pre_disabled)
                with col2:
                    pre_insulin_type = st.text_input(
                        "胰岛素种类",
                        value=default_patient.get("干预前胰岛素种类", "") if default_patient else "",
                        key="pre_insulin_type",
                        disabled=pre_disabled,
                        placeholder="如：门冬胰岛素、甘精胰岛素等"
                    )
                st.subheader("口服药")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pre_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_patient.get("干预前二甲双胍天/次")) if default_patient else None,
                                                        key="pre_met_times", disabled=pre_disabled)
                    pre_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0.0, step=250.0,
                                                        value=safe_float(default_patient.get("干预前二甲双胍剂量/次")) if default_patient else None,
                                                        key="pre_met_dose", disabled=pre_disabled)
                with col2:
                    pre_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_patient.get("干预前阿卡波糖天/次")) if default_patient else None,
                                                        key="pre_acb_times", disabled=pre_disabled)
                    pre_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0.0, step=50.0,
                                                        value=safe_float(default_patient.get("干预前阿卡波糖剂量/次")) if default_patient else None,
                                                        key="pre_acb_dose", disabled=pre_disabled)
                with col3:
                    pre_other_meds = st.text_area("其他药物", value="" if not default_patient else "", placeholder="每行：药名，每天次数，每次剂量",
                                                key="pre_other_meds", disabled=pre_disabled)

            with st.expander("生化指标", expanded=False):
                pre_bio_date = st.date_input("检测日期", value=safe_date(default_patient.get("干预前生化日期")) if default_patient else None, min_value=date(1900,1,1), key="pre_bio_date", disabled=pre_disabled)
                st.caption("如果与五点血糖检测日期相同，可不填")
                col1, col2 = st.columns(2)
                with col1:
                    pre_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1,
                                                value=safe_float(default_patient.get("干预前糖化")) if default_patient else None,
                                                key="pre_hba1c", disabled=pre_disabled)
                with col2:
                    pre_tg = st.number_input("总甘油三酯 TG (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前TG")) if default_patient else None,
                                            key="pre_tg", disabled=pre_disabled)
                col3, col4, col5 = st.columns(3)
                with col3:
                    pre_tc = st.number_input("总胆固醇 TC (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前TC")) if default_patient else None,
                                            key="pre_tc", disabled=pre_disabled)
                with col4:
                    pre_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前LDL")) if default_patient else None,
                                            key="pre_ldl", disabled=pre_disabled)
                with col5:
                    pre_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_patient.get("干预前HDL")) if default_patient else None,
                                            key="pre_hdl", disabled=pre_disabled)
                col6, col7 = st.columns(2)
                with col6:
                    pre_alt = st.number_input("ALT (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(default_patient.get("干预前ALT")) if default_patient else None,
                                            key="pre_alt", disabled=pre_disabled)
                with col7:
                    pre_ast = st.number_input("AST (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(default_patient.get("干预前AST")) if default_patient else None,
                                            key="pre_ast", disabled=pre_disabled)

            with st.expander("日常7点血糖", expanded=False):
                pre_7_date = st.date_input("检测日期", value=safe_date(default_patient.get("干预前7点日期")) if default_patient else None, min_value=date(1900,1,1), key="pre_7_date", disabled=pre_disabled)
                col_a, col_b = st.columns(2)
                with col_a:
                    pre_bf_before = st.number_input("早餐前", step=0.1, value=safe_float(default_patient.get("干预前早餐前")) if default_patient else None, key="pre_bf_before", disabled=pre_disabled)
                with col_b:
                    pre_bf_after = st.number_input("早餐后2h", step=0.1, value=safe_float(default_patient.get("干预前早餐后2h")) if default_patient else None, key="pre_bf_after", disabled=pre_disabled)
                col_c, col_d = st.columns(2)
                with col_c:
                    pre_lunch_before = st.number_input("午餐前", step=0.1, value=safe_float(default_patient.get("干预前午餐前")) if default_patient else None, key="pre_lunch_before", disabled=pre_disabled)
                with col_d:
                    pre_lunch_after = st.number_input("午餐后2h", step=0.1, value=safe_float(default_patient.get("干预前午餐后2h")) if default_patient else None, key="pre_lunch_after", disabled=pre_disabled)
                col_e, col_f = st.columns(2)
                with col_e:
                    pre_dinner_before = st.number_input("晚餐前", step=0.1, value=safe_float(default_patient.get("干预前晚餐前")) if default_patient else None, key="pre_dinner_before", disabled=pre_disabled)
                with col_f:
                    pre_dinner_after = st.number_input("晚餐后2h", step=0.1, value=safe_float(default_patient.get("干预前晚餐后2h")) if default_patient else None, key="pre_dinner_after", disabled=pre_disabled)
                pre_bed = st.number_input("睡前", step=0.1, value=safe_float(default_patient.get("干预前睡前")) if default_patient else None, key="pre_bed", disabled=pre_disabled)

            # 干预前身体不适描述
            with st.expander("身体不适描述", expanded=False):
                pre_discomfort = st.text_area(
                    "干预前身体不适情况",
                    value=selected_patient_data.get("干预前身体不适", "") if selected_patient_data else "",
                    key="pre_discomfort",
                    disabled=selected_patient_data is not None,
                    placeholder="请描述患者当前的主观不适，如头晕、乏力、口渴等……",
                    height=100
                )
        # 干预后数据（如果正在编辑随访，使用目标随访数据作为默认值）
        default_followup = None
        if st.session_state.edit_mode == "followup" and st.session_state.edit_target_followup:
            default_followup = st.session_state.edit_target_followup

        with st.expander("4️⃣ 干预后数据（基本指标、五点血糖、体感、药物、生化、7点血糖）", expanded=False):
            with st.expander("基本指标", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    post_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=pre_height, step=0.1, key="post_h", disabled=True)
                    post_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0,
                                                value=safe_float(default_followup.get("干预后体重")) if default_followup else None,
                                                step=0.1, key="post_w")
                with col2:
                    post_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0,
                                                value=safe_float(default_followup.get("干预后腰围")) if default_followup else None,
                                                step=0.1, key="post_wc")
                    post_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0,
                                            value=safe_float(default_followup.get("干预后臀围")) if default_followup else None,
                                            step=0.1, key="post_hc")
                col1, col2 = st.columns(2)
                with col1:
                    post_sbp = st.number_input("高压 (mmHg)", min_value=50.0, max_value=250.0,
                                            value=safe_float(default_followup.get("干预后高压")) if default_followup else None,
                                            step=1.0, key="post_sbp")
                with col2:
                    post_dbp = st.number_input("低压 (mmHg)", min_value=30.0, max_value=150.0,
                                            value=safe_float(default_followup.get("干预后低压")) if default_followup else None,
                                            step=1.0, key="post_dbp")
                post_bmi = calculate_bmi(post_height, post_weight)

            with st.expander("5点血糖", expanded=False):
                post_glyc_date = st.date_input("检测日期", value=safe_date(default_followup.get("干预后5点日期")) if default_followup else None, min_value=date(1900,1,1), key="post_glyc_date")
                post_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1,
                                        value=safe_float(default_followup.get("干预后FPG")) if default_followup else None,
                                        key="post_fpg")
                col1, col2 = st.columns(2)
                with col1:
                    post_pg30 = st.number_input("PG 30min", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后PG30")) if default_followup else None,
                                            key="post_30")
                with col2:
                    post_pg60 = st.number_input("PG 60min", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后PG60")) if default_followup else None,
                                            key="post_60")
                col3, col4 = st.columns(2)
                with col3:
                    post_pg120 = st.number_input("PG 120min", min_value=0.0, step=0.1,
                                                value=safe_float(default_followup.get("干预后PG120")) if default_followup else None,
                                                key="post_120")
                with col4:
                    post_pg180 = st.number_input("PG 180min", min_value=0.0, step=0.1,
                                                value=safe_float(default_followup.get("干预后PG180")) if default_followup else None,
                                                key="post_180")

            with st.expander("体感指标", expanded=False):
                st.caption("评分标准：0分为最差，10分为最好（即无该症状）")
                post_symptom_date = st.date_input("录入日期", value=safe_date(default_followup.get("干预后体感日期")) if default_followup else None, min_value=date(1900,1,1), key="symptom_post_date")
                st.caption("如果与五点血糖检测日期相同，可不填")
                post_scores = default_followup.get("干预后体感子项", {}) if default_followup else {}
                col1, col2, col3 = st.columns(3)
                with col1:
                    post_halitosis = st.number_input("口臭", 1, 10, value=post_scores.get("口臭"), key="post_hal")
                    post_defecation = st.number_input("排便情况", 1, 10, value=post_scores.get("排便情况"), key="post_def")
                    post_gi = st.number_input("胃肠道", 1, 10, value=post_scores.get("胃肠道"), key="post_gi")
                    post_numbness = st.number_input("四肢麻木", 1, 10, value=post_scores.get("四肢麻木"), key="post_num")
                with col2:
                    post_pruritus = st.number_input("皮肤瘙痒", 1, 10, value=post_scores.get("皮肤瘙痒"), key="post_pru")
                    post_sleep = st.number_input("睡眠", 1, 10, value=post_scores.get("睡眠"), key="post_sleep")
                    post_vision = st.number_input("视物", 1, 10, value=post_scores.get("视物"), key="post_vis")
                    post_fatigue = st.number_input("乏力", 1, 10, value=post_scores.get("乏力"), key="post_fat")
                with col3:
                    post_polydipsia = st.number_input("多饮", 1, 10, value=post_scores.get("多饮"), key="post_polyd")
                    post_polyphagia = st.number_input("多食", 1, 10, value=post_scores.get("多食"), key="post_polyp")
                    post_polyuria = st.number_input("多尿", 1, 10, value=post_scores.get("多尿"), key="post_polyu")
                    post_lumbago = st.number_input("腰膝酸软", 1, 10, value=post_scores.get("腰膝酸软"), key="post_lumb")
                col1, col2 = st.columns(2)
                with col1:
                    post_night_sweat = st.number_input("盗汗情况", 1, 10, value=post_scores.get("盗汗情况"), key="post_night")
                with col2:
                    post_mood = st.number_input("情绪状况", 1, 10, value=post_scores.get("情绪状况"), key="post_mood")
                post_symptom_scores = {
                    "口臭": post_halitosis, "排便情况": post_defecation, "胃肠道": post_gi,
                    "四肢麻木": post_numbness, "皮肤瘙痒": post_pruritus, "睡眠": post_sleep,
                    "视物": post_vision, "乏力": post_fatigue, "多饮": post_polydipsia,
                    "多食": post_polyphagia, "多尿": post_polyuria, "腰膝酸软": post_lumbago,
                    "盗汗情况": post_night_sweat, "情绪状况": post_mood
                }
                post_total = calculate_symptom_total(post_symptom_scores)

            with st.expander("糖尿病药物", expanded=False):
                st.subheader("胰岛素")
                col1, col2 = st.columns(2)
                with col1:
                    post_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_followup.get("干预后胰岛素次/天")) if default_followup else None,
                                                        key="post_ins_times")
                    post_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_followup.get("干预后胰岛素剂量/次")) if default_followup else None,
                                                        key="post_ins_dose")
                with col2:
                    post_insulin_type = st.text_input(
                        "胰岛素种类",
                        value=default_followup.get("干预后胰岛素种类", "") if default_followup else "",
                        key="post_insulin_type",
                        placeholder="如：门冬胰岛素、甘精胰岛素等"
                    )
                st.subheader("口服药")
                col1, col2, col3 = st.columns(3)
                with col1:
                    post_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_followup.get("干预后二甲双胍天/次")) if default_followup else None,
                                                        key="post_met_times")
                    post_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0.0, step=250.0,
                                                        value=safe_float(default_followup.get("干预后二甲双胍剂量/次")) if default_followup else None,
                                                        key="post_met_dose")
                with col2:
                    post_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(default_followup.get("干预后阿卡波糖天/次")) if default_followup else None,
                                                        key="post_acb_times")
                    post_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0.0, step=50.0,
                                                        value=safe_float(default_followup.get("干预后阿卡波糖剂量/次")) if default_followup else None,
                                                        key="post_acb_dose")
                with col3:
                    post_other_meds = st.text_area("其他药物", value="", placeholder="每行：药名，每天次数，每次剂量",
                                                key="post_other_meds")

            with st.expander("生化指标", expanded=False):
                post_bio_date = st.date_input("检测日期", value=safe_date(default_followup.get("干预后生化日期")) if default_followup else None, min_value=date(1900,1,1), key="post_bio_date")
                st.caption("如果与五点血糖检测日期相同，可不填")
                col1, col2 = st.columns(2)
                with col1:
                    post_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1,
                                                value=safe_float(default_followup.get("干预后糖化")) if default_followup else None,
                                                key="post_hba1c")
                with col2:
                    post_tg = st.number_input("总甘油三酯 TG (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后TG")) if default_followup else None,
                                            key="post_tg")
                col3, col4, col5 = st.columns(3)
                with col3:
                    post_tc = st.number_input("总胆固醇 TC (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后TC")) if default_followup else None,
                                            key="post_tc")
                with col4:
                    post_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后LDL")) if default_followup else None,
                                            key="post_ldl")
                with col5:
                    post_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(default_followup.get("干预后HDL")) if default_followup else None,
                                            key="post_hdl")
                col6, col7 = st.columns(2)
                with col6:
                    post_alt = st.number_input("ALT (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(default_followup.get("干预后ALT")) if default_followup else None,
                                            key="post_alt")
                with col7:
                    post_ast = st.number_input("AST (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(default_followup.get("干预后AST")) if default_followup else None,
                                            key="post_ast")

            with st.expander("日常7点血糖", expanded=False):
                post_7_date = st.date_input("检测日期", value=safe_date(default_followup.get("干预后7点日期")) if default_followup else None, min_value=date(1900,1,1), key="post_7_date")
                col_a, col_b = st.columns(2)
                with col_a:
                    post_bf_before = st.number_input("早餐前", step=0.1, value=safe_float(default_followup.get("干预后早餐前")) if default_followup else None, key="post_bf_before")
                with col_b:
                    post_bf_after = st.number_input("早餐后2h", step=0.1, value=safe_float(default_followup.get("干预后早餐后2h")) if default_followup else None, key="post_bf_after")
                col_c, col_d = st.columns(2)
                with col_c:
                    post_lunch_before = st.number_input("午餐前", step=0.1, value=safe_float(default_followup.get("干预后午餐前")) if default_followup else None, key="post_lunch_before")
                with col_d:
                    post_lunch_after = st.number_input("午餐后2h", step=0.1, value=safe_float(default_followup.get("干预后午餐后2h")) if default_followup else None, key="post_lunch_after")
                col_e, col_f = st.columns(2)
                with col_e:
                    post_dinner_before = st.number_input("晚餐前", step=0.1, value=safe_float(default_followup.get("干预后晚餐前")) if default_followup else None, key="post_dinner_before")
                with col_f:
                    post_dinner_after = st.number_input("晚餐后2h", step=0.1, value=safe_float(default_followup.get("干预后晚餐后2h")) if default_followup else None, key="post_dinner_after")
                post_bed = st.number_input("睡前", step=0.1, value=safe_float(default_followup.get("干预后睡前")) if default_followup else None, key="post_bed")

            # 干预后身体不适描述
            with st.expander("身体不适描述", expanded=False):
                post_discomfort = st.text_area(
                    "干预后身体不适情况",
                    value=None,   # 干预后默认为空
                    key="post_discomfort",
                    placeholder="请描述干预后身体不适的变化情况……",
                    height=100
                )
        # 干预方案与使用反馈
        with st.expander("5️⃣ 干预方案与使用反馈", expanded=False):
            intervention_products = st.multiselect("营养治疗产品（可多选）", ["畅快/清畅", "纽畅/唐畅", "纽畅B/唐畅B", "其他营养治疗"],
                                                  default=default_followup.get("干预方案产品", []) if default_followup else [])
            other_product_name = ""
            if "其他营养治疗" in intervention_products:
                other_product_name = st.text_input("请输入‘其他营养治疗’的具体名称", key="other_product_name")
            intervention_detail = st.text_area("干预方案细节（用量/用法/周期/搭配方式等）", placeholder="例如：畅快 每日1次 每次1包……",
                                              value=default_followup.get("干预方案细节", "") if default_followup else "", key="intervention_detail")
            st.markdown("---")
            st.subheader("使用反馈")
            feedback_symptoms = st.multiselect("常见不良反应", ["腹泻", "便秘", "腹胀", "恶心", "腹痛", "过敏/皮疹", "其他"],
                                             default=default_followup.get("使用反馈症状", []) if default_followup else [], key="feedback_symptoms")
            feedback_notes = st.text_area("反馈详细描述", placeholder="……", value=default_followup.get("使用反馈备注", "") if default_followup else "", key="feedback_notes")

        # 用药调整情况
        with st.expander("6️⃣ 用药调整情况", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                drug_pre_date = st.date_input("干预前日期", value=safe_date(default_patient.get("用药调整干预前日期")) if default_patient else None, min_value=date(1900,1,1), key="drug_pre_date", disabled=pre_disabled)
                drug_pre_med = st.text_area("干预前用药 (可简述)", value=default_patient.get("用药调整干预前用药", "") if default_patient else "", key="drug_pre_med", disabled=pre_disabled)
            with col2:
                drug_post_date = st.date_input("干预后日期", value=safe_date(default_followup.get("用药调整干预后日期")) if default_followup else None, min_value=date(1900,1,1), key="drug_post_date")
                drug_post_med = st.text_area("干预后用药 (可简述)", value=default_followup.get("用药调整干预后用药", "") if default_followup else "", key="drug_post_med")
            drug_reduction = st.selectbox("减药/停药", ["无变化", "减剂量", "减种类", "停用所有口服", "其他"],
                                         index=["无变化", "减剂量", "减种类", "停用所有口服", "其他"].index(default_followup.get("减药/停药情况", "无变化")) if default_followup else 0, key="drug_reduction")



        DUPLICATE_CHECK_FIELDS = [
            "干预后5点日期", "干预后体重", "干预后FPG", "干预后PG120", "干预后糖化",
            "干预后体感总分",
        ]

        submitted = st.form_submit_button("✅ 提交并保存患者信息")
        if submitted:
            if not name:
                st.error("患者姓名不能为空")
                st.stop()
            if not consent_given:
                st.error("❌ 请先阅读并勾选知情同意书，否则无法提交数据。")
                st.stop()

            # 异常预警
            check_dict = {
                "干预前体重": pre_weight, "干预后体重": post_weight,
                "干预前身高": pre_height,
                "干预前腰围": pre_waist, "干预后腰围": post_waist,
                "干预前FPG": pre_fpg, "干预后FPG": post_fpg,
                "干预前高压": pre_sbp, "干预前低压": pre_dbp,
                "干预后高压": post_sbp, "干预后低压": post_dbp,
                "干预前ALT": pre_alt, "干预前AST": pre_ast,
                "干预后ALT": post_alt, "干预后AST": post_ast,
            }
            warnings = smart_data_warnings(check_dict)
            if warnings:
                for w in warnings:
                    st.warning(w)

            pre_other_list = parse_other_meds(pre_other_meds)
            post_other_list = parse_other_meds(post_other_meds)

            final_products = intervention_products.copy()
            if "其他营养治疗" in final_products and other_product_name.strip():
                idx = final_products.index("其他营养治疗")
                final_products[idx] = other_product_name.strip()

            followup_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            if post_glyc_date is not None:
                followup_time = post_glyc_date.isoformat()

            new_followup = {
                "随访时间": followup_time,
                "干预后身高": post_height,
                "干预后体重": post_weight,
                "干预后BMI": post_bmi,
                "干预后腰围": post_waist,
                "干预后臀围": post_hip,
                "干预后高压": post_sbp,
                "干预后低压": post_dbp,
                "干预后体感日期": post_symptom_date,
                "干预后体感子项": post_symptom_scores,
                "干预后体感总分": post_total,
                "干预后胰岛素种类": post_insulin_type,
                "干预后胰岛素次/天": post_insulin_times,
                "干预后胰岛素剂量/次": post_insulin_dose,
                "干预后二甲双胍天/次": post_metformin_times,
                "干预后二甲双胍剂量/次": post_metformin_dose,
                "干预后阿卡波糖天/次": post_acarbose_times,
                "干预后阿卡波糖剂量/次": post_acarbose_dose,
                "干预后其他药物": post_other_list,
                "用药调整干预后日期": drug_post_date,
                "用药调整干预后用药": drug_post_med,
                "减药/停药情况": drug_reduction,
                "干预后生化日期": post_bio_date,
                "干预后糖化": post_hba1c,
                "干预后TC": post_tc,
                "干预后TG": post_tg,
                "干预后LDL": post_ldl,
                "干预后HDL": post_hdl,
                "干预后ALT": post_alt,
                "干预后AST": post_ast,
                "干预后5点日期": post_glyc_date,
                "干预后FPG": post_fpg,
                "干预后PG30": post_pg30,
                "干预后PG60": post_pg60,
                "干预后PG120": post_pg120,
                "干预后PG180": post_pg180,
                "干预后7点日期": post_7_date,
                "干预后早餐前": post_bf_before,
                "干预后早餐后2h": post_bf_after,
                "干预后午餐前": post_lunch_before,
                "干预后午餐后2h": post_lunch_after,
                "干预后晚餐前": post_dinner_before,
                "干预后晚餐后2h": post_dinner_after,
                "干预后睡前": post_bed,
                "干预方案产品": final_products,
                "干预方案产品文本": "，".join(final_products),
                "干预方案细节": intervention_detail,
                "使用反馈症状": feedback_symptoms,
                "使用反馈备注": feedback_notes,
                "干预后身体不适": post_discomfort,
            }

            def is_empty_followup(fu):
                key_post_fields = ["干预后体重", "干预后FPG", "干预后PG120", "干预后糖化",
                                   "干预后高压", "干预后低压"]
                for k in key_post_fields:
                    val = fu.get(k)
                    if val is not None and val != "" and val != 0.0:
                        return False
                if fu.get("干预后体感总分") is not None:
                    return False
                if fu.get("减药/停药情况", "无变化") != "无变化":
                    return False
                if fu.get("干预方案细节", "").strip() or fu.get("使用反馈备注", "").strip():
                    return False
                return True

            empty_followup = is_empty_followup(new_followup)

            # ===== 分情况处理 =====
            if st.session_state.edit_mode == "baseline" and st.session_state.edit_patient_id:
                # 编辑基线：更新现有患者的基线字段，保留随访记录
                target_patient = st.session_state.edit_target_patient
                if target_patient:
                    target_patient.update({
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
                        "干预前体感日期": pre_symptom_date,
                        "干预前体感子项": pre_symptom_scores,
                        "干预前体感总分": pre_total,
                        "干预前胰岛素种类": pre_insulin_type,
                        "干预前胰岛素次/天": pre_insulin_times,
                        "干预前胰岛素剂量/次": pre_insulin_dose,
                        "干预前二甲双胍天/次": pre_metformin_times,
                        "干预前二甲双胍剂量/次": pre_metformin_dose,
                        "干预前阿卡波糖天/次": pre_acarbose_times,
                        "干预前阿卡波糖剂量/次": pre_acarbose_dose,
                        "干预前其他药物": pre_other_list,
                        "用药调整干预前日期": drug_pre_date,
                        "用药调整干预前用药": drug_pre_med,
                        "干预前生化日期": pre_bio_date,
                        "干预前糖化": pre_hba1c,
                        "干预前TC": pre_tc,
                        "干预前TG": pre_tg,
                        "干预前LDL": pre_ldl,
                        "干预前HDL": pre_hdl,
                        "干预前ALT": pre_alt,
                        "干预前AST": pre_ast,
                        "干预前5点日期": pre_glyc_date,
                        "干预前FPG": pre_fpg,
                        "干预前PG30": pre_pg30,
                        "干预前PG60": pre_pg60,
                        "干预前PG120": pre_pg120,
                        "干预前PG180": pre_pg180,
                        "干预前7点日期": pre_7_date,
                        "干预前早餐前": pre_bf_before,
                        "干预前早餐后2h": pre_bf_after,
                        "干预前午餐前": pre_lunch_before,
                        "干预前午餐后2h": pre_lunch_after,
                        "干预前晚餐前": pre_dinner_before,
                        "干预前晚餐后2h": pre_dinner_after,
                        "干预前睡前": pre_bed,
                        "项目/医疗地区": project_region,
                        "健管师": health_coach,
                        "医生": doctor,
                        "诊所/门店名称": clinic_name,
                        "提交人": submitter,
                        "指导健管师": supervisor,
                        "备注": remarks,
                        "干预前身体不适": pre_discomfort,
                    })
                    update_patient_in_sheets(target_patient)
                    # 更新本地 patients 列表
                    for i, p in enumerate(st.session_state.patients):
                        if p.get("患者ID") == target_patient["患者ID"]:
                            st.session_state.patients[i] = target_patient
                            break
                    st.success("✅ 患者基线信息已更新")
                    st.session_state.edit_mode = None
                    st.session_state.edit_patient_id = None
                    st.session_state.edit_target_patient = None
                    # 刷新当前选中患者数据
                    selected_patient_data = target_patient
                    st.rerun()

            elif st.session_state.edit_mode == "followup" and st.session_state.edit_patient_id is not None and st.session_state.edit_followup_idx is not None:
                # 编辑随访：替换随访列表中的指定条目
                target_patient = st.session_state.edit_target_patient
                if target_patient:
                    followups_list = target_patient.get("随访记录", [])
                    idx = st.session_state.edit_followup_idx
                    if 0 <= idx < len(followups_list):
                        followups_list[idx] = new_followup
                        target_patient["随访记录"] = followups_list
                        update_patient_in_sheets(target_patient)
                        for i, p in enumerate(st.session_state.patients):
                            if p.get("患者ID") == target_patient["患者ID"]:
                                st.session_state.patients[i] = target_patient
                                break
                        st.success(f"✅ 第 {idx+1} 次随访记录已更新")
                    else:
                        st.error("随访索引错误")
                    st.session_state.edit_mode = None
                    st.session_state.edit_patient_id = None
                    st.session_state.edit_followup_idx = None
                    st.session_state.edit_target_patient = None
                    st.session_state.edit_target_followup = None
                    selected_patient_data = target_patient
                    st.rerun()

            else:
                # 新增模式：新建患者 或 为已有患者新增随访
                if selected_patient_name != "+ 新增患者" and selected_patient_data:
                    # 为已有患者新增随访
                    if not empty_followup:
                        last_fu = selected_patient_data["随访记录"][-1] if selected_patient_data["随访记录"] else None
                        duplicate = False
                        if last_fu:
                            duplicate = all(
                                last_fu.get(f) == new_followup.get(f)
                                for f in DUPLICATE_CHECK_FIELDS
                            )
                        if duplicate:
                            st.info("📝 本次干预后数据与最近一次随访完全相同，已跳过重复记录，仅更新其他信息。")
                        else:
                            selected_patient_data["随访记录"].append(new_followup)
                            st.success(f"✅ 已为 {selected_patient_name} 添加新的随访记录")
                            update_patient_in_sheets(selected_patient_data)
                    else:
                        st.info("📝 未填写任何干预后数据，仅更新基线信息（如有修改）。")
                        update_patient_in_sheets(selected_patient_data)
                    # 更新本地列表
                    for i, p in enumerate(st.session_state.patients):
                        if p.get("患者ID") == selected_patient_data.get("患者ID"):
                            st.session_state.patients[i] = selected_patient_data
                            break
                else:
                    # 新建患者
                    base_followups = [] if empty_followup else [new_followup]
                    new_patient_id = str(uuid.uuid4())
                    base_data = {
                        "患者ID": new_patient_id,
                        "提交者ID": submitter_id if not is_admin else "admin",
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
                        "干预前体感日期": pre_symptom_date,
                        "干预前体感子项": pre_symptom_scores,
                        "干预前体感总分": pre_total,
                        "干预前胰岛素种类": pre_insulin_type,
                        "干预前胰岛素次/天": pre_insulin_times,
                        "干预前胰岛素剂量/次": pre_insulin_dose,
                        "干预前二甲双胍天/次": pre_metformin_times,
                        "干预前二甲双胍剂量/次": pre_metformin_dose,
                        "干预前阿卡波糖天/次": pre_acarbose_times,
                        "干预前阿卡波糖剂量/次": pre_acarbose_dose,
                        "干预前其他药物": pre_other_list,
                        "用药调整干预前日期": drug_pre_date,
                        "用药调整干预前用药": drug_pre_med,
                        "干预前生化日期": pre_bio_date,
                        "干预前糖化": pre_hba1c,
                        "干预前TC": pre_tc,
                        "干预前TG": pre_tg,
                        "干预前LDL": pre_ldl,
                        "干预前HDL": pre_hdl,
                        "干预前ALT": pre_alt,
                        "干预前AST": pre_ast,
                        "干预前5点日期": pre_glyc_date,
                        "干预前FPG": pre_fpg,
                        "干预前PG30": pre_pg30,
                        "干预前PG60": pre_pg60,
                        "干预前PG120": pre_pg120,
                        "干预前PG180": pre_pg180,
                        "干预前7点日期": pre_7_date,
                        "干预前早餐前": pre_bf_before,
                        "干预前早餐后2h": pre_bf_after,
                        "干预前午餐前": pre_lunch_before,
                        "干预前午餐后2h": pre_lunch_after,
                        "干预前晚餐前": pre_dinner_before,
                        "干预前晚餐后2h": pre_dinner_after,
                        "干预前睡前": pre_bed,
                        "项目/医疗地区": project_region,
                        "健管师": health_coach,
                        "医生": doctor,
                        "诊所/门店名称": clinic_name,
                        "提交人": submitter,
                        "指导健管师": supervisor,
                        "备注": remarks,
                        "随访记录": base_followups
                    }
                    st.session_state.patients.append(base_data)
                    save_to_google_sheets(base_data)
                    if not empty_followup:
                        st.success(f"✅ 患者 {name} 已新增并录入首次随访数据")
                    else:
                        st.success(f"✅ 患者 {name} 的基线信息已保存，干预后数据待下次录入")
                    selected_patient_data = base_data
                st.balloons()

    # ===== AI 方案建议（使用当前选中的患者数据） =====
    if selected_patient_data:
        st.markdown("---")
        st.subheader("🤖 AI 智能方案建议")
        st.write(f"当前患者：**{selected_patient_data.get('患者姓名', '未知')}**")
        if st.button("生成个体化营养治疗方案", key="gen_plan_btn"):
            with st.spinner("正在分析..."):
                try:
                    plan = generate_plan(selected_patient_data)
                    st.session_state.ai_plan = plan
                    # 可选：将方案存入患者数据并同步云端（非强制）
                    selected_patient_data["AI方案"] = plan
                    # update_patient_in_sheets(selected_patient_data)  # 如需保存方案可取消注释
                except Exception as e:
                    st.session_state.ai_plan = f"❌ 生成失败：{str(e)}"
        if st.session_state.get("ai_plan"):
            st.text_area("AI 建议", value=st.session_state.ai_plan, height=600)

    # ===== 图表分析（基于当前选中的患者） =====
    if selected_patient_data and selected_patient_name != "+ 新增患者":
        patient = selected_patient_data
        st.subheader("📈 血糖曲线分析")
        display_mode = st.radio("展示模式", ["单次随访对比", "全部随访展示"], horizontal=True, key="glucose_display_mode")
        pre_values = [patient.get("干预前FPG"), patient.get("干预前PG30"),
                      patient.get("干预前PG60"), patient.get("干预前PG120"),
                      patient.get("干预前PG180")]
        followups = patient.get("随访记录", [])
        pre_date_str = ""
        if patient.get("干预前5点日期"):
            pre_date_str = f" ({patient['干预前5点日期'].isoformat()})"

        # 辅助函数（带参考线和异常标记）
        def get_marker_color(value):
            if value is None:
                return 'lightgray'
            if value > 11.1:
                return 'red'
            elif value < 3.9:
                return 'purple'
            else:
                return 'blue'

        def plot_glucose_curve_with_ref(values, title):
            if not all(values):
                return None, None
            times = [0, 0.5, 1, 2, 3]
            auc = 0
            for i in range(len(times)-1):
                auc += (values[i] + values[i+1]) / 2 * (times[i+1] - times[i])
            fig = go.Figure()
            colors = [get_marker_color(v) for v in values]
            fig.add_trace(go.Scatter(
                x=times, y=values, mode='lines+markers', name=title,
                line=dict(color='blue'), marker=dict(color=colors, size=8)
            ))
            # 三条原有参考线
            fig.add_hline(y=3.9, line_dash="dash", line_color="purple",
                        annotation_text="低血糖阈值 (3.9)", annotation_position="bottom right")
            fig.add_hline(y=7.8, line_dash="dash", line_color="red",
                        annotation_text="餐后2h正常上限 (7.8)", annotation_position="bottom right")
            fig.add_hline(y=11.1, line_dash="dash", line_color="orange",
                        annotation_text="糖尿病诊断阈值 (11.1)", annotation_position="top right")
            # 新增高风险警示线 16.7
            fig.add_hline(y=16.7, line_dash="dash", line_color="darkred",
                        annotation_text="高风险警示线 (16.7)", annotation_position="top right")
            # 动态 y 轴范围，确保包含 16.7
            y_max = max(values) if max(values) > 11.1 else 12
            y_min = min(values) if min(values) < 3.9 else 0
            y_max = max(y_max, 16.7)  # 保证警示线可见
            fig.update_layout(
                title=title,
                xaxis_title='时间 (小时)',
                yaxis_title='血糖 (mmol/L)',
                xaxis=dict(tickmode='array', tickvals=times, ticktext=['空腹','0.5h','1h','2h','3h']),
                yaxis=dict(range=[y_min - 0.5, y_max + 0.5])
            )
            return fig, round(auc, 2)

        def plot_combined_glucose_curve_with_ref(pre_vals, post_vals, title):
            if not pre_vals or not post_vals:
                return None, None, None
            times = [0, 0.5, 1, 2, 3]
            def compute_auc(vals):
                if not all(vals):
                    return None
                auc = 0
                for i in range(len(times)-1):
                    auc += (vals[i] + vals[i+1]) / 2 * (times[i+1] - times[i])
                return round(auc, 2)
            pre_auc = compute_auc(pre_vals)
            post_auc = compute_auc(post_vals)
            if pre_auc is None or post_auc is None:
                return None, None, None
            fig = go.Figure()
            pre_colors = [get_marker_color(v) for v in pre_vals]
            fig.add_trace(go.Scatter(
                x=times, y=pre_vals, mode='lines+markers', name='干预前',
                line=dict(color='blue'), marker=dict(color=pre_colors, size=8)
            ))
            post_colors = [get_marker_color(v) for v in post_vals]
            fig.add_trace(go.Scatter(
                x=times, y=post_vals, mode='lines+markers', name='干预后',
                line=dict(color='red'), marker=dict(color=post_colors, size=8)
            ))
            # 三条原有参考线
            fig.add_hline(y=3.9, line_dash="dash", line_color="purple",
                        annotation_text="低血糖阈值 (3.9)", annotation_position="bottom right")
            fig.add_hline(y=7.8, line_dash="dash", line_color="red",
                        annotation_text="餐后2h正常上限 (7.8)", annotation_position="bottom right")
            fig.add_hline(y=11.1, line_dash="dash", line_color="orange",
                        annotation_text="糖尿病诊断阈值 (11.1)", annotation_position="top right")
            # 新增高风险警示线 16.7
            fig.add_hline(y=16.7, line_dash="dash", line_color="darkred",
                        annotation_text="高风险警示线 (16.7)", annotation_position="top right")
            all_vals = [v for v in pre_vals + post_vals if v is not None]
            y_max = max(all_vals) if all_vals else 20
            y_min = min(all_vals) if all_vals else 0
            y_max = max(y_max, 16.7)  # 保证警示线可见
            fig.update_layout(
                title=title,
                xaxis_title='时间 (小时)',
                yaxis_title='血糖 (mmol/L)',
                xaxis=dict(tickmode='array', tickvals=times, ticktext=['空腹','0.5h','1h','2h','3h']),
                yaxis=dict(range=[y_min - 0.5, y_max + 0.5])
            )
            return fig, pre_auc, post_auc

        if display_mode == "单次随访对比":
            followup_options = ["未选择"] + [f"第{i+1}次随访 ({r.get('随访时间', '')})" for i, r in enumerate(followups)]
            selected_followup_idx = st.selectbox("选择随访记录", range(len(followup_options)),
                                                format_func=lambda x: followup_options[x], key="single_followup")
            if selected_followup_idx == 0:
                if all(pre_values):
                    fig, auc = plot_glucose_curve_with_ref(pre_values, f"干预前血糖曲线{pre_date_str}")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("AUC (mmol/L·h)", auc)
                else:
                    st.info("无干预前5点血糖数据")
            else:
                record = followups[selected_followup_idx - 1]
                post_values = [record.get("干预后FPG"), record.get("干预后PG30"),
                               record.get("干预后PG60"), record.get("干预后PG120"),
                               record.get("干预后PG180")]
                fig, pre_auc, post_auc = plot_combined_glucose_curve_with_ref(
                    pre_values, post_values,
                    f"{selected_patient_name} - 干预前后对比"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    col1, col2 = st.columns(2)
                    with col1: st.metric("干预前 AUC", pre_auc)
                    with col2: st.metric("干预后 AUC", post_auc)
                    # 编辑随访按钮
                    if st.button("✏️ 编辑本次随访", key="edit_followup_btn"):
                        st.session_state.edit_mode = "followup"
                        st.session_state.edit_patient_id = patient.get("患者ID")
                        st.session_state.edit_followup_idx = selected_followup_idx - 1
                        st.session_state.edit_target_patient = patient
                        st.session_state.edit_target_followup = record
                        st.rerun()
                else:
                    st.info("该次随访数据不完整，无法绘图")
                    
        else:
            # 全部随访展示（多点对比）
            valid_followups = []
            for i, rec in enumerate(followups):
                post_vals = [rec.get("干预后FPG"), rec.get("干预后PG30"),
                             rec.get("干预后PG60"), rec.get("干预后PG120"),
                             rec.get("干预后PG180")]
                if all(post_vals):
                    valid_followups.append((i, rec, post_vals))
            if not valid_followups and not all(pre_values):
                st.info("暂无完整的五点血糖数据")
            else:
                fig = go.Figure()
                times = [0, 0.5, 1, 2, 3]
                # 干预前曲线
                if all(pre_values):
                    pre_colors = [get_marker_color(v) for v in pre_values]
                    fig.add_trace(go.Scatter(
                        x=times, y=pre_values, mode='lines+markers',
                        name=f'干预前{pre_date_str}', line=dict(color='blue', width=3),
                        marker=dict(color=pre_colors, size=8)
                    ))
                # 各次随访曲线
                colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                for idx, (i, rec, post_vals) in enumerate(valid_followups):
                    color = colors[idx % len(colors)]
                    followup_date = rec.get("随访时间", f"第{i+1}次")
                    post_colors = [get_marker_color(v) for v in post_vals]
                    fig.add_trace(go.Scatter(
                        x=times, y=post_vals, mode='lines+markers',
                        name=f'随访{idx+1} ({followup_date})', line=dict(color=color),
                        marker=dict(color=post_colors, size=8)
                    ))
                # ========== 添加参考线 ==========
                fig.add_hline(y=3.9, line_dash="dash", line_color="purple",
                              annotation_text="低血糖阈值 (3.9)", annotation_position="bottom right")
                fig.add_hline(y=7.8, line_dash="dash", line_color="red",
                              annotation_text="餐后2h正常上限 (7.8)", annotation_position="bottom right")
                fig.add_hline(y=11.1, line_dash="dash", line_color="orange",
                              annotation_text="糖尿病诊断阈值 (11.1)", annotation_position="top right")
                # 新增高风险警示线 16.7
                fig.add_hline(y=16.7, line_dash="dash", line_color="darkred",
                              annotation_text="高风险警示线 (16.7)", annotation_position="top right")
                # ========== 动态调整 y 轴范围 ==========
                all_vals = []
                if all(pre_values): all_vals.extend(pre_values)
                for _, _, vals in valid_followups: all_vals.extend(vals)
                y_max = max(all_vals) if all_vals else 20
                y_min = min(all_vals) if all_vals else 0
                y_max = max(y_max, 16.7)   # 保证高风险线可见
                fig.update_layout(
                    title=f"{selected_patient_name} - 多点血糖对比",
                    xaxis_title='时间 (小时)', yaxis_title='血糖 (mmol/L)',
                    xaxis=dict(tickmode='array', tickvals=times, ticktext=['空腹','0.5h','1h','2h','3h']),
                    yaxis=dict(range=[y_min - 0.5, y_max + 0.5])
                )
                st.plotly_chart(fig, use_container_width=True)

                # 计算并展示 AUC（原有代码不变）
                def compute_auc(vals):
                    auc = 0
                    for j in range(len(times)-1):
                        auc += (vals[j] + vals[j+1]) / 2 * (times[j+1] - times[j])
                    return round(auc, 2)
                auc_data = []
                if all(pre_values):
                    auc_data.append({"记录": "干预前", "AUC": compute_auc(pre_values)})
                for i, rec, post_vals in valid_followups:
                    auc_data.append({"记录": f"随访{i+1}", "AUC": compute_auc(post_vals)})
                st.dataframe(pd.DataFrame(auc_data), use_container_width=True)

        # 7点血糖汇总表（可选）
        show_7point = st.checkbox("📋 显示日常7点血糖数据汇总", value=False)
        if show_7point:
            st.subheader("📋 日常7点血糖数据")
            time_labels = ["早餐前", "早餐后2h", "午餐前", "午餐后2h", "晚餐前", "晚餐后2h", "睡前"]
            rows = []
            pre_7 = [
                patient.get("干预前早餐前"), patient.get("干预前早餐后2h"),
                patient.get("干预前午餐前"), patient.get("干预前午餐后2h"),
                patient.get("干预前晚餐前"), patient.get("干预前晚餐后2h"),
                patient.get("干预前睡前")
            ]
            rows.append(("干预前", pre_7))
            for i, rec in enumerate(patient.get("随访记录", []), start=1):
                post_7 = [
                    rec.get("干预后早餐前"), rec.get("干预后早餐后2h"),
                    rec.get("干预后午餐前"), rec.get("干预后午餐后2h"),
                    rec.get("干预后晚餐前"), rec.get("干预后晚餐后2h"),
                    rec.get("干预后睡前")
                ]
                fu_label = f"随访{i} ({rec.get('随访时间', '')[:10]})"
                rows.append((fu_label, post_7))
            df_7point = pd.DataFrame({label: vals for label, vals in rows}, index=time_labels).T
            st.dataframe(df_7point, use_container_width=True)

        # 体感评分对比（可选）
        show_symptom = st.checkbox("📊 显示单项体感评分对比", value=False)
        if show_symptom:
            st.subheader("📊 单项体感评分对比")
            pre_symptom = patient.get("干预前体感子项", {})
            if not pre_symptom:
                st.info("无干预前体感数据")
            else:
                items = list(pre_symptom.keys())
                pre_vals = [pre_symptom.get(item) for item in items]
                pre_date = patient.get("干预前体感日期") or patient.get("干预前5点日期")
                pre_label = "干预前" + (f" ({pre_date.isoformat()})" if pre_date else "")
                fig = go.Figure()
                fig.add_trace(go.Bar(name=pre_label, x=items, y=pre_vals, marker_color="blue"))
                colors_bar = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                valid_followups = [f for f in patient.get("随访记录", []) if f.get("干预后体感子项")]
                for idx, rec in enumerate(valid_followups):
                    post_symptom = rec.get("干预后体感子项", {})
                    post_vals = [post_symptom.get(item) for item in items]
                    color = colors_bar[idx % len(colors_bar)]
                    fu_date = rec.get("随访时间") or rec.get("干预后体感日期") or rec.get("干预后5点日期")
                    fu_date_str = fu_date.isoformat() if isinstance(fu_date, date) else str(fu_date) if fu_date else ""
                    followup_label = f"随访{idx+1} ({fu_date_str})" if fu_date_str else f"随访{idx+1}"
                    fig.add_trace(go.Bar(name=followup_label, x=items, y=post_vals, marker_color=color))
                fig.update_layout(
                    title=f"{selected_patient_name} - 体感评分对比",
                    xaxis_title="体感项目",
                    yaxis_title="评分 (0最差, 10最好)",
                    barmode='group',
                    yaxis=dict(range=[0, 10])
                )
                st.plotly_chart(fig, use_container_width=True)

        # 生化指标趋势（可选）
        show_biochem = st.checkbox("📊 显示单项生化指标变化趋势，需消耗较多资源，耐心等待", value=False)
        if show_biochem:
            st.subheader("🧪 单项生化指标变化趋势")
            bio_fields = [
                ("糖化血红蛋白", "干预前糖化", "干预后糖化", "%"),
                ("总胆固醇 (TC)", "干预前TC", "干预后TC", "mmol/L"),
                ("甘油三酯 (TG)", "干预前TG", "干预后TG", "mmol/L"),
                ("低密度脂蛋白 (LDL-C)", "干预前LDL", "干预后LDL", "mmol/L"),
                ("高密度脂蛋白 (HDL-C)", "干预前HDL", "干预后HDL", "mmol/L"),
                ("谷丙转氨酶 (ALT)", "干预前ALT", "干预后ALT", "U/L"),
                ("谷草转氨酶 (AST)", "干预前AST", "干预后AST", "U/L"),
            ]
            followups_list = patient.get("随访记录", [])
            pre_bio_date = patient.get("干预前生化日期") or patient.get("干预前5点日期")
            pre_bio_label = "干预前" + (f" ({pre_bio_date.isoformat()})" if pre_bio_date else "")
            timepoints = [pre_bio_label]
            for i, r in enumerate(followups_list):
                fu_date = r.get("随访时间") or r.get("干预后生化日期") or r.get("干预后5点日期")
                fu_date_str = fu_date.isoformat() if isinstance(fu_date, date) else str(fu_date) if fu_date else ""
                label = f"随访{i+1}\n({fu_date_str[:10]})" if fu_date_str else f"随访{i+1}"
                timepoints.append(label)
            for field_name, pre_key, post_key, unit in bio_fields:
                pre_val = patient.get(pre_key)
                post_vals = [rec.get(post_key) for rec in followups_list]
                all_vals = [pre_val] + post_vals
                if all(v is None for v in all_vals):
                    continue
                x_indices = list(range(len(timepoints)))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_indices, y=all_vals, mode='lines+markers',
                                         name=field_name, line=dict(color='royalblue')))
                fig.update_layout(
                    title=f"{selected_patient_name} - {field_name}",
                    xaxis=dict(tickmode='array', tickvals=x_indices, ticktext=timepoints),
                    yaxis_title=unit,
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 请在上方选择或新增一位患者，即可查看图表分析")

if __name__ == "__main__":
    patient_info_entry()