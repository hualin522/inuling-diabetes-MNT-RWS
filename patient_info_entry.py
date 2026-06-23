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
            # 支持中英文逗号
            parts = line.replace('，', ',').split(',')
            if len(parts) >= 1:
                name = parts[0].strip()
                times = 0
                dose = 0
                if len(parts) >= 2:
                    try:
                        times = float(parts[1].strip())
                    except:
                        pass
                if len(parts) >= 3:
                    try:
                        dose = float(parts[2].strip())
                    except:
                        pass
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
        if isinstance(v, (dict, list)):
            # dict 和 list 均序列化为 JSON 字符串（与 Google Sheets 存储格式一致）
            def json_serial(obj):
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            items.append((new_key, json.dumps(v, default=json_serial, ensure_ascii=False)))
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
            # 空表：写表头 + 第一行数据（用 append_row 自动扩展网格）
            header_to_write = list(flat.keys())
            if len(header_to_write) > 130:
                st.warning("字段过多，将截断部分字段")
                header_to_write = header_to_write[:130]
            sheet.append_row(header_to_write)
            row_data = [flat.get(col, "") for col in header_to_write]
            sheet.append_row(row_data)
        else:
            # 补全表头：将 flat 中有但表头缺失的列批量追加到表头末尾
            existing = set(header_row)
            new_cols = []
            for col in flat.keys():
                if col not in existing:
                    new_cols.append(col)
                    existing.add(col)
            if "随访记录" not in header_row:
                new_cols.append("随访记录")
            if new_cols:
                start_col = len(header_row) + 1
                end_col = start_col + len(new_cols) - 1
                range_str = f'{gspread.utils.rowcol_to_a1(1, start_col)}:{gspread.utils.rowcol_to_a1(1, end_col)}'
                sheet.update(range_str, [new_cols])
                header_row.extend(new_cols)
            # 严格按表头顺序构建行数据
            row_data = [flat.get(col, "") for col in header_row]
            # 用 append_row 追加到末尾（gspread 会自动扩展表格行数上限）
            # 不用 update(A{next_row}) 因为可能超出 Google Sheets 当前网格上限
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
        # 改用 get_all_values 手动解析，避免 get_all_records 在表头含重复空字符串时报错
        all_values = sheet.get_all_values()
        if not all_values or len(all_values) < 2:
            return []
        headers = all_values[0]
        # 去重表头：重复名只保留首次出现，后续用 _n 后缀；空表头用 col_N 命名
        seen = {}
        uniq_headers = []
        for i, h in enumerate(headers):
            h = str(h).strip() if h is not None else ""
            if not h:
                h = f"col_{i}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            uniq_headers.append(h)
        all_data = [dict(zip(uniq_headers, row)) for row in all_values[1:]]
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
                    # 反序列化随访记录中的体感子项（与顶层干预前体感子项逻辑一致）
                    if "干预后体感子项" in record and isinstance(record["干预后体感子项"], str):
                        try:
                            record["干预后体感子项"] = json.loads(record["干预后体感子项"])
                        except:
                            record["干预后体感子项"] = {}
                    if not isinstance(record.get("干预后体感子项"), dict):
                        rebuilt_post = {}
                        for item in symptom_items:
                            flat_key = f"干预后体感子项_{item}"
                            if flat_key in record:
                                rebuilt_post[item] = safe_float(record[flat_key])
                        if rebuilt_post:
                            record["干预后体感子项"] = rebuilt_post
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

# 明文 prompts 覆盖路径：若存在则优先加载，便于在不重新加密 assets.enc.zip 的情况下
# 测试修改后的 prompts（如新增的 prediction_template）。否则回退到加密包内的 prompts.py。
PROMPTS_OVERRIDE_PATH = os.path.join(os.path.dirname(__file__), "encrypted_assets", "prompts_v2.py")

def load_prompts():
    if os.path.exists(PROMPTS_OVERRIDE_PATH):
        spec = importlib.util.spec_from_file_location("prompts_v2", PROMPTS_OVERRIDE_PATH)
        prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts)
        return prompts
    tmp_dir = load_encrypted_assets()
    prompts_path = os.path.join(tmp_dir, "prompts.py")
    spec = importlib.util.spec_from_file_location("prompts", prompts_path)
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    return prompts

prompts = load_prompts()
pre_templates = prompts.pre_templates
post_templates = prompts.post_templates
# 干预效果预测模板（基于相似患者结局）；若覆盖文件未提供则为 None，预测功能将优雅降级
prediction_template = getattr(prompts, "prediction_template", None)

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

    # 干预前模式：检索相似患者，为模板第4部分提供数据
    similar_ref = "（暂无相似患者数据）"
    outcome_stats = "（暂无统计）"
    if mode == "pre":
        try:
            similar = find_similar_patients(patient_combined_data, top_k=10)
            if similar:
                similar_ref = format_similar_patients_context(similar)
                agg = _aggregate_similar_outcomes(similar)
                agg_lines = []
                for _label, _stats in agg.items():
                    agg_lines.append(
                        f"{_label}: n={_stats['n']}, 均值={_stats['mean']}, "
                        f"中位数={_stats['median']}, 范围=[{_stats['min']}, {_stats['max']}]"
                    )
                outcome_stats = "\n".join(agg_lines) if agg_lines else "（结局数据不足）"
            else:
                similar_ref = "（未检索到含干预后结局数据的相似患者，请基于临床经验谨慎预测）"
                outcome_stats = "（无）"
        except Exception as _e:
            similar_ref = f"（相似患者检索失败：{_e}，请基于临床经验预测）"
            outcome_stats = "（无）"

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
        "similar_patients_reference": similar_ref,
        "outcome_statistics": outcome_stats,
    }
    result = rag_chain.invoke(invoke_input)
    return result["answer"]

# ============================================
# 相似患者检索与干预效果预测模块
# ============================================
# 设计说明：
# 1. 从 Google Sheets 全量患者中检索与新患者"干预前"特征最相似的历史患者；
# 2. 仅纳入至少有一次含干预后数据的随访记录的患者（作为效果预测的参照）；
# 3. 数值特征按临床合理范围做 min-max 归一化至 [0,1]，消除量纲差异；
# 4. 缺失维度采用 nan-aware 加权均方距离（只在双方均有值的维度上计算），
#    避免缺失值偏置；
# 5. 性别 / 项目地区作为类别特征，不匹配时施加固定惩罚；
# 6. 检索过程对全量数据向量化（numpy 广播），数万条规模下毫秒级完成；
#    特征矩阵随 similarity_pool 缓存在 session_state，仅在"刷新"时重建。
# 注：不引入 FAISS——数万规模下 numpy 暴力检索已足够（<50ms），
#     且免去了索引构建/重建的复杂度。若数据增长至 10 万+ 可再考虑 FAISS。

# 数值特征：(字段名, 最小值, 最大值, 权重)
NUMERIC_FEATURE_SPEC = [
    ("年龄",              0, 100, 1.0),
    ("病史年",            0, 40,  0.8),
    ("干预前BMI",        15, 45,  1.0),
    ("干预前FPG",         3, 25,  1.5),
    ("干预前PG120",       3, 30,  1.5),
    ("干预前糖化",        4, 15,  1.2),
    ("干预前高压",       80, 250, 0.4),
    ("干预前低压",       50, 150, 0.4),
    ("干预前腰围",       50, 150, 0.6),
    ("干预前TG",          0, 10,  0.3),
    ("干预前TC",          2, 12,  0.3),
    ("干预前LDL",         0, 8,   0.3),
    ("干预前HDL",         0, 3,   0.3),
    ("干预前ALT",         0, 500, 0.2),
    ("干预前AST",         0, 500, 0.2),
]

# 体感子项（每项 1-10 分），归一化除以 10
SYMPTOM_FEATURE_ITEMS = [
    "口臭", "排便情况", "胃肠道", "四肢麻木", "皮肤瘙痒", "睡眠", "视物",
    "乏力", "多饮", "多食", "多尿", "腰膝酸软", "盗汗情况", "情绪状况",
]
SYMPTOM_ITEM_WEIGHT = 0.12  # 14 项合计 ≈ 1.68（次要于血糖/人口学特征）

# 类别特征不匹配惩罚（加到最终距离上）
GENDER_MISMATCH_PENALTY = 1.0
REGION_MISMATCH_PENALTY = 0.5


def _sim_to_float(val):
    """安全转 float，失败/空返回 None。"""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _has_post_outcome(patient):
    """判断患者是否含有可用的干预后结局数据（作为相似参照的前提）。"""
    followups = patient.get("随访记录", [])
    if not isinstance(followups, list) or not followups:
        return False
    for fu in followups:
        if any(_sim_to_float(fu.get(k)) is not None for k in
               ["干预后FPG", "干预后PG120", "干预后糖化", "干预后体重"]):
            return True
    return False


def _extract_feature_vector(patient):
    """
    从单个患者字典抽取归一化特征向量与类别标记。
    返回 (numeric_vec, symptom_vec, gender, region)：
      - numeric_vec: 长度=len(NUMERIC_FEATURE_SPEC) 的 np.array，缺失填 np.nan
      - symptom_vec: 长度=len(SYMPTOM_FEATURE_ITEMS) 的 np.array，缺失填 np.nan
      - gender / region: 字符串
    """
    numeric = []
    for name, lo, hi, _w in NUMERIC_FEATURE_SPEC:
        raw = _sim_to_float(patient.get(name))
        if raw is None:
            numeric.append(np.nan)
        else:
            raw = min(max(raw, lo), hi)          # 夹逼到临床范围
            numeric.append((raw - lo) / (hi - lo))
    symptom_dict = patient.get("干预前体感子项", {}) or {}
    if not isinstance(symptom_dict, dict):
        symptom_dict = {}
    symptom = []
    for item in SYMPTOM_FEATURE_ITEMS:
        raw = _sim_to_float(symptom_dict.get(item))
        if raw is None:
            symptom.append(np.nan)
        else:
            raw = min(max(raw, 0), 10)
            symptom.append(raw / 10.0)
    gender = str(patient.get("性别", "")).strip()
    region = str(patient.get("项目/医疗地区", "")).strip()
    return (np.array(numeric, dtype=float),
            np.array(symptom, dtype=float), gender, region)


def _build_feature_matrix(patients):
    """
    构建全量患者的特征矩阵（仅含可作为参照的患者）。
    返回 (numeric_matrix, symptom_matrix, genders, regions, ref_indices)。
    """
    ref_indices = []
    numeric_rows = []
    symptom_rows = []
    genders = []
    regions = []
    for i, p in enumerate(patients):
        if not _has_post_outcome(p):
            continue
        n, s, g, r = _extract_feature_vector(p)
        numeric_rows.append(n)
        symptom_rows.append(s)
        genders.append(g)
        regions.append(r)
        ref_indices.append(i)
    if not ref_indices:
        return (np.empty((0, len(NUMERIC_FEATURE_SPEC))),
                np.empty((0, len(SYMPTOM_FEATURE_ITEMS))),
                [], [], [])
    return (np.vstack(numeric_rows), np.vstack(symptom_rows),
            genders, regions, ref_indices)


def _weighted_nan_distance(query_num, query_sym, mat_num, mat_sym,
                           query_gender, query_region, genders, regions):
    """
    计算查询向量到矩阵每一行的加权距离（nan-aware）。
    返回长度=矩阵行数的 np.array，越小越相似。
    """
    weights_num = np.array([w for *_x, w in NUMERIC_FEATURE_SPEC], dtype=float)
    # 数值维度：只在双方均非 nan 的维度上计算加权均方差
    if mat_num.shape[0] > 0:
        q_num = query_num[np.newaxis, :]                 # (1, D)
        diff2 = (mat_num - q_num) ** 2                    # (N, D)
        valid = ~np.isnan(mat_num) & ~np.isnan(q_num)
        weighted = weights_num[np.newaxis, :] * diff2
        weighted = np.where(valid, weighted, 0.0)
        sum_w = np.sum(weights_num[np.newaxis, :] * valid, axis=1)
        sum_w[sum_w == 0] = np.nan                        # 全缺维度 → 后续过滤
        dist_num2 = np.sum(weighted, axis=1) / sum_w      # 加权均方
    else:
        dist_num2 = np.array([])

    # 体感维度：同法
    w_sym = np.full(len(SYMPTOM_FEATURE_ITEMS), SYMPTOM_ITEM_WEIGHT, dtype=float)
    if mat_sym.shape[0] > 0:
        q_sym = query_sym[np.newaxis, :]
        diff2s = (mat_sym - q_sym) ** 2
        valid_s = ~np.isnan(mat_sym) & ~np.isnan(q_sym)
        weighteds = w_sym[np.newaxis, :] * diff2s
        weighteds = np.where(valid_s, weighteds, 0.0)
        sum_ws = np.sum(w_sym[np.newaxis, :] * valid_s, axis=1)
        sum_ws[sum_ws == 0] = np.nan
        dist_sym2 = np.sum(weighteds, axis=1) / sum_ws
    else:
        dist_sym2 = np.array([])

    # 合并数值与体感（两者均 nan-aware 均方，相加后开根）
    if dist_num2.size > 0:
        combined = (np.where(np.isnan(dist_num2), 0.0, dist_num2)
                    + np.where(np.isnan(dist_sym2), 0.0, dist_sym2))
        both_invalid = np.isnan(dist_num2) & np.isnan(dist_sym2)
        combined = np.where(both_invalid, np.inf, combined)
        dist = np.sqrt(combined)
    else:
        dist = np.array([])

    # 类别惩罚
    if dist.size > 0:
        gender_pen = np.where(
            np.array([bool(g) and g == query_gender for g in genders]),
            0.0, GENDER_MISMATCH_PENALTY,
        )
        region_pen = np.where(
            np.array([bool(r) and r == query_region for r in regions]),
            0.0, REGION_MISMATCH_PENALTY,
        )
        dist = dist + gender_pen + region_pen
    return dist


def get_similarity_pool():
    """获取（并缓存于 session_state）全量患者数据用于相似检索。"""
    if not st.session_state.get("similarity_pool_loaded", False):
        with st.spinner("正在加载全量患者数据用于相似患者检索（仅首次，约数秒）..."):
            st.session_state.similarity_pool = load_patients_from_sheets(submitter_id=None)
            st.session_state.similarity_pool_loaded = True
    return st.session_state.similarity_pool


def find_similar_patients(query_patient, top_k=10):
    """
    从缓存的全量患者池检索 top_k 最相似患者。
    返回 list[dict]：每项含 patient / distance / similarity / rank。
    """
    pool = get_similarity_pool()
    if not pool:
        return []
    num_mat, sym_mat, genders, regions, ref_idx = _build_feature_matrix(pool)
    if not ref_idx:
        return []
    q_num, q_sym, q_g, q_r = _extract_feature_vector(query_patient)
    dist = _weighted_nan_distance(q_num, q_sym, num_mat, sym_mat,
                                  q_g, q_r, genders, regions)
    valid_mask = np.isfinite(dist)
    if not valid_mask.any():
        return []
    valid_indices = np.where(valid_mask)[0]
    order = np.argsort(dist[valid_mask])
    results = []
    for rank, j in enumerate(order[:top_k], start=1):
        real_j = valid_indices[j]
        d = float(dist[real_j])
        sim = 1.0 / (1.0 + d)                             # 距离→相似度 0~1
        results.append({
            "patient": pool[ref_idx[real_j]],
            "distance": round(d, 4),
            "similarity": round(sim, 4),
            "rank": rank,
        })
    return results


def _symptom_total(sd):
    """计算体感子项总分。"""
    if not isinstance(sd, dict):
        return None
    vals = [_sim_to_float(v) for v in sd.values()]
    vals = [v for v in vals if v is not None]
    return round(sum(vals), 1) if vals else None


def _anon_patient_summary(patient, rank, distance, similarity):
    """生成单个相似患者的脱敏摘要（不含姓名/电话）。"""
    def fmt(v):
        return "—" if v is None or v == "" else v
    def delta(pre, post):
        a = _sim_to_float(pre); b = _sim_to_float(post)
        if a is None or b is None:
            return "—"
        return f"{b - a:+.1f}"
    fu_list = patient.get("随访记录", []) or []
    last_fu = fu_list[-1] if fu_list else {}
    pre_sym = patient.get("干预前体感子项", {}) or {}
    post_sym = last_fu.get("干预后体感子项", {}) or {}
    pre_total = _symptom_total(pre_sym)
    post_total = _symptom_total(post_sym)
    lines = [
        f"【相似患者 {rank}】 相似度={similarity:.2f}（距离={distance:.3f}）",
        f"  基线: 性别={fmt(patient.get('性别'))}, 年龄={fmt(patient.get('年龄'))}岁, "
        f"BMI={fmt(patient.get('干预前BMI'))}, 病史={fmt(patient.get('病史年'))}年, "
        f"地区={fmt(patient.get('项目/医疗地区'))}",
        f"  干预前血糖: FPG={fmt(patient.get('干预前FPG'))}, "
        f"PG120={fmt(patient.get('干预前PG120'))}, 糖化={fmt(patient.get('干预前糖化'))}%",
        f"  干预前血压: {fmt(patient.get('干预前高压'))}/{fmt(patient.get('干预前低压'))} mmHg, "
        f"腰围={fmt(patient.get('干预前腰围'))}cm",
        f"  干预前体感总分: {fmt(pre_total)}",
        f"  并发症: {fmt(patient.get('并发症'))}; 其他慢病: {fmt(patient.get('其他慢病'))}",
        f"  随访次数: {len(fu_list)}",
    ]
    if last_fu:
        d_total = ("—" if pre_total is None or post_total is None
                   else f"{post_total - pre_total:+.1f}")
        lines.append(
            f"  最近随访(干预后): 体重={fmt(last_fu.get('干预后体重'))}kg "
            f"(Δ={delta(patient.get('干预前体重'), last_fu.get('干预后体重'))}kg), "
            f"BMI={fmt(last_fu.get('干预后BMI'))}, "
            f"FPG={fmt(last_fu.get('干预后FPG'))} "
            f"(Δ={delta(patient.get('干预前FPG'), last_fu.get('干预后FPG'))}), "
            f"PG120={fmt(last_fu.get('干预后PG120'))} "
            f"(Δ={delta(patient.get('干预前PG120'), last_fu.get('干预后PG120'))}), "
            f"糖化={fmt(last_fu.get('干预后糖化'))}% "
            f"(Δ={delta(patient.get('干预前糖化'), last_fu.get('干预后糖化'))}%)"
        )
        lines.append(f"  干预后体感总分: {fmt(post_total)} (Δ={d_total})")
        lines.append(f"  减药/停药: {fmt(last_fu.get('减药/停药情况'))}")
        lines.append(f"  使用产品: {fmt(last_fu.get('干预方案产品文本'))}")
    return "\n".join(lines)


def format_similar_patients_context(similar_list):
    """将相似患者列表格式化为 AI 上下文字符串。"""
    if not similar_list:
        return "（未检索到含干预后结局数据的相似患者）"
    blocks = [_anon_patient_summary(item["patient"], item["rank"],
                                    item["distance"], item["similarity"])
              for item in similar_list]
    header = f"共检索到 {len(similar_list)} 位相似患者（按相似度降序，已脱敏）：\n"
    return header + "\n\n".join(blocks)


def _aggregate_similar_outcomes(similar_list):
    """汇总相似患者关键结局变化量(Δ=干预后-干预前)的统计量。"""
    import statistics
    keys = [
        ("ΔFPG", "干预前FPG", "干预后FPG"),
        ("ΔPG120", "干预前PG120", "干预后PG120"),
        ("Δ糖化", "干预前糖化", "干预后糖化"),
        ("Δ体重(kg)", "干预前体重", "干预后体重"),
        ("ΔBMI", "干预前BMI", "干预后BMI"),
        ("Δ腰围(cm)", "干预前腰围", "干预后腰围"),
        ("Δ高压", "干预前高压", "干预后高压"),
        ("Δ低压", "干预前低压", "干预后低压"),
    ]
    agg = {}
    for label, pre_k, post_k in keys:
        deltas = []
        for item in similar_list:
            p = item["patient"]
            fu = (p.get("随访记录") or [None])
            fu = fu[-1] if fu else {}
            a = _sim_to_float(p.get(pre_k))
            b = _sim_to_float(fu.get(post_k)) if fu else None
            if a is not None and b is not None:
                deltas.append(b - a)
        if deltas:
            agg[label] = {
                "n": len(deltas),
                "mean": round(statistics.mean(deltas), 2),
                "median": round(statistics.median(deltas), 2),
                "min": round(min(deltas), 2),
                "max": round(max(deltas), 2),
            }
    return agg


def predict_intervention_effect(new_patient, similar_patients):
    """
    基于相似患者的干预结局，调用 AI 预测新患者的干预效果。
    返回 AI 生成的预测报告文本（直接 LLM 调用，不走 RAG 检索）。
    """
    if not similar_patients:
        return ("❌ 未检索到含干预后结局数据的相似患者，无法进行效果预测。"
                "建议先积累更多含干预后随访数据的案例。")
    if prediction_template is None:
        return "❌ 预测模板未加载，请检查 prompts_v2.py 是否包含 prediction_template。"
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "❌ 未配置 DEEPSEEK_API_KEY。"

    similar_context = format_similar_patients_context(similar_patients)
    agg = _aggregate_similar_outcomes(similar_patients)
    agg_lines = []
    for label, stats in agg.items():
        agg_lines.append(
            f"{label}: n={stats['n']}, 均值={stats['mean']}, 中位数={stats['median']}, "
            f"范围=[{stats['min']}, {stats['max']}]"
        )
    agg_text = "\n".join(agg_lines) if agg_lines else "（结局数据不足）"

    def symptom_dict_to_str(sd):
        if not sd or not isinstance(sd, dict):
            return "无数据"
        items = [f"{k}：{v}分" for k, v in sd.items() if _sim_to_float(v) is not None]
        return "；".join(items) if items else "无数据"

    prompt = ChatPromptTemplate.from_template(prediction_template)
    llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key, temperature=0.4)
    chain = prompt | llm
    invoke_input = {
        "gender": new_patient.get("性别", "未知"),
        "age": new_patient.get("年龄", "未知"),
        "disease_years": new_patient.get("病史年", "未知"),
        "region": new_patient.get("项目/医疗地区", "未知"),
        "height": new_patient.get("干预前身高", "未知"),
        "pre_weight": new_patient.get("干预前体重", "未知"),
        "pre_bmi": new_patient.get("干预前BMI", "未知"),
        "pre_waist": new_patient.get("干预前腰围", "未知"),
        "pre_sbp": new_patient.get("干预前高压", "未知"),
        "pre_dbp": new_patient.get("干预前低压", "未知"),
        "pre_fpg": new_patient.get("干预前FPG", "未知"),
        "pre_pg2h": new_patient.get("干预前PG120", "未知"),
        "pre_hba1c": new_patient.get("干预前糖化", "未知"),
        "pre_symptom_detail": symptom_dict_to_str(new_patient.get("干预前体感子项", {})),
        "chronic": new_patient.get("其他慢病", "无"),
        "complications": new_patient.get("并发症", "无"),
        "similar_patients_reference": similar_context,
        "outcome_statistics": agg_text,
        "similar_count": len(similar_patients),
    }
    try:
        resp = chain.invoke(invoke_input)
        return resp.content
    except Exception as e:
        return f"❌ 预测生成失败：{e}"


# ============================================
# 主界面
# ============================================
def patient_info_entry():
    st.header("📋 《糖尿病医学营养治疗真实世界研究》案例收集")

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

    # 根据 form_reset 标志动态设置 index
    if st.session_state.get("form_reset", False):
        default_index = 0          # 强制选中“+ 新增患者”
        st.session_state.form_reset = False   # 立即清除标志
    else:
        default_index = None       # 保持之前的选择（记忆）

    col_sel, col_edit_btn = st.columns([3,1])
    with col_sel:
        selected_patient_name = st.selectbox(
            "选择已有患者（可自动填充干预前数据）",
            patient_names,
            index=default_index,
            key="selected_patient"
        )
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
    
        # 在获取 selected_patient_data 之后，添加以下同步逻辑
    if selected_patient_data and selected_patient_name != "+ 新增患者" and st.session_state.edit_mode != "baseline":
        # 避免重复同步（根据患者ID判断）
        current_patient_id = selected_patient_data.get("患者ID")
        last_synced_id = st.session_state.get("last_synced_pre_patient_id")
        if current_patient_id != last_synced_id:
            # 定义干预前字段到 session_state key 的映射
            pre_field_mapping = {
                # 基本指标
                "pre_h": ("干预前身高", safe_float),
                "pre_w": ("干预前体重", safe_float),
                "pre_wc": ("干预前腰围", safe_float),
                "pre_hc": ("干预前臀围", safe_float),
                "pre_sbp": ("干预前高压", safe_float),
                "pre_dbp": ("干预前低压", safe_float),
                # 五点血糖
                "pre_fpg": ("干预前FPG", safe_float),
                "pre_pg30": ("干预前PG30", safe_float),
                "pre_pg60": ("干预前PG60", safe_float),
                "pre_pg120": ("干预前PG120", safe_float),
                "pre_pg180": ("干预前PG180", safe_float),
                # 生化
                "pre_hba1c": ("干预前糖化", safe_float),
                "pre_tg": ("干预前TG", safe_float),
                "pre_tc": ("干预前TC", safe_float),
                "pre_ldl": ("干预前LDL", safe_float),
                "pre_hdl": ("干预前HDL", safe_float),
                "pre_alt": ("干预前ALT", safe_float),
                "pre_ast": ("干预前AST", safe_float),
                # 胰岛素
                "pre_ins_times": ("干预前胰岛素次/天", safe_float),
                "pre_ins_dose": ("干预前胰岛素剂量/次", safe_float),
                "pre_insulin_type": ("干预前胰岛素种类", str),
                # 口服药
                "pre_met_times": ("干预前二甲双胍天/次", safe_float),
                "pre_met_dose": ("干预前二甲双胍剂量/次", safe_float),
                "pre_acb_times": ("干预前阿卡波糖天/次", safe_float),
                "pre_acb_dose": ("干预前阿卡波糖剂量/次", safe_float),
                # 7点血糖
                "pre_bf_before": ("干预前早餐前", safe_float),
                "pre_bf_after": ("干预前早餐后2h", safe_float),
                "pre_lunch_before": ("干预前午餐前", safe_float),
                "pre_lunch_after": ("干预前午餐后2h", safe_float),
                "pre_dinner_before": ("干预前晚餐前", safe_float),
                "pre_dinner_after": ("干预前晚餐后2h", safe_float),
                "pre_bed": ("干预前睡前", safe_float),
                # 日期字段
                "pre_glyc_date": ("干预前5点日期", safe_date),
                "pre_bio_date": ("干预前生化日期", safe_date),
                "pre_7_date": ("干预前7点日期", safe_date),
                "symptom_pre_date": ("干预前体感日期", safe_date),
                "drug_pre_date": ("用药调整干预前日期", safe_date),
                # 文本字段
                "drug_pre_med": ("用药调整干预前用药", str),
                "pre_discomfort": ("干预前身体不适", str),
            }
            # 批量同步标量字段
            for key, (field, converter) in pre_field_mapping.items():
                value = selected_patient_data.get(field)
                if converter is safe_float:
                    value = safe_float(value)
                elif converter is safe_date:
                    value = safe_date(value)
                elif converter is str and value is None:
                    value = ""
                st.session_state[key] = value

            # 同步体感子项（嵌套字典）
            pre_symptom = selected_patient_data.get("干预前体感子项", {})
            symptom_keys = {
                "pre_hal": "口臭", "pre_def": "排便情况", "pre_gi": "胃肠道",
                "pre_num": "四肢麻木", "pre_pru": "皮肤瘙痒", "pre_sleep": "睡眠",
                "pre_vis": "视物", "pre_fat": "乏力", "pre_polyd": "多饮",
                "pre_polyp": "多食", "pre_polyu": "多尿", "pre_lumb": "腰膝酸软",
                "pre_night": "盗汗情况", "pre_mood": "情绪状况"
            }
            for skey, symptom_name in symptom_keys.items():
                st.session_state[skey] = pre_symptom.get(symptom_name)

            # 同步其他药物（将列表转换为多行文本）
            other_meds = selected_patient_data.get("干预前其他药物", [])
            if other_meds and isinstance(other_meds, list):
                lines = []
                for med in other_meds:
                    if isinstance(med, dict):
                        name = med.get("药名", "")
                        times = med.get("每天次数", "")
                        dose = med.get("每次剂量", "")
                        lines.append(f"{name},{times},{dose}".rstrip(','))
                st.session_state.pre_other_meds = "\n".join(lines)
            else:
                st.session_state.pre_other_meds = ""

            # 同步基本信息（这些字段可能未禁用，但保持一致性）
            st.session_state.birth = safe_date(selected_patient_data.get("出生日期"))
            st.session_state.diag = safe_date(selected_patient_data.get("确诊日期"))
            st.session_state.age_manual = safe_float(selected_patient_data.get("年龄"))
            st.session_state.disease_manual = safe_float(selected_patient_data.get("病史年"))
            # 其他文本字段
            st.session_state.location = selected_patient_data.get("所在地", "")
            st.session_state.complications = selected_patient_data.get("并发症", "")
            st.session_state.other_chronic = selected_patient_data.get("其他慢病", "")
            st.session_state.health_coach = selected_patient_data.get("健管师", "")
            st.session_state.doctor = selected_patient_data.get("医生", "")
            st.session_state.clinic_name = selected_patient_data.get("诊所/门店名称", "")
            st.session_state.submitter = selected_patient_data.get("提交人", "")
            st.session_state.supervisor = selected_patient_data.get("指导健管师", "")
            st.session_state.remarks = selected_patient_data.get("备注", "")
            # 记录已同步的患者ID
            st.session_state.last_synced_pre_patient_id = current_patient_id

    # 当选择“新增患者”时，清空干预前相关session_state（可选）
    if selected_patient_name == "+ 新增患者" and st.session_state.edit_mode != "baseline":
        # 清除所有以 pre_ 开头的 session_state 键
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("pre_")]
        for k in keys_to_clear:
            st.session_state[k] = None
        # 也清除基本信息键
        st.session_state.birth = None
        st.session_state.diag = None
        st.session_state.age_manual = 0
        st.session_state.disease_manual = 0.0
        st.session_state.last_synced_pre_patient_id = None
                
                # ==================== 新增：项目类型回填逻辑 ====================
    if selected_patient_data:
        stored_project = selected_patient_data.get("项目/医疗地区", "")
        # 确定期望的项目类型和自定义文本
        if stored_project in ["医疗", "合作项目", "金顶", "赢创"]:
            expected_type = stored_project
            expected_custom = ""
        elif stored_project and stored_project.strip():
            expected_type = "其他"
            expected_custom = stored_project
        else:
            expected_type = "医疗"
            expected_custom = ""

        # 获取当前 session_state 中的值
        current_type = st.session_state.get("project_type_select", "医疗")
        current_custom = st.session_state.get("project_custom_input", "")

        # 仅在需要更新时执行
        if current_type != expected_type or current_custom != expected_custom:
            st.session_state.project_type_select = expected_type
            st.session_state.project_custom_input = expected_custom
            # 同步 current_project_region，确保后续提交使用正确值
            if expected_type == "其他":
                st.session_state.current_project_region = expected_custom
            else:
                st.session_state.current_project_region = expected_type
            st.rerun()
    # ==============================================================

    # 表单默认值来源（优先级：编辑数据 > 已有患者数据 > 空）
    default_patient = None
    if st.session_state.edit_mode == "baseline" and st.session_state.edit_target_patient:
        default_patient = st.session_state.edit_target_patient
    else:
        default_patient = selected_patient_data
    
    # ===== 项目类型选择（位于表单外部，可实时联动） =====
    st.subheader("📌 项目类型与案例来源")
    col_proj1, col_proj2 = st.columns([1, 2])
    with col_proj1:
        project_type = st.selectbox(
            "项目类型",
            ["医疗", "合作项目", "金顶", "赢创", "其他"],
            key="project_type_select"
        )
    with col_proj2:
        project_custom = st.text_input(
            "如果选择“其他”，请填写具体项目名称",
            value=st.session_state.get("project_custom", ""),
            key="project_custom_input"
        )

    # 确定最终的项目区域标识
    if project_type == "其他":
        current_project_region = project_custom.strip()
    else:
        current_project_region = project_type
    st.session_state["current_project_region"] = current_project_region

    # 根据项目类型动态生成营养产品选项列表
    def get_product_options(project_region):
        if project_region == "医疗":
            return ["畅快", "纽畅", "纽畅B", "其他营养治疗"]
        elif project_region == "合作项目":
            return ["清畅", "唐畅", "唐畅B", "其他营养治疗"]
        elif project_region == "金顶":
            return ["清谷夫", "唐平匠", "唐来匠", "其他营养治疗"]
        elif project_region == "赢创":
            return ["益比特畅清", "益比特修畅元", "益比特修夷稳", "其他营养治疗"]                
        else:
            # 自定义项目或未匹配，提供通用选项（沿用原有带斜杠的展示方式）
            return ["畅快/清畅", "纽畅/唐畅", "纽畅B/唐畅B", "其他营养治疗"]

    product_options = get_product_options(current_project_region)
    st.markdown("---")

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

    with st.form(key="patient_form", clear_on_submit=False, enter_to_submit=False):
        # 知情同意书
        with st.expander("📜 知情同意书（请阅读后勾选同意）", expanded=True):
            st.markdown("""
            **《糖尿病医学营养治疗（MNT)真实世界研究》案例收集项目**

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
                    st.caption("每行格式：药名,每天次数,每次剂量（例如：格列本脲,2,500），次数和剂量可省略")
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
                    disabled=pre_disabled,
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
                    st.caption("每行格式：药名,每天次数,每次剂量（例如：格列本脲,2,500），次数和剂量可省略")
                    # 将列表格式转回多行文本（与干预前同步逻辑一致）
                    _post_other_val = ""
                    if default_followup:
                        _raw = default_followup.get("干预后其他药物", "")
                        if isinstance(_raw, list):
                            _lines = []
                            for _med in _raw:
                                if isinstance(_med, dict):
                                    _n = _med.get("药名", "")
                                    _t = _med.get("每天次数", "")
                                    _d = _med.get("每次剂量", "")
                                    _lines.append(f"{_n},{_t},{_d}".rstrip(','))
                            _post_other_val = "\n".join(_lines)
                        elif isinstance(_raw, str):
                            _post_other_val = _raw
                    post_other_meds = st.text_area("其他药物", value=_post_other_val, placeholder="每行：药名，每天次数，每次剂量",
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
                    value=default_followup.get("干预后身体不适", "") if default_followup else "",
                    key="post_discomfort",
                    placeholder="请描述干预后身体不适的变化情况……",
                    height=100
                )
        # 干预方案与使用反馈
        with st.expander("5️⃣ 干预方案与使用反馈", expanded=False):
            # 动态产品选项
            intervention_products = st.multiselect(
                "营养治疗产品（可多选）",
                options=product_options,
                default=default_followup.get("干预方案产品", []) if default_followup else []
            )
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
            drug_reduction_options = ["无变化", "减剂量", "减种类", "停用胰岛素", "停用所有口服药", "停用所有糖尿病药物", "其他"]
            drug_reduction_index = 0
            if default_followup:
                stored_val = default_followup.get("减药/停药情况", "无变化")
                try:
                    drug_reduction_index = drug_reduction_options.index(stored_val)
                except ValueError:
                    drug_reduction_index = 0
            drug_reduction = st.selectbox("减药/停药", drug_reduction_options,
                                         index=drug_reduction_index, key="drug_reduction")



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

            followup_time = datetime.now().strftime("%Y-%m-%d")
            if post_glyc_date is not None:
                followup_time = post_glyc_date.isoformat()
            
            project_region = st.session_state.get("current_project_region", "医疗")

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
                            st.session_state.last_patient = selected_patient_data
                    else:
                        st.info("📝 未填写任何干预后数据，无新增随访记录。可直接在下方查看图表分析或生成AI方案。")
                        st.session_state.last_patient = selected_patient_data
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
                    st.session_state.last_patient = base_data
                st.balloons()
                st.session_state.form_reset = True
                #st.rerun()

    # ===== AI 方案建议（使用当前选中的患者数据） =====
    # 优先使用最后一次提交的患者，若不存在则使用当前下拉框选择的患者
    patient_for_plan = st.session_state.get("last_patient") or selected_patient_data
    if patient_for_plan:
        st.markdown("---")
        st.subheader("🤖 AI 智能方案建议")
        st.write(f"当前患者：**{patient_for_plan.get('患者姓名', '未知')}**")
        if st.button("生成个体化营养治疗方案", key="gen_plan_btn"):
            with st.spinner("正在分析..."):
                try:
                    plan = generate_plan(patient_for_plan)
                    st.session_state.ai_plan = plan
                    # 可选：将方案存入患者数据
                    #patient_for_plan["AI方案"] = plan
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