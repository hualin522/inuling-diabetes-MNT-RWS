import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import os
import json

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

def save_to_google_sheets(patient_dict):
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(st.secrets["google_sheets"]["spreadsheet_id"]).sheet1
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
        st.success("✅ 数据已同步至 Google Sheets")
    except Exception as e:
        st.warning(f"⚠️ Google Sheets 写入失败（数据已保存在本地列表中）: {e}")

def load_patients_from_sheets(submitter_id=None):
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(st.secrets["google_sheets"]["spreadsheet_id"]).sheet1
        all_data = sheet.get_all_records()
        if not all_data:
            return []

        if submitter_id:
            all_data = [row for row in all_data if str(row.get("提交者ID", "")) == str(submitter_id)]

        # 日期、数值字段列表（与之前相同）
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
            "年龄", "病史年"           # 这两项也可能是数字
        ]
        symptom_items = [
            "口臭", "排便情况", "胃肠道", "四肢麻木", "皮肤瘙痒", "睡眠",
            "视物", "乏力", "多饮", "多食", "多尿", "腰膝酸软", "盗汗情况", "情绪状况"
        ]

        filtered_data = []
        for row in all_data:
            if not row.get("提交者ID"):
                continue

            # 数值转换（对所有数值字段使用 safe_float）
            for nf in numeric_fields:
                if nf in row:
                    row[nf] = safe_float(row[nf])
            # 日期转换
            for df in date_fields:
                if df in row:
                    row[df] = safe_date(row[df])

            # 体感子项还原：优先 JSON，其次展平列重建
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

            # 随访记录处理
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

        # 合并同一 (提交者ID, 患者姓名) 的行
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
        st.error(f"从 Google Sheets 加载数据失败：{e}")
        return []

# ============================================
# AI 模块（保留原有完整模板）
# ============================================
@st.cache_resource
def load_knowledge_base(pdf_dir="pdf_data"):
    if not os.path.exists(pdf_dir):
        st.error(f"知识库目录 {pdf_dir} 不存在，请创建并放入 PDF 文件")
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

def build_rag_chain(vectordb, mode="pre"):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    if mode == "pre":
        template = """
你是一位资深的临床营养师和健康管理师。非常了解糖尿病发病的“肠道菌群紊乱-慢性炎症-表观遗传修饰改变”机制：糖尿病发病的主要原因是胰岛素抵抗和胰岛素分泌受损，胰岛素抵抗是糖尿病的始动因素。大量研究证明，糖尿病人存在肠道菌群紊乱问题，有益菌尤其是丁酸菌缺乏，菌群紊乱造成肠屏障受损进一步引发慢性炎症，慢性炎症是引起胰岛素抵抗的重要因素。同时肠道菌群的代谢产物（短链脂肪酸（SCFAs）、次级胆酸、支链氨基酸（BCAA）、氧化三甲胺（TMAO）、咪唑丙酸等）也会影响胰岛素敏感性。此外，肠道菌群紊乱造成的发酵产物失衡会对糖尿病人的表观遗传产生影响，许多与胰岛素分泌或其他代谢相关的基因表达出现问题，造成高糖代谢记忆，对糖尿病并发症发展有重要影响。
基于该机制，你擅长通过科学合理的应用武汉英纽林的系列富含多种益生元的功能营养食品及中药膳营养产品，调节糖尿病患者肠道菌群，促进丁酸发酵，修复肠屏障阻断慢性炎症，促进患者自身的GLP-1分泌，GLP-1可促进胰岛素的分泌和胰岛β细胞功能性再生；并通过调节肠道菌群发酵产物调节表观遗传修饰，实现对高糖代谢记忆的缓解，从而逆转糖尿病患者的胰岛素抵抗及胰岛素分泌障碍问题。
你深入的了解了英纽林的系列产品：
1、畅快：快速平衡肠道菌群。
1) 应用：主要应用于肠道菌群失调、肠道不适应症如便秘、腹泻、肠炎等
2) 产品成分：菊粉、低聚果糖、低聚木糖、低聚半乳糖、棉子低聚糖
3) 规格：10g/袋，30 袋/盒
2、纽畅：代谢核心专利配方，精准增殖丁酸菌，修复肠黏膜，减少炎症，修复代谢。
1) 应用方向：主要应用于糖脂代谢及并发症等慢性病人群。
2) 产品成分：菊粉、燕麦β－葡聚糖、低聚半乳糖、低聚异麦芽糖、维生素Ｃ
3) 规格：10g/袋，30 袋/盒
3、纽畅B：综合调节肠道菌群发酵产物，调节表观遗传。
应用方向：肝脂代谢异常的糖尿病患者、肝火旺盛容易上火的等人群。
产品成分：菊粉、低聚果糖、金银花粉、甜菜碱、维生素C、维生素B2、维生素B6、维生素B12、叶酸
规格：10g/袋，30 袋/盒
4、纽畅伴侣：
1) 应用方向：主要应用于代谢性等慢性病人群、体重管理人群。
2) 产品成分：
高蛋白型：大豆分离蛋白、白芸豆粉、聚葡萄糖、玉米粉、糙米粉、菊粉(≤3g)、乳清蛋白、薏米粉、中链甘油三酯、青稞粉、葛根粉、圆苞车前子粉、决明子粉、维生素C、维生素B2、维生素B6、叶酸、维生素B12。
高纤维型：燕麦粉、聚葡萄糖、苦荞粉、糙米粉、大豆分离蛋白、乳清蛋白、菊粉(≤3g)、中链甘油三酯、蓝莓粉、山药粉、茯苓粉、青稞粉、桑叶粉、乳清蛋白、枸杞粉、苦瓜粉、紫薯粉、魔芋粉、碳酸钙、硫酸镁、维生素C、焦磷酸铁、乳酸锌、维生素B2、维生素B6、叶酸、维生素B12。
3) 规格：35g/袋，15 袋/盒
5、安欣畅
1) 应用方向：主要应用于干预睡眠、焦虑等神经性疾病人群。
2) 产品成分：菊粉、低聚果糖、γ-氨基丁酸、酸枣仁、核桃低聚肽、茯苓、针叶樱桃粉、碳酸钙、焦磷酸铁、乳酸锌
3) 规格：5g/袋，15 袋/盒
6、卫平畅
1) 应用方向：主要适用于胃食管反流、胃炎、胃粘膜损伤、胃溃疡人群等。
2) 产品成分：菊粉、低聚果糖、L-谷氨酰胺、蓝莓粉、阿拉伯木聚糖、小麦低聚肽、岩藻多糖、壳寡糖、海藻酸钠、壳聚糖、高良姜提取物、芦荟提取物、葡甘露聚糖、茶多酚
3) 规格：10g/袋，15 袋/盒
你充分掌握了英纽林系列产品的应用方案：
（一）主要根据五点血糖中的空腹血糖FPG、餐后2 小时血糖PG120min 来出具基本方案：
6.1＜FPG＜7 和/或7.8＜PG120min＜11.1：早餐前畅快1 袋+晚餐前纽畅1 袋
7≤FPG＜8 和/或11.1≤PG120min＜13：早餐前畅快1 袋+午餐前纽畅1 袋+晚餐前纽畅B 1 袋
8≤FPG＜10 和/或13≤PG120min＜16.7：早餐前畅快1 袋+午餐前纽畅1 袋+晚餐前纽畅1 袋+睡前（20 点-21 点）纽畅B 1 袋
FPG≥10 和/或PG120min≥16.7：早畅快1 袋、纽畅1 袋+午纽畅1 袋+晚餐前纽畅1 袋+睡前（20 点-21 点）纽畅B 1 袋
（二）根据患者其他表征在在以上方案基础上增加其他产品搭配：
1、如患者伴随肥胖，根据BMI 增加纽畅伴侣：
24＜BMI≤28：早餐增加纽畅伴侣高蛋白半袋+晚餐增加纽畅伴侣高纤维半袋
BMI＞28：早餐增加纽畅伴侣高蛋白1 袋+晚餐增加纽畅伴侣高纤维1 袋
2、如患者体感中胃肠道评分0-7 分，或有肠炎胃炎、幽门感染、胃食管反流，或有10 年以上糖尿病病史，长期服用药物，午餐前增加卫平畅1 袋；
3、如患者尿酸＞420，午餐后、晚餐后1 小时增加嘌立清各1 片，配合氢氧理疗；
4、如患者体感中睡眠评分0-7 分，睡前半小时增加安欣畅1 袋，配合氢氧理疗；
5、如患者体感中视力模糊、四肢麻木、皮肤瘙痒4-6 分，增加维生素B1、B12。

基于上述知识，你根据当前患者的基础信息（如BMI）、血糖水平（FBG和PG120，以及五点血糖情况、七点血糖情况）、体感指标及生化指标、并发症情况等，制定个体化的干预方案，并分五部分清晰作答：

【本地专业文档】
{context}

【患者干预前数据】
身高：{height} cm
体重：{pre_weight} kg
BMI：{pre_bmi}
腰围：{pre_waist} cm
高压：{pre_sbp} mmHg
低压：{pre_dbp} mmHg
空腹血糖：{pre_fpg} mmol/L
餐后2h血糖：{pre_pg2h} mmol/L
糖化血红蛋白：{pre_hba1c}%
体感详细评分（0分最差，10分最好）：{pre_symptom_detail}
其他慢病：{chronic}
并发症：{complications}
已选用英纽林产品：{selected_products}
联合使用方案细节：{intervention_detail}

【用户问题】
{input}

请按照以下五部分输出：

1. 患者糖尿病病情分析
（根据上述指标，评估患者当前糖尿病严重程度、代谢综合征风险、可能并发症等，如果有相应的体感评分则分析患者的主观不适状况，语气专业温和。）

2. 英纽林营养产品应用方案
（针对该患者的血糖情况和体感状况，结合英纽林产品（重点是畅快、纽畅、纽畅B）的说明书和应用说明，推荐适合该患者的产品、剂量、服用时间、周期。如果患者已选择了某种产品组合，请基于该组合给出具体的服用方案。对产品功效说明时，应严格按照产品的主要功能成分展开。）

3. 日常饮食、中医干预和运动管理建议
（给出具体、可操作的饮食原则与食谱建议；针对患者的体感症状，推荐适合的符合国家要求的药食同源类中药；并给出适合患者的运动类型、频率、强度建议。并与营养方案配合。）

4. 干预效果预期分析
（科学预估在规范使用产品并配合生活调整后，3~6个月内各项指标可能的改善幅度，如血糖、体重、糖化等。）

5. 总结
（用一段鼓励的话语总结整体方案，强调坚持的重要性，表达积极预期。）
"""
    else:
        template = """
你是一位富有亲和力的临床营养师和健康管理师。
你非常了解糖尿病发病的“肠道菌群紊乱-慢性炎症-表观遗传修饰改变”机制：糖尿病发病的主要原因是胰岛素抵抗和胰岛素分泌受损，胰岛素抵抗是糖尿病的始动因素。大量研究证明，糖尿病人存在肠道菌群紊乱问题，有益菌尤其是丁酸菌缺乏，菌群紊乱造成肠屏障受损进一步引发慢性炎症，慢性炎症是引起胰岛素抵抗的重要因素。同时肠道菌群的代谢产物（短链脂肪酸（SCFAs）、次级胆酸、支链氨基酸（BCAA）、氧化三甲胺（TMAO）、咪唑丙酸等）也会影响胰岛素敏感性。此外，肠道菌群紊乱造成的发酵产物失衡会对糖尿病人的表观遗传产生影响，许多与胰岛素分泌或其他代谢相关的基因表达出现问题，造成高糖代谢记忆，对糖尿病并发症发展有重要影响。
基于该机制，你擅长通过科学合理的应用武汉英纽林的系列富含多种益生元的功能营养食品及中药膳营养产品，调节糖尿病患者肠道菌群，促进丁酸发酵，修复肠屏障阻断慢性炎症，促进患者自身的GLP-1分泌，GLP-1可促进胰岛素的分泌和胰岛β细胞功能性再生；并通过调节肠道菌群发酵产物调节表观遗传修饰，实现对高糖代谢记忆的缓解，从而逆转糖尿病患者的胰岛素抵抗及胰岛素分泌障碍问题。
你深入的了解了英纽林的系列产品：
1、畅快：快速平衡肠道菌群。
1) 应用：主要应用于肠道菌群失调、肠道不适应症如便秘、腹泻、肠炎等
2) 产品成分：菊粉、低聚果糖、低聚木糖、低聚半乳糖、棉子低聚糖
3) 规格：10g/袋，30 袋/盒
2、纽畅：代谢核心专利配方，精准增殖丁酸菌，修复肠黏膜，减少炎症，修复代谢。
1) 应用方向：主要应用于糖脂代谢及并发症等慢性病人群。
2) 产品成分：菊粉、燕麦β－葡聚糖、低聚半乳糖、低聚异麦芽糖、维生素Ｃ
3) 规格：10g/袋，30 袋/盒
3、纽畅B：综合调节肠道菌群发酵产物，调节表观遗传。
应用方向：肝脂代谢异常的糖尿病患者、肝火旺盛容易上火的等人群。
产品成分：菊粉、低聚果糖、金银花粉、甜菜碱、维生素C、维生素B2、维生素B6、维生素B12、叶酸
规格：10g/袋，30 袋/盒
4、纽畅伴侣：
1) 应用方向：主要应用于代谢性等慢性病人群、体重管理人群。
2) 产品成分：
高蛋白型：大豆分离蛋白、白芸豆粉、聚葡萄糖、玉米粉、糙米粉、菊粉(≤3g)、乳清蛋白、薏米粉、中链甘油三酯、青稞粉、葛根粉、圆苞车前子粉、决明子粉、维生素C、维生素B2、维生素B6、叶酸、维生素B12。
高纤维型：燕麦粉、聚葡萄糖、苦荞粉、糙米粉、大豆分离蛋白、乳清蛋白、菊粉(≤3g)、中链甘油三酯、蓝莓粉、山药粉、茯苓粉、青稞粉、桑叶粉、乳清蛋白、枸杞粉、苦瓜粉、紫薯粉、魔芋粉、碳酸钙、硫酸镁、维生素C、焦磷酸铁、乳酸锌、维生素B2、维生素B6、叶酸、维生素B12。
3) 规格：35g/袋，15 袋/盒
5、安欣畅
1) 应用方向：主要应用于干预睡眠、焦虑等神经性疾病人群。
2) 产品成分：菊粉、低聚果糖、γ-氨基丁酸、酸枣仁、核桃低聚肽、茯苓、针叶樱桃粉、碳酸钙、焦磷酸铁、乳酸锌
3) 规格：5g/袋，15 袋/盒
6、卫平畅
1) 应用方向：主要适用于胃食管反流、胃炎、胃粘膜损伤、胃溃疡人群等。
2) 产品成分：菊粉、低聚果糖、L-谷氨酰胺、蓝莓粉、阿拉伯木聚糖、小麦低聚肽、岩藻多糖、壳寡糖、海藻酸钠、壳聚糖、高良姜提取物、芦荟提取物、葡甘露聚糖、茶多酚
3) 规格：10g/袋，15 袋/盒
你充分掌握了英纽林系列产品的应用方案：
（一）主要根据五点血糖中的空腹血糖FPG、餐后2 小时血糖PG120min 来出具基本方案：
6.1＜FPG＜7 和/或7.8＜PG120min＜11.1：早餐前畅快1 袋+晚餐前纽畅1 袋
7≤FPG＜8 和/或11.1≤PG120min＜13：早餐前畅快1 袋+午餐前纽畅1 袋+晚餐前纽畅B 1 袋
8≤FPG＜10 和/或13≤PG120min＜16.7：早餐前畅快1 袋+午餐前纽畅1 袋+晚餐前纽畅1 袋+睡前（20 点-21 点）纽畅B 1 袋
FPG≥10 和/或PG120min≥16.7：早畅快1 袋、纽畅1 袋+午纽畅1 袋+晚餐前纽畅1 袋+睡前（20 点-21 点）纽畅B 1 袋
（二）根据患者其他表征在在以上方案基础上增加其他产品搭配：
1、如患者伴随肥胖，根据BMI 增加纽畅伴侣：
24＜BMI≤28：早餐增加纽畅伴侣高蛋白半袋+晚餐增加纽畅伴侣高纤维半袋
BMI＞28：早餐增加纽畅伴侣高蛋白1 袋+晚餐增加纽畅伴侣高纤维1 袋
2、如患者体感中胃肠道评分0-7 分，或有肠炎胃炎、幽门感染、胃食管反流，或有10 年以上糖尿病病史，长期服用药物，午餐前增加卫平畅1 袋；
3、如患者尿酸＞420，午餐后、晚餐后1 小时增加嘌立清各1 片，配合氢氧理疗；
4、如患者体感中睡眠评分0-7 分，睡前半小时增加安欣畅1 袋，配合氢氧理疗；
5、如患者体感中视力模糊、四肢麻木、皮肤瘙痒0-7 分，建议额外增加维生素B1、B12。

基于上述知识，你擅长用积极、鼓励的方式解读英纽林系列营养产品对糖尿病营养治疗、逆转糖尿病患者的胰岛素抵抗及胰岛素分泌障碍问题的效果。请根据以下资料，为这位患者进行干预前后对比分析，并给出下阶段建议。

【本地专业文档】
{context}

【患者干预前数据】
身高：{height} cm
体重：{pre_weight} kg
BMI：{pre_bmi}
腰围：{pre_waist} cm
高压：{pre_sbp} mmHg
低压：{pre_dbp} mmHg
空腹血糖：{pre_fpg} mmol/L
餐后2h血糖：{pre_pg2h} mmol/L
糖化血红蛋白：{pre_hba1c}%
体感详细评分（0分最差，10分最好）：{pre_symptom_detail}
其他慢病：{chronic}
并发症：{complications}

【患者干预后数据】
体重：{post_weight} kg
BMI：{post_bmi}
腰围：{post_waist} cm
高压：{post_sbp} mmHg
低压：{post_dbp} mmHg
空腹血糖：{post_fpg} mmol/L
餐后2h血糖：{post_pg2h} mmol/L
糖化血红蛋白：{post_hba1c}%
体感详细评分（0分最差，10分最好）：{post_symptom_detail}
已选用英纽林产品：{selected_products}
联合使用方案细节：{intervention_detail}

【使用反馈】
不良反应：{feedback_symptoms}
详细描述：{feedback_notes}

【用户问题】
{input}

请以热情、鼓励的口吻输出以下两部分：

1. 糖尿病改善情况分析
（对比干预前后的关键指标及体感变化，用通俗易懂的语言解释哪些方面有明显改善，哪些还需要继续努力。哪怕指标仅微小改善，也要用“已经迈出重要一步”“身体正在向好的方向调整”等语言给予充分肯定。如果患者已选择了某种产品组合，请基于该组合推荐合适的具体应用方案，如果出现某些暂时的不良反应，请科学解释并安抚，强调这是向好的过渡现象。）

2. 下一阶段营养干预建议
（结合本地文档和当前改善程度，以及使用反馈中提到的状况，推荐下一阶段的产品使用调整（例如是否需要更换种类、调整剂量、配合其他辅助措施等）。同时给出饮食、药食同源中药和运动方面的优化建议，帮助患者朝着更好的方向前进。）
"""
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

    # 判断是否有干预后数据：优先检查随访记录，其次检查顶层字段
    has_post = False
    followups = patient_combined_data.get("随访记录", [])
    if isinstance(followups, list) and followups:
        # 如果随访记录里至少有一条包含关键干预后指标，则认为有干预后数据
        for fu in followups:
            if any([
                fu.get("干预后FPG"), fu.get("干预后PG120"),
                fu.get("干预后糖化"), fu.get("干预后体重")
            ]):
                has_post = True
                break
    else:
        # 兼容旧数据：顶层存在任何干预后字段也视为有干预后
        has_post = any([
            patient_combined_data.get("干预后FPG"),
            patient_combined_data.get("干预后PG120"),
            patient_combined_data.get("干预后糖化"),
            patient_combined_data.get("干预后体重")
        ])
    mode = "post" if has_post else "pre"
    input_text = "请为这位糖尿病患者依据英纽林产品说明制定个体化的营养治疗方案，并预测可能的效果" if mode == "pre" else "请为这位糖尿病患者进行干预前后对比分析，并给出下一阶段的营养建议。"
    rag_chain = build_rag_chain(vectordb, mode)

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
        # 如果随访记录存在，优先取最新一条随访的干预后数据（或取第一条，可由您指定）
        if followups and isinstance(followups, list):
            latest_followup = followups[-1]  # 取最后一次随访
            post_symptom = latest_followup.get("干预后体感子项", {})
        else:
            # 兼容旧模式：直接从顶层字段读取
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
    
    # 其余部分（fb_symptoms, selected_products 等）保持不变，调用 invoke
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
    }
    result = rag_chain.invoke(invoke_input)
    return result["answer"]

# ============================================
# 主界面
# ============================================
def patient_info_entry():
    st.header("📋 英纽林糖尿病营养治疗真实世界研究案例收集")

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

    if "patients" not in st.session_state:
        st.session_state.patients = []
    if "loaded_from_cloud" not in st.session_state:
        st.session_state.loaded_from_cloud = False

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

    # 患者选择（严格按提交者ID过滤）
    if is_admin:
        all_patient_names = list({p["患者姓名"] for p in st.session_state.patients})
    else:
        all_patient_names = list({p["患者姓名"] for p in st.session_state.patients if p.get("提交者ID") == submitter_id})
    patient_names = ["+ 新增患者"] + all_patient_names
    selected_patient_name = st.selectbox("选择已有患者（可自动填充干预前数据）", patient_names, key="selected_patient")

    selected_patient_data = None
    if selected_patient_name != "+ 新增患者":
        for p in st.session_state.patients:
            if p["患者姓名"] == selected_patient_name:
                if is_admin or p.get("提交者ID") == submitter_id:
                    selected_patient_data = p
                    break

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
        # 1. 用户基本信息
        with st.expander("1️⃣ 用户基本信息", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                name = st.text_input("患者姓名 *", value=selected_patient_data["患者姓名"] if selected_patient_data else "")
                gender = st.selectbox("性别", ["男", "女"], index=["男", "女"].index(selected_patient_data["性别"]) if selected_patient_data and selected_patient_data.get("性别") in ["男", "女"] else 0)
                phone = st.text_input("联系电话", value=selected_patient_data.get("联系电话", "") if selected_patient_data else "")
            with col2:
                default_birth = safe_date(selected_patient_data.get("出生日期")) if selected_patient_data else None
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
                default_diag = safe_date(selected_patient_data.get("确诊日期")) if selected_patient_data else None
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
                location = st.text_input("所在地/省/市/区", value=selected_patient_data.get("所在地", "") if selected_patient_data else "")
                complications = st.text_input("并发症", value=selected_patient_data.get("并发症", "") if selected_patient_data else "")
                other_chronic = st.text_input("其他慢病", value=selected_patient_data.get("其他慢病", "") if selected_patient_data else "")

        # ========== 干预前数据（大板块） ==========
        with st.expander("2️⃣ 干预前数据（基本指标、五点血糖、体感、药物、生化、7点血糖）", expanded=False):
                # 干预前基本指标
            with st.expander("基本指标", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    pre_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0,
                                                value=safe_float(selected_patient_data.get("干预前身高")) if selected_patient_data else None,
                                                step=0.1, key="pre_h", disabled=selected_patient_data is not None)
                    pre_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0,
                                                value=safe_float(selected_patient_data.get("干预前体重")) if selected_patient_data else None,
                                                step=0.1, key="pre_w", disabled=selected_patient_data is not None)
                with col2:
                    pre_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0,
                                                value=safe_float(selected_patient_data.get("干预前腰围")) if selected_patient_data else None,
                                                step=0.1, disabled=selected_patient_data is not None)
                    pre_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0,
                                            value=safe_float(selected_patient_data.get("干预前臀围")) if selected_patient_data else None,
                                            step=0.1, disabled=selected_patient_data is not None)
                col1, col2 = st.columns(2)
                with col1:
                    pre_sbp = st.number_input("高压 (mmHg)", min_value=50.0, max_value=250.0,
                                            value=safe_float(selected_patient_data.get("干预前高压")) if selected_patient_data else None,
                                            step=1.0, disabled=selected_patient_data is not None)
                with col2:
                    pre_dbp = st.number_input("低压 (mmHg)", min_value=30.0, max_value=150.0,
                                            value=safe_float(selected_patient_data.get("干预前低压")) if selected_patient_data else None,
                                            step=1.0, disabled=selected_patient_data is not None)
                pre_bmi = calculate_bmi(pre_height, pre_weight)

                # 干预前5点血糖
            with st.expander("5点血糖", expanded=False):
                pre_glyc_date = st.date_input("检测日期", value=safe_date(selected_patient_data.get("干预前5点日期")) if selected_patient_data else None, min_value=date(1900,1,1), key="pre_glyc_date", disabled=selected_patient_data is not None)
                pre_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1,
                                        value=safe_float(selected_patient_data.get("干预前FPG")) if selected_patient_data else None,
                                        key="pre_fpg", disabled=selected_patient_data is not None)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pre_pg30 = st.number_input("PG 30min (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前PG30")) if selected_patient_data else None,
                                            key="pre_pg30", disabled=selected_patient_data is not None)
                with col2:
                    pre_pg60 = st.number_input("PG 60min (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前PG60")) if selected_patient_data else None,
                                            key="pre_pg60", disabled=selected_patient_data is not None)
                with col3:
                    pre_pg120 = st.number_input("PG 120min (mmol/L)", min_value=0.0, step=0.1,
                                                value=safe_float(selected_patient_data.get("干预前PG120")) if selected_patient_data else None,
                                                key="pre_pg120", disabled=selected_patient_data is not None)
                with col4:
                    pre_pg180 = st.number_input("PG 180min (mmol/L)", min_value=0.0, step=0.1,
                                                value=safe_float(selected_patient_data.get("干预前PG180")) if selected_patient_data else None,
                                                key="pre_pg180", disabled=selected_patient_data is not None)

                # 干预前体感指标
            with st.expander("体感指标", expanded=False):
                st.caption("评分标准：0分为最差，10分为最好（即无该症状）")
                pre_symptom_date = st.date_input("录入日期", value=safe_date(selected_patient_data.get("干预前体感日期")) if selected_patient_data else None, min_value=date(1900,1,1), key="symptom_pre_date", disabled=selected_patient_data is not None)
                st.caption("如果与五点血糖检测日期相同，可不填")
                pre_scores = selected_patient_data.get("干预前体感子项", {}) if selected_patient_data else {}
                col1, col2, col3 = st.columns(3)
                with col1:
                    pre_halitosis = st.number_input("口臭", 1, 10, value=pre_scores.get("口臭"), key="pre_hal", disabled=selected_patient_data is not None)
                    pre_defecation = st.number_input("排便情况", 1, 10, value=pre_scores.get("排便情况"), key="pre_def", disabled=selected_patient_data is not None)
                    pre_gi = st.number_input("胃肠道", 1, 10, value=pre_scores.get("胃肠道"), key="pre_gi", disabled=selected_patient_data is not None)
                    pre_numbness = st.number_input("四肢麻木", 1, 10, value=pre_scores.get("四肢麻木"), key="pre_num", disabled=selected_patient_data is not None)
                with col2:
                    pre_pruritus = st.number_input("皮肤瘙痒", 1, 10, value=pre_scores.get("皮肤瘙痒"), key="pre_pru", disabled=selected_patient_data is not None)
                    pre_sleep = st.number_input("睡眠", 1, 10, value=pre_scores.get("睡眠"), key="pre_sleep", disabled=selected_patient_data is not None)
                    pre_vision = st.number_input("视物", 1, 10, value=pre_scores.get("视物"), key="pre_vis", disabled=selected_patient_data is not None)
                    pre_fatigue = st.number_input("乏力", 1, 10, value=pre_scores.get("乏力"), key="pre_fat", disabled=selected_patient_data is not None)
                with col3:
                    pre_polydipsia = st.number_input("多饮", 1, 10, value=pre_scores.get("多饮"), key="pre_polyd", disabled=selected_patient_data is not None)
                    pre_polyphagia = st.number_input("多食", 1, 10, value=pre_scores.get("多食"), key="pre_polyp", disabled=selected_patient_data is not None)
                    pre_polyuria = st.number_input("多尿", 1, 10, value=pre_scores.get("多尿"), key="pre_polyu", disabled=selected_patient_data is not None)
                    pre_lumbago = st.number_input("腰膝酸软", 1, 10, value=pre_scores.get("腰膝酸软"), key="pre_lumb", disabled=selected_patient_data is not None)
                col1, col2 = st.columns(2)
                with col1:
                    pre_night_sweat = st.number_input("盗汗情况", 1, 10, value=pre_scores.get("盗汗情况"), key="pre_night", disabled=selected_patient_data is not None)
                with col2:
                    pre_mood = st.number_input("情绪状况", 1, 10, value=pre_scores.get("情绪状况"), key="pre_mood", disabled=selected_patient_data is not None)
                pre_symptom_scores = {
                    "口臭": pre_halitosis, "排便情况": pre_defecation, "胃肠道": pre_gi,
                    "四肢麻木": pre_numbness, "皮肤瘙痒": pre_pruritus, "睡眠": pre_sleep,
                    "视物": pre_vision, "乏力": pre_fatigue, "多饮": pre_polydipsia,
                    "多食": pre_polyphagia, "多尿": pre_polyuria, "腰膝酸软": pre_lumbago,
                    "盗汗情况": pre_night_sweat, "情绪状况": pre_mood
                }
                pre_total = calculate_symptom_total(pre_symptom_scores)

                # 干预前糖尿病药物
            with st.expander("糖尿病药物", expanded=False):
                st.subheader("胰岛素")
                col1, col2 = st.columns(2)
                with col1:
                    pre_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0.0, step=1.0,
                                                        value=safe_float(selected_patient_data.get("干预前胰岛素次/天")) if selected_patient_data else None,
                                                        key="pre_ins_times", disabled=selected_patient_data is not None)
                    pre_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0,
                                                        value=safe_float(selected_patient_data.get("干预前胰岛素剂量/次")) if selected_patient_data else None,
                                                        key="pre_ins_dose", disabled=selected_patient_data is not None)
                with col2:
                    pass
                st.subheader("口服药")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pre_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(selected_patient_data.get("干预前二甲双胍天/次")) if selected_patient_data else None,
                                                        key="pre_met_times", disabled=selected_patient_data is not None)
                    pre_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0.0, step=250.0,
                                                        value=safe_float(selected_patient_data.get("干预前二甲双胍剂量/次")) if selected_patient_data else None,
                                                        key="pre_met_dose", disabled=selected_patient_data is not None)
                with col2:
                    pre_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0.0, step=1.0,
                                                        value=safe_float(selected_patient_data.get("干预前阿卡波糖天/次")) if selected_patient_data else None,
                                                        key="pre_acb_times", disabled=selected_patient_data is not None)
                    pre_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0.0, step=50.0,
                                                        value=safe_float(selected_patient_data.get("干预前阿卡波糖剂量/次")) if selected_patient_data else None,
                                                        key="pre_acb_dose", disabled=selected_patient_data is not None)
                with col3:
                    pre_other_meds = st.text_area("其他药物", value="" if not selected_patient_data else "", placeholder="每行：药名，每天次数，每次剂量",
                                                key="pre_other_meds", disabled=selected_patient_data is not None)

                # 干预前生化指标
            with st.expander("生化指标", expanded=False):
                pre_bio_date = st.date_input("检测日期", value=safe_date(selected_patient_data.get("干预前生化日期")) if selected_patient_data else None, min_value=date(1900,1,1), key="pre_bio_date", disabled=selected_patient_data is not None)
                st.caption("如果与五点血糖检测日期相同，可不填")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pre_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1,
                                                value=safe_float(selected_patient_data.get("干预前糖化")) if selected_patient_data else None,
                                                key="pre_hba1c", disabled=selected_patient_data is not None)
                    pre_tc = st.number_input("TC (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前TC")) if selected_patient_data else None,
                                            key="pre_tc", disabled=selected_patient_data is not None)
                with col2:
                    pre_tg = st.number_input("TG (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前TG")) if selected_patient_data else None,
                                            key="pre_tg", disabled=selected_patient_data is not None)
                    pre_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前LDL")) if selected_patient_data else None,
                                            key="pre_ldl", disabled=selected_patient_data is not None)
                with col3:
                    pre_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1,
                                            value=safe_float(selected_patient_data.get("干预前HDL")) if selected_patient_data else None,
                                            key="pre_hdl", disabled=selected_patient_data is not None)
                    pre_alt = st.number_input("ALT (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(selected_patient_data.get("干预前ALT")) if selected_patient_data else None,
                                            key="pre_alt", disabled=selected_patient_data is not None)
                with col4:
                    pre_ast = st.number_input("AST (U/L)", min_value=0.0, step=1.0,
                                            value=safe_float(selected_patient_data.get("干预前AST")) if selected_patient_data else None,
                                            key="pre_ast", disabled=selected_patient_data is not None)

                # 干预前7点血糖
            with st.expander("日常7点血糖", expanded=False):
                pre_7_date = st.date_input("检测日期", value=safe_date(selected_patient_data.get("干预前7点日期")) if selected_patient_data else None, min_value=date(1900,1,1), key="pre_7_date", disabled=selected_patient_data is not None)
                cols = st.columns(7)
                with cols[0]:
                    pre_bf_before = st.number_input("早餐前", step=0.1, value=safe_float(selected_patient_data.get("干预前早餐前")) if selected_patient_data else None, key="pre_bf_before", disabled=selected_patient_data is not None)
                with cols[1]:
                    pre_bf_after = st.number_input("早餐后2h", step=0.1, value=safe_float(selected_patient_data.get("干预前早餐后2h")) if selected_patient_data else None, key="pre_bf_after", disabled=selected_patient_data is not None)
                with cols[2]:
                    pre_lunch_before = st.number_input("午餐前", step=0.1, value=safe_float(selected_patient_data.get("干预前午餐前")) if selected_patient_data else None, key="pre_lunch_before", disabled=selected_patient_data is not None)
                with cols[3]:
                    pre_lunch_after = st.number_input("午餐后2h", step=0.1, value=safe_float(selected_patient_data.get("干预前午餐后2h")) if selected_patient_data else None, key="pre_lunch_after", disabled=selected_patient_data is not None)
                with cols[4]:
                    pre_dinner_before = st.number_input("晚餐前", step=0.1, value=safe_float(selected_patient_data.get("干预前晚餐前")) if selected_patient_data else None, key="pre_dinner_before", disabled=selected_patient_data is not None)
                with cols[5]:
                    pre_dinner_after = st.number_input("晚餐后2h", step=0.1, value=safe_float(selected_patient_data.get("干预前晚餐后2h")) if selected_patient_data else None, key="pre_dinner_after", disabled=selected_patient_data is not None)
                with cols[6]:
                    pre_bed = st.number_input("睡前", step=0.1, value=safe_float(selected_patient_data.get("干预前睡前")) if selected_patient_data else None, key="pre_bed", disabled=selected_patient_data is not None)

        # ========== 干预后数据（大板块） ==========
        with st.expander("3️⃣ 干预后数据（基本指标、五点血糖、体感、药物、生化、7点血糖）", expanded=False):
                # 干预后基本指标
            with st.expander("基本指标", expanded=False): 
                col1, col2 = st.columns(2)
                with col1:
                    post_height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=pre_height, step=0.1, key="post_h", disabled=True)
                    post_weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=None, step=0.1, key="post_w")
                with col2:
                    post_waist = st.number_input("腰围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1, key="post_wc")
                    post_hip = st.number_input("臀围 (cm)", min_value=50.0, max_value=200.0, value=None, step=0.1, key="post_hc")
                col1, col2 = st.columns(2)
                with col1:
                    post_sbp = st.number_input("高压 (mmHg)", min_value=50.0, max_value=250.0, value=None, step=1.0, key="post_sbp")
                with col2:
                    post_dbp = st.number_input("低压 (mmHg)", min_value=30.0, max_value=150.0, value=None, step=1.0, key="post_dbp")
                post_bmi = calculate_bmi(post_height, post_weight)

                # 干预后5点血糖
            with st.expander("5点血糖", expanded=False):
                post_glyc_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_glyc_date")
                post_fpg = st.number_input("FPG 空腹血糖 (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_fpg")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    post_pg30 = st.number_input("PG 30min", min_value=0.0, step=0.1, value=None, key="post_30")
                with col2:
                    post_pg60 = st.number_input("PG 60min", min_value=0.0, step=0.1, value=None, key="post_60")
                with col3:
                    post_pg120 = st.number_input("PG 120min", min_value=0.0, step=0.1, value=None, key="post_120")
                with col4:
                    post_pg180 = st.number_input("PG 180min", min_value=0.0, step=0.1, value=None, key="post_180")

                # 干预后体感指标
            with st.expander("体感指标", expanded=False):
                st.caption("评分标准：0分为最差，10分为最好（即无该症状）")
                post_symptom_date = st.date_input("录入日期", value=None, min_value=date(1900,1,1), key="symptom_post_date")
                st.caption("如果与五点血糖检测日期相同，可不填")
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

                # 干预后糖尿病药物
            with st.expander("糖尿病药物", expanded=False):
                st.subheader("胰岛素")
                col1, col2 = st.columns(2)
                with col1:
                    post_insulin_times = st.number_input("胰岛素 (次/天)", min_value=0.0, step=1.0, value=None, key="post_ins_times")
                    post_insulin_dose = st.number_input("剂量/次 (IU)", min_value=0.0, step=1.0, value=None, key="post_ins_dose")
                with col2:
                    pass
                st.subheader("口服药")
                col1, col2, col3 = st.columns(3)
                with col1:
                    post_metformin_times = st.number_input("二甲双胍 (天/次)", min_value=0.0, step=1.0, value=None, key="post_met_times")
                    post_metformin_dose = st.number_input("二甲双胍 剂量/次 (mg)", min_value=0.0, step=250.0, value=None, key="post_met_dose")
                with col2:
                    post_acarbose_times = st.number_input("阿卡波糖 (天/次)", min_value=0.0, step=1.0, value=None, key="post_acb_times")
                    post_acarbose_dose = st.number_input("阿卡波糖 剂量/次 (mg)", min_value=0.0, step=50.0, value=None, key="post_acb_dose")
                with col3:
                    post_other_meds = st.text_area("其他药物", placeholder="每行：药名，每天次数，每次剂量", key="post_other_meds")

                # 干预后生化指标
            with st.expander("生化指标", expanded=False):
                post_bio_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_bio_date")
                st.caption("如果与五点血糖检测日期相同，可不填")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    post_hba1c = st.number_input("糖化/%", min_value=0.0, max_value=20.0, step=0.1, value=None, key="post_hba1c")
                    post_tc = st.number_input("TC (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_tc")
                with col2:
                    post_tg = st.number_input("TG (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_tg")
                    post_ldl = st.number_input("LDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_ldl")
                with col3:
                    post_hdl = st.number_input("HDL-C (mmol/L)", min_value=0.0, step=0.1, value=None, key="post_hdl")
                    post_alt = st.number_input("ALT (U/L)", min_value=0.0, step=1.0, value=None, key="post_alt")
                with col4:
                    post_ast = st.number_input("AST (U/L)", min_value=0.0, step=1.0, value=None, key="post_ast")

                # 干预后7点血糖
            with st.expander("日常7点血糖", expanded=False):
                post_7_date = st.date_input("检测日期", value=None, min_value=date(1900,1,1), key="post_7_date")
                cols = st.columns(7)
                with cols[0]:
                    post_bf_before = st.number_input("早餐前", step=0.1, value=None, key="post_bf_before")
                with cols[1]:
                    post_bf_after = st.number_input("早餐后2h", step=0.1, value=None, key="post_bf_after")
                with cols[2]:
                    post_lunch_before = st.number_input("午餐前", step=0.1, value=None, key="post_lunch_before")
                with cols[3]:
                    post_lunch_after = st.number_input("午餐后2h", step=0.1, value=None, key="post_lunch_after")
                with cols[4]:
                    post_dinner_before = st.number_input("晚餐前", step=0.1, value=None, key="post_dinner_before")
                with cols[5]:
                    post_dinner_after = st.number_input("晚餐后2h", step=0.1, value=None, key="post_dinner_after")
                with cols[6]:
                    post_bed = st.number_input("睡前", step=0.1, value=None, key="post_bed")

        # ========== 干预方案 ==========
        with st.expander("4️⃣ 干预方案与使用反馈", expanded=False):
            intervention_products = st.multiselect("营养治疗产品（可多选）", ["畅快", "纽畅", "纽畅B", "其他营养治疗"], key="intervention_products")
            other_product_name = ""
            if "其他营养治疗" in intervention_products:
                other_product_name = st.text_input("请输入‘其他营养治疗’的具体名称", key="other_product_name")
            intervention_detail = st.text_area("干预方案细节（用量/用法/周期/搭配方式等）", placeholder="例如：畅快 每日1次 每次1包……", key="intervention_detail")
            st.markdown("---")
            st.subheader("使用反馈")
            feedback_symptoms = st.multiselect("常见不良反应", ["腹泻", "便秘", "腹胀", "恶心", "腹痛", "过敏/皮疹", "其他"], key="feedback_symptoms")
            feedback_notes = st.text_area("反馈详细描述", placeholder="……", key="feedback_notes")

        # ========== 用药调整情况 ==========
        with st.expander("5️⃣ 用药调整情况", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                drug_pre_date = st.date_input("干预前日期", value=safe_date(selected_patient_data.get("用药调整干预前日期")) if selected_patient_data else None, min_value=date(1900,1,1), key="drug_pre_date", disabled=selected_patient_data is not None)
                drug_pre_med = st.text_area("干预前用药 (可简述)", value=selected_patient_data.get("用药调整干预前用药", "") if selected_patient_data else "", key="drug_pre_med", disabled=selected_patient_data is not None)
            with col2:
                drug_post_date = st.date_input("干预后日期", value=None, min_value=date(1900,1,1), key="drug_post_date")
                drug_post_med = st.text_area("干预后用药 (可简述)", key="drug_post_med")
            drug_reduction = st.selectbox("减药/停药", ["无变化", "减剂量", "减种类", "停用所有口服", "其他"], key="drug_reduction")

        # ========== 案例来源与备注 ==========
        with st.expander("6️⃣ 案例来源 & 备注", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                project_region = st.text_input("项目/医疗地区", value=selected_patient_data.get("项目/医疗地区", "") if selected_patient_data else "")
                health_coach = st.text_input("健管师", value=selected_patient_data.get("健管师", "") if selected_patient_data else "")
                doctor = st.text_input("医生", value=selected_patient_data.get("医生", "") if selected_patient_data else "")
                clinic_name = st.text_input("诊所/门店名称", value=selected_patient_data.get("诊所/门店名称", "") if selected_patient_data else "")
            with col2:
                submitter = st.text_input("提交人", value=selected_patient_data.get("提交人", "") if selected_patient_data else "")
                supervisor = st.text_input("指导健管师", value=selected_patient_data.get("指导健管师", "") if selected_patient_data else "")
            remarks = st.text_area("备注信息", value=selected_patient_data.get("备注", "") if selected_patient_data else "")

        # 提交
        submitted = st.form_submit_button("✅ 提交并保存患者信息")
        if submitted:
            if not name:
                st.error("患者姓名不能为空")
                st.stop()

            # 异常值预警
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

            # 随访时间优先使用干预后5点血糖日期
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
            }

            # 判断是否为空随访
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

            if selected_patient_name != "+ 新增患者" and selected_patient_data:
                selected_patient_data["提交者ID"] = submitter_id if not is_admin else "admin"
                if "随访记录" not in selected_patient_data:
                    selected_patient_data["随访记录"] = []
                if not empty_followup:
                    selected_patient_data["随访记录"].append(new_followup)
                    st.success(f"✅ 已为 {selected_patient_name} 添加新的随访记录")
                else:
                    st.info("📝 未填写任何干预后数据，仅更新基线信息（如有修改）。")
                st.session_state.last_patient = selected_patient_data
            else:
                base_followups = [] if empty_followup else [new_followup]
                base_data = {
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
                st.session_state.last_patient = base_data
                if not empty_followup:
                    st.success(f"✅ 患者 {name} 已新增并录入首次随访数据")
                else:
                    st.success(f"✅ 患者 {name} 的基线信息已保存，干预后数据待下次录入")

            # 保存到 Google Sheets
            save_data = st.session_state.last_patient.copy()
            if "随访记录" in save_data and isinstance(save_data["随访记录"], list) and save_data["随访记录"]:
                latest = save_data["随访记录"][-1]
                save_data["最新随访FPG"] = latest.get("干预后FPG", "")
                save_data["最新随访PG30"] = latest.get("干预后PG30", "")
                save_data["最新随访PG60"] = latest.get("干预后PG60", "")
                save_data["最新随访PG120"] = latest.get("干预后PG120", "")
                save_data["最新随访PG180"] = latest.get("干预后PG180", "")
            else:
                for k in ["最新随访FPG", "最新随访PG30", "最新随访PG60", "最新随访PG120", "最新随访PG180"]:
                    save_data[k] = ""
            # 将干预前体感子项转为 JSON 字符串，避免被 flatten_dict 展开
            if "干预前体感子项" in save_data and isinstance(save_data["干预前体感子项"], dict):
                save_data["干预前体感子项"] = json.dumps(save_data["干预前体感子项"], ensure_ascii=False)
            if "随访记录" in save_data and isinstance(save_data["随访记录"], list):
                save_data["随访记录"] = json.dumps(save_data["随访记录"], ensure_ascii=False, default=str)
            if "gcp_service_account" in st.secrets and "google_sheets" in st.secrets:
                save_to_google_sheets(save_data)
            else:
                st.info("💡 提示：配置 Google Sheets 后数据将自动云端汇总")
            st.balloons()

    # ===== AI 方案建议 =====
    if st.session_state.get("last_patient"):
        st.markdown("---")
        st.subheader("🤖 AI 智能方案建议")
        patient_for_plan = st.session_state.last_patient
        st.write(f"当前患者：**{patient_for_plan.get('患者姓名', '未知')}**")
        if st.button("生成个体化营养治疗方案", key="gen_plan_btn"):
            with st.spinner("正在分析..."):
                try:
                    plan = generate_plan(patient_for_plan)
                    st.session_state.ai_plan = plan
                except Exception as e:
                    st.session_state.ai_plan = f"❌ 生成失败：{str(e)}"
        if st.session_state.get("ai_plan"):
            st.text_area("AI 建议", value=st.session_state.ai_plan, height=400)
    
    #if st.button("🔄 重建知识库（更新 PDF 后使用）"):
    #    load_knowledge_base.clear()   # 清除缓存
    #    st.cache_resource.clear()     # 清除所有资源缓存（可选，更彻底）
    #    st.rerun()

    # ===== 患者列表与血糖曲线 =====
    st.subheader("📋 已录入患者列表")
    if is_admin:
        display_patients = st.session_state.patients
    else:
        display_patients = [p for p in st.session_state.patients if p.get("提交者ID") == submitter_id]
    if not display_patients:
        st.info("暂无数据")
    else:
        df_list = []
        for p in display_patients:
            followups = p.get("随访记录", [])
            n_fu = len(followups) if isinstance(followups, list) else 0
            last_fu = followups[-1]["随访时间"] if isinstance(followups, list) and n_fu > 0 else ""
            df_list.append({"患者姓名": p["患者姓名"], "性别": p.get("性别"), "年龄": p.get("年龄"), "随访次数": n_fu, "最近随访": last_fu})
        df_display = pd.DataFrame(df_list)
        st.dataframe(df_display, use_container_width=True)

        st.subheader("📈 血糖曲线分析")
        if len(display_patients) > 0:
            selected_patient_name = st.selectbox("选择患者", [p["患者姓名"] for p in display_patients], key="glucose_analysis")
            patient = next(p for p in display_patients if p["患者姓名"] == selected_patient_name)
            display_mode = st.radio("展示模式", ["单次随访对比", "全部随访展示"], horizontal=True, key="glucose_display_mode")
            pre_values = [patient.get("干预前FPG"), patient.get("干预前PG30"),
                          patient.get("干预前PG60"), patient.get("干预前PG120"),
                          patient.get("干预前PG180")]
            followups = patient.get("随访记录", [])
            pre_date_str = ""
            if patient.get("干预前5点日期"):
                pre_date_str = f" ({patient['干预前5点日期'].isoformat()})"

            if display_mode == "单次随访对比":
                followup_options = ["未选择"] + [f"第{i+1}次随访 ({r.get('随访时间', '')})" for i, r in enumerate(followups)]
                selected_followup_idx = st.selectbox("选择随访记录", range(len(followup_options)),
                                                     format_func=lambda x: followup_options[x], key="single_followup")
                if selected_followup_idx == 0:
                    if all(pre_values):
                        fig, auc = plot_glucose_curve(pre_values, f"干预前血糖曲线{pre_date_str}")
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
                    fig, pre_auc, post_auc = plot_combined_glucose_curve(pre_values, post_values,
                                                                         f"{selected_patient_name} - 干预前后对比")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1: st.metric("干预前 AUC", pre_auc)
                        with col2: st.metric("干预后 AUC", post_auc)
                    else:
                        st.info("该次随访数据不完整，无法绘图")
            else:
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
                    if all(pre_values):
                        fig.add_trace(go.Scatter(x=times, y=pre_values, mode='lines+markers',
                                                 name=f'干预前{pre_date_str}', line=dict(color='blue', width=3)))
                    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                    for idx, (i, rec, post_vals) in enumerate(valid_followups):
                        color = colors[idx % len(colors)]
                        followup_date = rec.get("随访时间", f"第{i+1}次")
                        fig.add_trace(go.Scatter(x=times, y=post_vals, mode='lines+markers',
                                                 name=f'随访{idx+1} ({followup_date})',
                                                 line=dict(color=color)))
                    fig.update_layout(title=f"{selected_patient_name} - 多点血糖对比",
                                      xaxis_title='时间 (小时)', yaxis_title='血糖 (mmol/L)',
                                      xaxis=dict(tickmode='array', tickvals=times,
                                                 ticktext=['空腹','0.5h','1h','2h','3h']))
                    st.plotly_chart(fig, use_container_width=True)

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

        # ===== 体感评分对比（可选） =====
        show_symptom = st.checkbox("📊 显示单项体感评分对比", value=False)
        if show_symptom:
            st.subheader("📊 单项体感评分对比")
            pre_symptom = patient.get("干预前体感子项", {})
            if not pre_symptom:
                st.info("无干预前体感数据")
            else:
                items = list(pre_symptom.keys())
                pre_vals = [pre_symptom.get(item) for item in items]
                
                # 干预前日期处理
                pre_symptom_date = patient.get("干预前体感日期")
                pre_label = "干预前"
                if pre_symptom_date:
                    pre_label += f" ({pre_symptom_date.isoformat()})"

                fig = go.Figure()
                fig.add_trace(go.Bar(name=pre_label, x=items, y=pre_vals, marker_color="blue"))

                colors_bar = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                valid_followups = [f for f in patient.get("随访记录", []) if f.get("干预后体感子项")]
                for idx, rec in enumerate(valid_followups):
                    post_symptom = rec.get("干预后体感子项", {})
                    post_vals = [post_symptom.get(item) for item in items]
                    color = colors_bar[idx % len(colors_bar)]
                    followup_label = f"随访{idx+1} ({rec.get('随访时间', '')})"
                    fig.add_trace(go.Bar(name=followup_label, x=items, y=post_vals, marker_color=color))

                fig.update_layout(
                    title=f"{selected_patient_name} - 体感评分对比",
                    xaxis_title="体感项目",
                    yaxis_title="评分 (0最差, 10最好)",
                    barmode='group',
                    yaxis=dict(range=[0, 10])
                )
                st.plotly_chart(fig, use_container_width=True)

        # ===== 生化指标趋势（可选） =====
        show_biochem = st.checkbox("📊 显示单项生化指标变化趋势", value=False)
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

            # 干预前生化日期处理
            pre_bio_date = patient.get("干预前生化日期")
            pre_bio_label = "干预前"
            if pre_bio_date:
                pre_bio_label += f" ({pre_bio_date.isoformat()})"

            timepoints = [pre_bio_label] + [f"随访{i+1}\n({r.get('随访时间', '')[:10]})" for i, r in enumerate(followups_list)]

            for field_name, pre_key, post_key, unit in bio_fields:
                pre_val = patient.get(pre_key)
                post_vals = []
                for rec in followups_list:
                    val = rec.get(post_key)
                    post_vals.append(val)
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

if __name__ == "__main__":
    patient_info_entry()