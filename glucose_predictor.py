"""
血糖预测模块 —— LightGBM 回归模型
======================================
基于 7,342 例 T2DM 患者营养干预数据训练。
模型以 KNN(k=5) 填补缺失值 + StandardScaler 标准化后输入 LightGBM。

使用方法:
    from glucose_predictor import GlucosePredictor
    pred = GlucosePredictor()
    result = pred.predict(age=65, sex=0, bmi=25.5, duration=8.0, tcm_score=105,
                          fpg=9.0, pg30=12.0, pg60=15.0, pg120=14.5, pg180=12.0)
    # result = {"post_fpg": 7.96, "post_pg120": 8.51, "delta_fpg": 1.04, "delta_pg120": 5.99}

缺失值处理: 任意特征可传入 None，由 KNN 自动估算。
至少需要提供一项特征即可预测；推荐至少提供年龄、BMI 和 FPG。
"""
import os
import pickle
import json
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "glucose_model")

FEATURES = [
    "年龄", "性别编码", "BMI", "病史_年", "干预前_体感总分",
    "干预前_FPG", "干预前_PG_30min", "干预前_PG_60min",
    "干预前_PG_120min", "干预前_PG_180min",
]

FEATURE_DEFAULTS = {
    # 从训练集均值推算的典型默认值（当全部留空时的 fallback）
    "年龄": 66.0, "性别编码": 0.4, "BMI": 26.0, "病史_年": 18.0,
    "干预前_体感总分": 105.0,
    "干预前_FPG": 9.4, "干预前_PG_30min": 12.2,
    "干预前_PG_60min": 15.0, "干预前_PG_120min": 14.4,
    "干预前_PG_180min": 12.1,
}


class GlucosePredictor:
    """LightGBM 血糖预测器，懒加载模型文件。"""

    def __init__(self):
        self._fpg_model = None
        self._pg120_model = None
        self._scaler = None
        self._imputer = None
        self._meta = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        with open(os.path.join(_MODEL_DIR, "model_fpg.pkl"), "rb") as f:
            self._fpg_model = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "model_pg120.pkl"), "rb") as f:
            self._pg120_model = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "scaler.pkl"), "rb") as f:
            self._scaler = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "imputer.pkl"), "rb") as f:
            self._imputer = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "metadata.json"), "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        self._loaded = True

    def predict(self, age=None, sex=None, bmi=None, duration=None,
                tcm_score=None, fpg=None, pg30=None, pg60=None,
                pg120=None, pg180=None):
        """
        预测干预后血糖。

        Parameters
        ----------
        所有参数为 float 或 None。None 表示该特征缺失，由 KNN 自动填补。

        Returns
        -------
        dict with keys:
            post_fpg: 预测干预后空腹血糖 (mmol/L)
            post_pg120: 预测干预后餐后120min血糖 (mmol/L)
            delta_fpg: 预测 FPG 改善幅度 = pre_fpg - post_fpg（仅当 fpg 非 None）
            delta_pg120: 预测 PG120 改善幅度（仅当 pg120 非 None）
            n_provided: 用户实际提供的特征数
            n_total: 总特征数 (10)
            missing_features: 被填补的特征名列表
        """
        self._ensure_loaded()

        raw_values = {
            "年龄": age, "性别编码": sex, "BMI": bmi,
            "病史_年": duration, "干预前_体感总分": tcm_score,
            "干预前_FPG": fpg, "干预前_PG_30min": pg30,
            "干预前_PG_60min": pg60, "干预前_PG_120min": pg120,
            "干预前_PG_180min": pg180,
        }

        X_list = []
        missing = []
        for ft in FEATURES:
            v = raw_values[ft]
            if v is None:
                X_list.append(np.nan)
                missing.append(ft)
            else:
                X_list.append(float(v))

        n_provided = len(FEATURES) - len(missing)

        # KNN imputation + scaling + predict
        X = np.array([X_list])
        X_imp = self._imputer.transform(X)
        X_scaled = self._scaler.transform(X_imp)

        post_fpg = round(float(self._fpg_model.predict(X_scaled)[0]), 2)
        post_pg120 = round(float(self._pg120_model.predict(X_scaled)[0]), 2)

        result = {
            "post_fpg": post_fpg,
            "post_pg120": post_pg120,
            "n_provided": n_provided,
            "n_total": len(FEATURES),
            "missing_features": missing,
        }

        if fpg is not None:
            result["delta_fpg"] = round(fpg - post_fpg, 2)
        if pg120 is not None:
            result["delta_pg120"] = round(pg120 - post_pg120, 2)

        return result

    def predict_from_dict(self, patient_dict):
        """
        从字典形式的患者数据预测。

        支持的键名（兼容 Google Sheets 列名和内部变量名）:
            年龄 / 干预前年龄, 性别编码 / 性别,
            BMI / 干预前BMI, 病史_年 / 病史年,
            干预前_体感总分, 干预前_FPG / 干预前FPG,
            干预前_PG_30min, 干预前_PG_60min,
            干预前_PG_120min / 干预前PG120,
            干预前_PG_180min
        """
        def _get(key, aliases):
            for k in aliases:
                v = patient_dict.get(k)
                if v is not None and v != "" and v != "nan":
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            return None

        sex_raw = patient_dict.get("性别", patient_dict.get("性别编码"))
        sex_code = None
        if sex_raw is not None:
            s = str(sex_raw).strip()
            if s in ("男", "1", "male", "Male"):
                sex_code = 1.0
            elif s in ("女", "0", "female", "Female"):
                sex_code = 0.0

        return self.predict(
            age=_get("年龄", ["干预前年龄", "年龄"]),
            sex=sex_code,
            bmi=_get("BMI", ["干预前BMI", "BMI"]),
            duration=_get("病史_年", ["病史年", "病史_年", "干预前病史年"]),
            tcm_score=_get("干预前_体感总分", ["干预前体感总分", "干预前_体感总分"]),
            fpg=_get("干预前_FPG", ["干预前FPG", "干预前_FPG"]),
            pg30=_get("干预前_PG_30min", ["干预前PG30", "干预前_PG_30min"]),
            pg60=_get("干预前_PG_60min", ["干预前PG60", "干预前_PG_60min"]),
            pg120=_get("干预前_PG_120min", ["干预前PG120", "干预前_PG_120min"]),
            pg180=_get("干预前_PG_180min", ["干预前PG180", "干预前_PG_180min"]),
        )

    def format_prediction_text(self, patient_dict):
        """
        生成用于注入 AI prompt 的结构化预测文本。
        返回 (prediction_text, ml_context_summary)。
        """
        result = self.predict_from_dict(patient_dict)
        lines = []
        lines.append("【机器学习模型预测（LightGBM 回归）】")
        lines.append(f"  模型信息: 基于 {self._meta.get('n_fpg', '?')} 例患者训练，5折交叉验证 R²≈0.45")
        lines.append(f"  数据完整度: {result['n_provided']}/{result['n_total']} 项已提供")

        if result['missing_features']:
            missing_cn = [f.split('_')[-1] if '_' in f else f for f in result['missing_features']]
            lines.append(f"  缺失填补: KNN 估算了 {', '.join(missing_cn)}")

        lines.append(f"  预测干预后空腹血糖 (FPG): {result['post_fpg']} mmol/L")
        if 'delta_fpg' in result:
            direction = "↓下降" if result['delta_fpg'] > 0 else ("↑上升" if result['delta_fpg'] < 0 else "→持平")
            lines.append(f"    预计改善: {direction} {abs(result['delta_fpg'])} mmol/L")

        lines.append(f"  预测干预后餐后2h血糖 (PG120): {result['post_pg120']} mmol/L")
        if 'delta_pg120' in result:
            direction = "↓下降" if result['delta_pg120'] > 0 else ("↑上升" if result['delta_pg120'] < 0 else "→持平")
            lines.append(f"    预计改善: {direction} {abs(result['delta_pg120'])} mmol/L")

        lines.append("  [重要说明] 该预测为统计估计值，模型 MAE≈1.06 mmol/L，实际效果存在个体差异。")

        prediction_text = "\n".join(lines)

        # Short summary for embedding into the main pre-template
        ml_summary = (
            f"● ML模型预测: 干预后FPG={result['post_fpg']}mmol/L"
            f"{' (↓'+str(result['delta_fpg'])+')' if 'delta_fpg' in result else ''}"
            f", 干预后PG120={result['post_pg120']}mmol/L"
            f"{' (↓'+str(result['delta_pg120'])+')' if 'delta_pg120' in result else ''}"
            f", 模型R²≈0.45, MAE≈1.06"
        )
        if result['missing_features']:
            ml_summary += f", 含KNN填补项({result['n_provided']}/{result['n_total']})"

        return prediction_text, ml_summary


# 全局单例
_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GlucosePredictor()
    return _predictor_instance
