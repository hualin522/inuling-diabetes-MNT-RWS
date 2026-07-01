"""
血糖预测模块 —— LightGBM 回归 + LNN 衰减曲线轨迹
====================================================
两大预测能力:

1. 静态预测 (GlucosePredictor):
   LightGBM 回归, 基于 7,342 例 T2DM 患者数据训练,
   以 KNN(k=5) 填补缺失值后预测干预后约 30 天的 FPG/PG120。

2. 轨迹预测 (GlucoseTrajectoryPredictor):
   组合 LightGBM 点预测 + LNN 经验衰减曲线 ΔFPG(t) = A × (1-e^(-t/τ)),
   预测任意时间点 (15/30/90 天) 的血糖值并估算临床缓解时间。
   衰减参数 A=2.62, τ=49 来自 7,307 例数据的指数拟合。

使用方法:
    from glucose_predictor import GlucosePredictor, GlucoseTrajectoryPredictor

    # 静态预测
    pred = GlucosePredictor()
    result = pred.predict(age=65, sex=0, bmi=25.5, fpg=9.0, ...)

    # 轨迹预测 (多时间点 + 缓解时间估算)
    traj = GlucoseTrajectoryPredictor()
    result = traj.predict_at(patient_dict, days=[15, 30, 90])
    remission = traj.estimate_remission(patient_dict)

缺失值处理: 任意特征可传入 None, 由 KNN 自动估算。
至少需要提供一项特征即可预测; 推荐至少提供年龄、BMI 和 FPG。
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
_trajectory_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GlucosePredictor()
    return _predictor_instance


# ============================================================
# LNN 衰减曲线参数（基于 7,307 例数据拟合）
# ============================================================
DEFAULT_A_FPG = 2.62       # FPG 最大改善幅度 (mmol/L)
DEFAULT_TAU = 49.0         # 时间常数 (天): 63% 改善在此天达成
DEFAULT_A_PG120 = 4.50     # PG120 改善幅度大于 FPG
DEFAULT_TAU_PG120 = 45.0

# 临床缓解阈值
FPG_REMISSION = 7.0        # 空腹血糖 < 7.0
PG120_REMISSION = 11.1     # 餐后2h < 11.1


class GlucoseTrajectoryPredictor:
    """组合 LightGBM 点预测 + LNN 衰减曲线的血糖轨迹预测器。

    原理: ΔFPG(t) = A × (1 − e^(−t/τ))
    A 和 τ 由患者特征（FPG、年龄、BMI）个性化调制。

    参数优先从 glucose_model/metadata.json 的 trajectory_params 字段读取,
    若 JSON 中不存在则回退到硬编码默认值。
    """

    def __init__(self, A_fpg=None, tau_fpg=None,
                 A_pg120=None, tau_pg120=None):
        # 优先从 metadata.json 读取; 若无则用默认值
        tp = self._load_trajectory_meta()
        self.A_fpg = A_fpg if A_fpg is not None else tp["A_fpg"]
        self.tau_fpg = tau_fpg if tau_fpg is not None else tp["tau_fpg"]
        self.A_pg120 = A_pg120 if A_pg120 is not None else tp["A_pg120"]
        self.tau_pg120 = tau_pg120 if tau_pg120 is not None else tp["tau_pg120"]
        self._static_predictor = None

    @staticmethod
    def _load_trajectory_meta():
        """从 metadata.json 加载轨迹参数, 失败则返回默认值。"""
        try:
            meta_path = os.path.join(_MODEL_DIR, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tp = meta.get("trajectory_params", {})
            return {
                "A_fpg": tp.get("A_fpg", DEFAULT_A_FPG),
                "tau_fpg": tp.get("tau_fpg", DEFAULT_TAU),
                "A_pg120": tp.get("A_pg120", DEFAULT_A_PG120),
                "tau_pg120": tp.get("tau_pg120", DEFAULT_TAU_PG120),
            }
        except Exception:
            return {"A_fpg": DEFAULT_A_FPG, "tau_fpg": DEFAULT_TAU,
                    "A_pg120": DEFAULT_A_PG120, "tau_pg120": DEFAULT_TAU_PG120}

    def _ensure_static(self):
        if self._static_predictor is None:
            self._static_predictor = GlucosePredictor()
            self._static_predictor._ensure_loaded()
        return self._static_predictor

    # ---------- 个性化参数调制 ----------

    def _modulate_params(self, patient):
        """根据患者特征调制衰减参数。返回 (A_fpg, tau, A_pg120, tau_pg120)。"""
        fpg = float(patient.get("干预前FPG", patient.get("fpg", 9.0)))
        age = float(patient.get("年龄", patient.get("age", 60)))
        bmi = float(patient.get("干预前BMI", patient.get("bmi", 26)))

        fpg_ratio = np.clip(1.0 + (fpg - 9.0) / 3.0 * 0.2, 0.5, 1.8)
        age_ratio = np.clip(1.0 + (age - 60) / 10.0 * 0.1, 0.7, 1.4)
        bmi_ratio = np.clip(1.0 + (bmi - 26) / 5.0 * 0.1, 0.8, 1.3)

        return (self.A_fpg * fpg_ratio,
                self.tau_fpg * age_ratio * bmi_ratio,
                self.A_pg120 * fpg_ratio * 0.85,
                self.tau_pg120 * age_ratio * bmi_ratio)

    # ---------- 轨迹预测 ----------

    @staticmethod
    def _decay(t, A, tau):
        return A * (1 - np.exp(-np.asarray(t, dtype=float) / tau))

    def predict_at(self, patient, days):
        """预测患者在干预 days 天后的血糖。

        Args:
            patient: 患者数据字典
            days: 目标天数 (int 或 list)

        Returns:
            dict: {"fpg": float|list, "pg120": ..., "delta_fpg": ..., ...}
        """
        A_fpg, tau, A_pg120, tau_pg120 = self._modulate_params(patient)
        days_arr = np.atleast_1d(np.asarray(days, dtype=float))
        is_scalar = np.isscalar(days)

        # 静态模型基准预测
        static_pred = None
        try:
            static_pred = self._ensure_static().predict_from_dict(patient)
        except Exception:
            pass

        pre_fpg = float(patient.get("干预前FPG", patient.get("fpg",
                          static_pred.get("pre_fpg", 9.0) if static_pred else 9.0)))
        pre_pg120 = float(patient.get("干预前PG120", patient.get("pg120",
                            static_pred.get("pre_pg120", 14.0) if static_pred else 14.0)))

        delta_fpg = self._decay(days_arr, A_fpg, tau)
        delta_pg120 = self._decay(days_arr, A_pg120, tau_pg120)
        pred_fpg = pre_fpg - delta_fpg
        pred_pg120 = pre_pg120 - delta_pg120

        # 若接近 30 天且有静态预测，用静态模型校准
        if static_pred:
            mask_30 = np.abs(days_arr - 30) <= 5
            if mask_30.any():
                pred_fpg[mask_30] = static_pred.get("post_fpg", pred_fpg[mask_30])
                pred_pg120[mask_30] = static_pred.get("post_pg120", pred_pg120[mask_30])

        result = {
            "days": days_arr,
            "pre_fpg": pre_fpg, "pre_pg120": pre_pg120,
            "fpg": pred_fpg if not is_scalar else float(pred_fpg[0]),
            "pg120": pred_pg120 if not is_scalar else float(pred_pg120[0]),
            "delta_fpg": delta_fpg if not is_scalar else float(delta_fpg[0]),
            "delta_pg120": delta_pg120 if not is_scalar else float(delta_pg120[0]),
            "a_fpg": A_fpg, "tau_fpg": tau,
            "a_pg120": A_pg120, "tau_pg120": tau_pg120,
        }
        if static_pred:
            result["static_30d"] = static_pred
        return result

    # ---------- 临床缓解时间估算 ----------

    def estimate_remission(self, patient):
        """估算达成临床缓解 (FPG<7.0 且 PG120<11.1) 所需天数。"""
        A_fpg, tau, A_pg120, tau_pg120 = self._modulate_params(patient)
        pre_fpg = float(patient.get("干预前FPG", patient.get("fpg", 9.0)))
        pre_pg120 = float(patient.get("干预前PG120", patient.get("pg120", 14.0)))

        def _inv_decay(target, pre, A, tau):
            delta_needed = pre - target
            if delta_needed <= 0:
                return 0
            if delta_needed >= A * 0.95:
                return None
            ratio = delta_needed / A
            return -tau * np.log(1 - ratio) if ratio < 1.0 else None

        fpg_days = _inv_decay(FPG_REMISSION, pre_fpg, A_fpg, tau)
        pg120_days = _inv_decay(PG120_REMISSION, pre_pg120, A_pg120, tau_pg120)

        combined = None
        if fpg_days is not None and pg120_days is not None:
            combined = max(fpg_days, pg120_days)
        elif fpg_days is not None:
            combined = fpg_days
        elif pg120_days is not None:
            combined = pg120_days

        # 分析文本
        lines = [f"基线 FPG={pre_fpg:.1f}, PG120={pre_pg120:.1f}"]
        fpg_s = f"{fpg_days:.0f}d" if fpg_days else "已达标"
        pg120_s = f"{pg120_days:.0f}d" if pg120_days else "已达标"
        lines.append(f"FPG 达标 (<{FPG_REMISSION}): 预计 {fpg_s}")
        lines.append(f"PG120 达标 (<{PG120_REMISSION}): 预计 {pg120_s}")

        if combined is not None:
            lines.append(f"综合缓解: 预计 {combined:.0f} 天 ({combined/30:.1f} 月)")
            if combined <= 30:
                lines.append("结论: 短期干预 (≤1月) 可达成缓解。")
            elif combined <= 90:
                lines.append("结论: 中等周期 (1-3月) 可达成缓解。")
            elif combined <= 180:
                lines.append("结论: 需较长周期 (3-6月)。")
            else:
                lines.append("结论: 需 6 月以上。推荐联合药物辅助。")
        else:
            lines.append("结论: 以群组平均成效估算, 缓解暂不可预期。")

        return {
            "fpg_days": fpg_days, "pg120_days": pg120_days,
            "combined_days": combined,
            "remission_likely": combined is not None and combined <= 365,
            "analysis": "\n".join(lines),
        }

    # ---------- 多时间点 + 相似案例 ----------

    def predict_with_similar_cases(self, patient, similar_patients,
                                    timepoints=(15, 30, 90)):
        """结合相似案例校准的多时间点轨迹预测。"""
        traj = self.predict_at(patient, list(timepoints))
        remission = self.estimate_remission(patient)

        # 相似案例基准
        benchmarks = {"n_similar": len(similar_patients) if similar_patients else 0}
        if similar_patients:
            deltas_f, deltas_p, durations = [], [], []
            for p in similar_patients:
                try:
                    pre_f = float(p.get("干预前FPG", p.get("干预前_FPG", None)))
                    post_f = float(p.get("干预后T1_FPG", p.get("干预后FPG", None)))
                    pre_p = float(p.get("干预前PG120", p.get("干预前_PG_120min", None)))
                    post_p = float(p.get("干预后T1_PG_120min", p.get("干预后PG120", None)))
                    dur = float(p.get("干预天数", p.get("duration", None)))
                    if all(v is not None and not np.isnan(v) for v in [pre_f, post_f, dur]):
                        deltas_f.append(pre_f - post_f); durations.append(dur)
                    if all(v is not None and not np.isnan(v) for v in [pre_p, post_p]):
                        deltas_p.append(pre_p - post_p)
                except (ValueError, KeyError, TypeError):
                    continue
            benchmarks["n_with_outcome_fpg"] = len(deltas_f)
            if deltas_f:
                benchmarks["similar_median_delta_fpg"] = round(float(np.median(deltas_f)), 2)
                benchmarks["similar_median_duration"] = round(float(np.median(durations)), 0)
            if deltas_p:
                benchmarks["similar_median_delta_pg120"] = round(float(np.median(deltas_p)), 2)

        summary = self._format_summary(traj, remission, benchmarks, timepoints)
        return {"trajectory": traj, "benchmarks": benchmarks,
                "remission": remission, "summary_text": summary}

    def _format_summary(self, traj, remission, benchmarks, timepoints):
        days = np.atleast_1d(traj["days"])
        fpgs = np.atleast_1d(traj["fpg"])
        pg120s = np.atleast_1d(traj["pg120"])
        deltas_f = np.atleast_1d(traj["delta_fpg"])
        deltas_p = np.atleast_1d(traj["delta_pg120"])

        lines = [
            "【LNN 连续时间血糖轨迹预测】（基于 7,307 例衰减曲线 + 个性化调制）",
            "",
            "预测的多时间点血糖值:",
        ]
        for i in range(len(days)):
            lines.append(f"  - 干预 {int(days[i])} 天: "
                         f"FPG ≈ {fpgs[i]:.1f} (Δ{deltas_f[i]:.1f}), "
                         f"PG120 ≈ {pg120s[i]:.1f} (Δ{deltas_p[i]:.1f})")
        lines.append("")
        lines.append(f"个性化参数: A_FPG={traj['a_fpg']:.2f}, τ={traj['tau_fpg']:.0f}d")

        lines.append(""); lines.append("临床缓解估计:")
        if remission["combined_days"] is not None:
            lines.append(f"  预计 {remission['combined_days']:.0f} 天 "
                         f"({remission['combined_days']/30:.1f} 月) "
                         f"达成临床缓解 (FPG<{FPG_REMISSION}, PG120<{PG120_REMISSION})")
        else:
            lines.append("  ⚠ 以群组平均成效估算, 难以达成临床缓解。")

        if benchmarks.get("similar_median_delta_fpg"):
            lines.append("")
            lines.append(f"相似患者 ({benchmarks['n_similar']} 例) 中位改善: "
                         f"ΔFPG={benchmarks['similar_median_delta_fpg']}")

        return "\n".join(lines)


# ============================================================
# 预测质量评级系统 (Prediction Quality Assessor)
# ============================================================

class PredictionQualityAssessor:
    """预测数据质量评级器。

    基于四维度评分, 给出 A/B/C/D 综合等级:
      1. 数据完整度 (Completeness): 实填特征数 / 总特征数
      2. 模型置信度 (Confidence): 基于 CV 残差分布的预期误差区间
      3. 特征可靠性 (Reliability): 核心特征(FPG)是否离群
      4. 方法一致性 (Consensus): LightGBM vs LNN vs 相似案例的一致性

    等级含义:
      A: 高置信度 (MAE 预期 < 0.8, 可直接采纳)
      B: 中等置信度 (MAE 预期 0.8-1.2, 可参考但需验证)
      C: 低置信度 (MAE 预期 1.2-2.0, 仅作趋势参考)
      D: 不可靠 (MAE 预期 > 2.0, 建议补充数据后重新预测)
    """

    # CV 分析得出的基准值
    BASELINE_MAE = 1.054       # 总体 MAE
    ERROR_P50 = 0.68           # 中位绝对误差
    ERROR_P90 = 2.38           # 90 分位绝对误差

    # 特征关键性权重 (基于 7,342 例分析)
    FEATURE_WEIGHTS = {
        "干预前_FPG": 5.0,     # 最重要, 缺则难以预测
        "干预前_PG_120min": 4.0,
        "BMI": 3.0,
        "年龄": 2.5,
        "干预前_PG_60min": 2.0,
        "干预前_PG_180min": 2.0,
        "病史_年": 1.5,
        "干预前_PG_30min": 1.5,
        "干预天数": 1.0,
        "干预前_体感总分": 0.5,
        "性别编码": 0.3,
    }

    def __init__(self):
        self._meta_loaded = False
        self._feature_means = {}
        self._feature_stds = {}

    def _load_meta(self):
        if self._meta_loaded:
            return
        try:
            meta_path = os.path.join(_MODEL_DIR, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._feature_means = {k: v["mean"]
                for k, v in meta.get("feature_stats", {}).items()}
            self._feature_stds = {k: v["std"]
                for k, v in meta.get("feature_stats", {}).items()}
        except Exception:
            pass
        self._meta_loaded = True

    def assess(self, patient, static_result=None, trajectory_result=None,
               similar_benchmarks=None):
        """
        综合评估预测质量。

        Returns:
            dict with:
              grade: 'A'|'B'|'C'|'D'
              scores: {completeness, confidence, reliability, consensus} (0-100)
              expected_mae: 预期 MAE 范围
              recommendations: 改进建议列表
              summary: 中文评估摘要
        """
        self._load_meta()

        scores = {}
        recommendations = []

        # ---- 维度1: 数据完整度 ----
        provided_feats = []
        missing_feats = []
        for feat in FEATURES:
            v = patient.get(feat)
            if v is not None and v != "" and v != "nan":
                provided_feats.append(feat)
            else:
                missing_feats.append(feat)

        # 加权完整度: 关键特征权重高
        total_weight = sum(self.FEATURE_WEIGHTS.get(f, 1.0) for f in FEATURES)
        provided_weight = sum(self.FEATURE_WEIGHTS.get(f, 1.0) for f in provided_feats)
        completeness = (provided_weight / total_weight) * 100 if total_weight > 0 else 0
        scores["completeness"] = round(completeness, 1)

        n_missing = len(missing_feats)
        if n_missing == 0:
            pass  # 全量数据
        elif n_missing <= 2:
            recommendations.append(f"缺少 {n_missing} 项特征 ({', '.join(missing_feats)}), 轻微影响预测精度")
        elif n_missing <= 5:
            recommendations.append(f"缺少 {n_missing} 项特征, 建议补充以提高置信度")
        else:
            recommendations.append(f"数据严重不完整 (缺 {n_missing} 项), 预测仅供参考")
            scores["completeness"] = max(scores["completeness"], 20.0)

        # ---- 维度2: 模型置信度 (基于特征分布 + 缺失数) ----
        # 规则: 越接近训练集均值, 预测越可靠
        reliability_penalty = 0
        fpg_raw = patient.get("干预前_FPG")
        if fpg_raw is not None:
            try:
                fpg_val = float(fpg_raw)
                fpg_mean = self._feature_means.get("干预前_FPG", 9.4)
                fpg_std = self._feature_stds.get("干预前_FPG", 3.0)
                z_score = abs(fpg_val - fpg_mean) / max(fpg_std, 1.0)
                if z_score > 2.0:
                    reliability_penalty += 15  # 极度离群
                    recommendations.append(f"基线 FPG={fpg_val:.1f} 偏离均值较远 (z={z_score:.1f}), "
                                          f"预测不确定性增大")
                elif z_score > 1.0:
                    reliability_penalty += 5
            except (ValueError, TypeError):
                pass

        # 缺失数惩罚 (基于实验: 0 missing=MAE 1.03, 6+ missing=MAE 1.47)
        if n_missing == 0:
            conf_base = 85
        elif n_missing <= 3:
            conf_base = 80
        elif n_missing <= 5:
            conf_base = 70
        else:
            conf_base = 55

        confidence = max(conf_base - reliability_penalty, 30.0)
        scores["confidence"] = round(confidence, 1)

        # 预期 MAE
        if confidence >= 80:
            expected_mae = (0.8, 1.0)
        elif confidence >= 60:
            expected_mae = (1.0, 1.5)
        elif confidence >= 40:
            expected_mae = (1.5, 2.5)
        else:
            expected_mae = (2.0, 4.0)

        # ---- 维度3: 特征可靠性 ----
        reliability = 100.0
        for feat in provided_feats:
            if feat in self._feature_means and feat in self._feature_stds:
                try:
                    val = float(patient[feat])
                    mean = self._feature_means[feat]
                    std = max(self._feature_stds[feat], 1.0)
                    z = abs(val - mean) / std
                    if z > 3.0:
                        reliability -= 10
                        if z > 5.0 and feat in ("干预前_FPG", "BMI"):
                            recommendations.append(
                                f"⚠ {feat}={val:.1f} 为极端值 (z={z:.1f}), 请核实数据")
                except (ValueError, TypeError):
                    reliability -= 2
        scores["reliability"] = round(max(reliability, 0.0), 1)

        # ---- 维度4: 方法一致性 ----
        consensus = 50.0  # 默认: 无其他预测方法可供比较
        if static_result and trajectory_result:
            static_fpg = static_result.get("post_fpg")
            traj_30d = None
            days_arr = np.atleast_1d(trajectory_result.get("days", []))
            fpg_arr = np.atleast_1d(trajectory_result.get("fpg", []))
            mask_30 = np.abs(days_arr - 30) <= 5
            if mask_30.any():
                traj_30d = float(fpg_arr[mask_30][0])

            if static_fpg and traj_30d:
                diff = abs(static_fpg - traj_30d)
                if diff < 0.5:
                    consensus = 95  # 高度一致
                elif diff < 1.0:
                    consensus = 80
                elif diff < 2.0:
                    consensus = 60
                    recommendations.append(f"LightGBM 与 LNN 轨迹预测差异 {diff:.1f} mmol/L, "
                                          f"建议参考相似案例辅助判断")
                else:
                    consensus = 40
                    recommendations.append(f"预测方法分歧较大 ({diff:.1f} mmol/L), "
                                          f"建议收集更多特征后重新预测")

                # 相似案例校准
                if similar_benchmarks and similar_benchmarks.get("similar_median_delta_fpg"):
                    sim_delta = similar_benchmarks["similar_median_delta_fpg"]
                    ml_delta = static_result.get("delta_fpg", 0)
                    if ml_delta and abs(sim_delta - ml_delta) < 1.5:
                        consensus = min(consensus + 10, 100)
                    elif ml_delta and abs(sim_delta - ml_delta) > 3.0:
                        consensus = max(consensus - 15, 30)
                        recommendations.append("ML 预测与相似案例差异较大, 综合置信度降低")

        scores["consensus"] = round(consensus, 1)

        # ---- 综合等级 ----
        overall = (scores["completeness"] * 0.25 +
                   scores["confidence"] * 0.30 +
                   scores["reliability"] * 0.15 +
                   scores["consensus"] * 0.30)

        if overall >= 80:
            grade = "A"
            grade_desc = "高置信度——可直接采纳预测结果"
        elif overall >= 60:
            grade = "B"
            grade_desc = "中等置信度——可参考, 建议随访验证"
        elif overall >= 40:
            grade = "C"
            grade_desc = "低置信度——仅作趋势参考, 需更多数据支持"
        else:
            grade = "D"
            grade_desc = "不可靠——建议补充核心特征后重新预测"

        # 摘要
        summary_lines = [
            f"预测质量等级: {grade} ({grade_desc})",
            f"综合评分: {overall:.0f}/100",
            f"预期误差: MAE ≈ {expected_mae[0]:.1f}–{expected_mae[1]:.1f} mmol/L",
            f"数据完整度: {scores['completeness']:.0f}% | "
            f"模型置信度: {scores['confidence']:.0f}% | "
            f"特征可靠性: {scores['reliability']:.0f}% | "
            f"方法一致性: {scores['consensus']:.0f}%",
        ]
        if recommendations:
            summary_lines.append("改进建议:")
            for r in recommendations:
                summary_lines.append(f"  · {r}")

        return {
            "grade": grade,
            "grade_desc": grade_desc,
            "overall": round(overall, 1),
            "scores": scores,
            "expected_mae": expected_mae,
            "recommendations": recommendations,
            "summary": "\n".join(summary_lines),
            "n_missing": n_missing,
            "missing_features": missing_feats,
            "n_provided": len(provided_feats),
        }


def assess_prediction_quality(patient, static_result=None,
                               trajectory_result=None, similar_benchmarks=None):
    """一站式预测质量评估。"""
    assessor = PredictionQualityAssessor()
    return assessor.assess(patient, static_result, trajectory_result, similar_benchmarks)


def get_trajectory_predictor():
    """获取全局轨迹预测器（单例, 懒初始化）。"""
    global _trajectory_instance
    if _trajectory_instance is None:
        _trajectory_instance = GlucoseTrajectoryPredictor()
    return _trajectory_instance


def multi_timepoint_prediction(patient, similar_patients=None,
                                timepoints=(15, 30, 90)):
    """一站式多时间点预测（供外部直接调用）。"""
    tp = get_trajectory_predictor()
    return tp.predict_with_similar_cases(patient, similar_patients or [],
                                          timepoints)
