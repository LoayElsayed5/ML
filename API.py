from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import joblib
import json
import io
import re

# Optional libs for PDF / OCR
# pip install pdfplumber pillow pytesseract pdf2image python-multipart openpyxl
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None


BASE_DIR = Path(__file__).resolve().parent

# ---------------------------
# Load Artifacts
# ---------------------------
gdm_artifact = joblib.load(BASE_DIR / "gdm_artifact.pkl")
gdm_model = gdm_artifact["model"]
gdm_scaler = gdm_artifact["scaler"]
gdm_threshold = gdm_artifact["threshold"]
gdm_feature_order = gdm_artifact["feature_order"]

preeclampsia_artifact = joblib.load(BASE_DIR / "preeclampsia_artifact.pkl")
preeclampsia_model = preeclampsia_artifact["model"]
preeclampsia_scaler = preeclampsia_artifact["scaler"]
preeclampsia_threshold = preeclampsia_artifact["threshold"]
preeclampsia_raw_feature_order = preeclampsia_artifact["raw_feature_order"]
preeclampsia_processed_feature_names = preeclampsia_artifact["feature_names"]


app = FastAPI(title="Her Journey Medical Prediction API")


# ---------------------------
# Models
# ---------------------------
class GDMInput(BaseModel):
    Age: int
    No_of_Pregnancy: int
    Gestation_in_previous_Pregnancy: int
    BMI: float
    HDL: float
    Family_History: int
    unexplained_prenetal_loss: int
    Large_Child_or_Birth_Default: int
    PCOS: int
    Sys: float
    dia: int
    OGTT: float
    Hemoglobin: float
    Sedentary_Lifestyle: int
    Prediabetes: int


class PreeclampsiaInput(BaseModel):
    age: int
    parity: int
    gravida: int
    bmi: float
    gestational_age_weeks: int
    chronic_hypertension: int
    pregestational_diabetes: int
    chronic_kidney_disease: int
    multiple_pregnancy: int
    previous_preeclampsia: int
    family_history_preeclampsia: int
    antiphospholipid_syndrome: int

    platelets_k_ul: float
    ast_u_l: float
    alt_u_l: float
    creatinine_mg_dl: float
    ldh_u_l: float
    uric_acid_mg_dl: float
    hemoglobin_g_dl: float

    headache: int
    visual_disturbances: int
    epigastric_pain: int
    edema: int
    nausea_vomiting: int
    fetal_growth_restriction: int
    acute_kidney_injury: int
    pulmonary_edema: int


class PredictResponse(BaseModel):
    label: str
    probability: float


class FilePredictResponse(BaseModel):
    label: Optional[str] = None
    probability: Optional[float] = None
    extracted_features: Dict[str, Any]
    missing_fields: List[str]
    message: str


# ---------------------------
# GDM Config
# ---------------------------
INT_FEATURES = {
    "Age",
    "No_of_Pregnancy",
    "Gestation_in_previous_Pregnancy",
    "Family_History",
    "unexplained_prenetal_loss",
    "Large_Child_or_Birth_Default",
    "PCOS",
    "dia",
    "Sedentary_Lifestyle",
    "Prediabetes",
}

FLOAT_FEATURES = {
    "BMI",
    "HDL",
    "Sys",
    "OGTT",
    "Hemoglobin",
}

BINARY_FEATURES = {
    "Family_History",
    "unexplained_prenetal_loss",
    "Large_Child_or_Birth_Default",
    "PCOS",
    "Sedentary_Lifestyle",
    "Prediabetes",
}

FEATURE_ALIASES = {
    "Age": [
        "age", "patient age"
    ],
    "No_of_Pregnancy": [
        "no_of_pregnancy", "number of pregnancy", "no of pregnancy",
        "gravida", "pregnancy count"
    ],
    "Gestation_in_previous_Pregnancy": [
        "gestation_in_previous_pregnancy",
        "gestation in previous pregnancy",
        "previous gestation", "previous pregnancy gestation"
    ],
    "BMI": [
        "bmi", "body mass index"
    ],
    "HDL": [
        "hdl", "hdl cholesterol", "high density lipoprotein"
    ],
    "Family_History": [
        "family_history", "family history", "fh"
    ],
    "unexplained_prenetal_loss": [
        "unexplained_prenetal_loss",
        "unexplained prenatal loss",
        "prenatal loss",
        "prenetal loss"
    ],
    "Large_Child_or_Birth_Default": [
        "large_child_or_birth_default",
        "large child or birth default",
        "large child",
        "birth defect",
        "birth default",
        "macrosomia"
    ],
    "PCOS": [
        "pcos", "polycystic ovary syndrome"
    ],
    "Sys": [
        "sys", "systolic", "sbp", "blood pressure"
    ],
    "dia": [
        "dia", "diastolic", "dbp", "blood pressure"
    ],
    "OGTT": [
        "ogtt", "oral glucose tolerance test", "glucose tolerance test"
    ],
    "Hemoglobin": [
        "hemoglobin", "haemoglobin", "hb", "hgb"
    ],
    "Sedentary_Lifestyle": [
        "sedentary_lifestyle", "sedentary lifestyle", "physical inactivity"
    ],
    "Prediabetes": [
        "prediabetes", "pre-diabetes", "pre diabetes"
    ],
}


# ---------------------------
# GDM Helpers
# ---------------------------
def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def alias_to_pattern(alias: str) -> str:
    alias = alias.lower().strip()
    alias = re.escape(alias)
    alias = alias.replace(r"\ ", r"[\s_\-]*")
    alias = alias.replace(r"\_", r"[\s_\-]*")
    return alias


def cast_value(feature: str, value: Any):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

    # Binary string values
    if feature in BINARY_FEATURES and isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "positive", "present", "true", "1"}:
            return 1
        if v in {"no", "negative", "absent", "false", "0"}:
            return 0

    try:
        if feature in INT_FEATURES:
            return int(float(value))
        if feature in FLOAT_FEATURES:
            return float(value)
    except Exception:
        return None

    return value


def extract_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    normalized_row = {normalize_key(k): v for k, v in row.items()}

    result = {}

    for feature in gdm_feature_order:
        aliases = FEATURE_ALIASES.get(feature, []) + [feature]
        for alias in aliases:
            nk = normalize_key(alias)
            if nk in normalized_row and pd.notna(normalized_row[nk]):
                result[feature] = cast_value(feature, normalized_row[nk])
                break

    return result


def extract_bp(text: str):
    """
    Try to extract BP like:
    BP: 120/80
    Blood Pressure 120 / 80
    """
    m = re.search(
        r"(?:blood[\s_\-]*pressure|bp)\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        text,
        re.IGNORECASE
    )
    if m:
        return float(m.group(1)), int(m.group(2))
    return None, None


def extract_feature_from_text(text: str, feature: str):
    text = text.lower()

    # BP special case
    if feature == "Sys" or feature == "dia":
        sys_val, dia_val = extract_bp(text)
        if feature == "Sys" and sys_val is not None:
            return sys_val
        if feature == "dia" and dia_val is not None:
            return dia_val

    aliases = FEATURE_ALIASES.get(feature, []) + [feature]

    for alias in aliases:
        pat = alias_to_pattern(alias)

        if feature in BINARY_FEATURES:
            # Example: family history: yes / no
            m = re.search(
                rf"{pat}.{{0,25}}?\b(yes|no|positive|negative|present|absent|true|false|0|1)\b",
                text,
                re.IGNORECASE | re.DOTALL
            )
            if m:
                return cast_value(feature, m.group(1))

        else:
            # Example: HDL: 48.7
            m = re.search(
                rf"{pat}.{{0,20}}?([0-9]+(?:\.[0-9]+)?)",
                text,
                re.IGNORECASE | re.DOTALL
            )
            if m:
                return cast_value(feature, m.group(1))

    return None


def extract_from_text(text: str) -> Dict[str, Any]:
    result = {}
    for feature in gdm_feature_order:
        val = extract_feature_from_text(text, feature)
        if val is not None:
            result[feature] = val
    return result


def read_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if pdfplumber is None:
        raise HTTPException(status_code=500, detail="pdfplumber is not installed")

    extracted = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            extracted.append(page_text)

    text = "\n".join(extracted).strip()

    # OCR fallback if PDF has no extractable text
    if len(text) < 20 and convert_from_bytes and pytesseract:
        images = convert_from_bytes(file_bytes)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        text = "\n".join(ocr_texts)

    return text


def read_text_from_image_bytes(file_bytes: bytes) -> str:
    if Image is None or pytesseract is None:
        raise HTTPException(
            status_code=500,
            detail="Pillow and pytesseract are required for image OCR"
        )

    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)


def extract_features_from_uploaded_file(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    # CSV
    if ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
        return extract_from_dataframe(df)

    # Excel
    if ext in {"xlsx", "xls"}:
        df = pd.read_excel(io.BytesIO(file_bytes))
        return extract_from_dataframe(df)

    # JSON
    if ext == "json":
        obj = json.loads(file_bytes.decode("utf-8"))

        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            df = pd.DataFrame(obj)
            return extract_from_dataframe(df)

        if isinstance(obj, dict):
            normalized_obj = {normalize_key(k): v for k, v in obj.items()}
            result = {}
            for feature in gdm_feature_order:
                aliases = FEATURE_ALIASES.get(feature, []) + [feature]
                for alias in aliases:
                    nk = normalize_key(alias)
                    if nk in normalized_obj:
                        result[feature] = cast_value(feature, normalized_obj[nk])
                        break
            return result

        return {}

    # TXT
    if ext == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        return extract_from_text(text)

    # PDF
    if ext == "pdf":
        text = read_text_from_pdf_bytes(file_bytes)
        return extract_from_text(text)

    # Images
    if ext in {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}:
        text = read_text_from_image_bytes(file_bytes)
        return extract_from_text(text)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: .{ext}"
    )


def validate_and_predict_gdm(features: Dict[str, Any]):
    missing_fields = [f for f in gdm_feature_order if f not in features or features[f] is None]

    if missing_fields:
        return None, None, missing_fields

    input_df = pd.DataFrame([features], columns=gdm_feature_order)
    input_scaled = gdm_scaler.transform(input_df)
    prob = float(gdm_model.predict_proba(input_scaled)[0, 1])
    pred = int(prob >= gdm_threshold)

    label = "GDM Positive" if pred == 1 else "GDM Negative"
    return label, round(prob, 4), []


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Her Journey Medical Prediction API is running",
        "endpoints": {
            "gdm": "/gdm/predict",
            "gdm_file": "/gdm/predict-file",
            "preeclampsia": "/preeclampsia/predict"
        }
    }


@app.post("/gdm/predict", response_model=PredictResponse)
def predict_gdm(data: GDMInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict], columns=gdm_feature_order)
    input_scaled = gdm_scaler.transform(input_df)
    prob = float(gdm_model.predict_proba(input_scaled)[0, 1])
    pred = int(prob >= gdm_threshold)

    return {
        "label": "GDM Positive" if pred == 1 else "GDM Negative",
        "probability": round(prob, 4)
    }


@app.post("/gdm/predict-file", response_model=FilePredictResponse)
async def predict_gdm_file(
    file: UploadFile = File(...),
    extras: Optional[str] = Form(None)
):
    """
    Upload a report file and optionally send missing values in extras as JSON string.

    Example extras:
    {
      "Family_History": 1,
      "PCOS": 0,
      "Sedentary_Lifestyle": 1,
      "Prediabetes": 0,
      "No_of_Pregnancy": 2,
      "Gestation_in_previous_Pregnancy": 1,
      "unexplained_prenetal_loss": 0,
      "Large_Child_or_Birth_Default": 0
    }
    """
    file_bytes = await file.read()

    extracted = extract_features_from_uploaded_file(file.filename, file_bytes)

    # Merge extras if provided
    if extras:
        try:
            extras_dict = json.loads(extras)
            for k, v in extras_dict.items():
                if k in gdm_feature_order:
                    extracted[k] = cast_value(k, v)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extras must be valid JSON string")

    # predict if all available
    label, prob, missing_fields = validate_and_predict_gdm(extracted)

    if missing_fields:
        return {
            "label": None,
            "probability": None,
            "extracted_features": extracted,
            "missing_fields": missing_fields,
            "message": "Some required features are missing. Please provide them in extras."
        }

    return {
        "label": label,
        "probability": prob,
        "extracted_features": extracted,
        "missing_fields": [],
        "message": "Prediction completed successfully."
    }


@app.post("/preeclampsia/predict", response_model=PredictResponse)
def predict_preeclampsia(data: PreeclampsiaInput):
    input_dict = data.model_dump()

    input_df = pd.DataFrame([input_dict], columns=preeclampsia_raw_feature_order)

    input_scaled = preeclampsia_scaler.transform(input_df)

    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=preeclampsia_processed_feature_names
    )

    prob = float(preeclampsia_model.predict_proba(input_scaled_df)[0, 1])
    pred = int(prob >= preeclampsia_threshold)

    return {
        "label": "Preeclampsia Positive" if pred == 1 else "Preeclampsia Negative",
        "probability": round(prob, 8)
    }
