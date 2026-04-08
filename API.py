'''
python -m uvicorn API:app --reload 
cd "C:\c++ project\ICPC\grad prog"



from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "A7AAAAA World"}

@app.post("/")
async def post():
    return {"message": "A7AAAAA World"}



@app.put("/",description="this is a put endpoint")
async def put():
    return{"message":"this is a put request"}

@app.get("/users")
async def list_users():
    return{"message":"this is users list"}

@app.get("/users/1",include_in_schema=False)
async def admin_user():
    return{"message":"this is the admin portal"}

@app.get("/users/{user_id}")
async def get_user(user_id : int ):
    return{"user id":user_id}
'''
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

artifact = joblib.load("gdm_artifact.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
threshold = artifact["threshold"]
feature_order = artifact["feature_order"]
app = FastAPI()


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


class Response(BaseModel):
    label: str
    probability: float


@app.get("/")
def root():
    return {"message": "GDM API is running"}

@app.post("/predict", response_model=Response)
def predict(data: GDMInput):
    input_dict = data.model_dump() 
    input_df = pd.DataFrame([input_dict], columns=feature_order)
    input_scaled = scaler.transform(input_df)
    prob = float(model.predict_proba(input_scaled)[0, 1])
    pred = int(prob >= threshold)
    return {
        "label": "GDM Positive" if pred == 1 else "GDM Negative",
        "probability": round(prob, 4)
    }