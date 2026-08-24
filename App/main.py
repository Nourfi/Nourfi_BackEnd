from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
from App.core.database import get_db
from App.core.security import hash_password, create_access_token
from App.models.user import User
from App.core.security import verify_password 

app = FastAPI()
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr   
    password: str
@app.post("/api/v1/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="This Email is already registered")

    hashed_pw = hash_password(payload.password)
    new_user = User(
        user_id=str(uuid.uuid4()),  
        name=payload.name,
        email=payload.email,
        password=hashed_pw,
        created_at=datetime.utcnow(),
    )
    db.add(new_user)
    db.commit()
    token = create_access_token({"sub": new_user.user_id})

    return {"access_token": token, "token_type": "bearer"}


@app.get("/")
def read_root():
    return {"status": "backend is running"}
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/api/v1/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.user_id})
    return {"access_token": token, "token_type": "bearer"}
