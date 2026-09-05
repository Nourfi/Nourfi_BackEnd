from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime
from App.core.database import get_db
from App.core.security import hash_password, create_access_token
from App.models.user import User
from App.core.security import verify_password 
from App.core.encryption import encrypt_value, decrypt_value
from App.models.financial_profile import FinancialProfile
from App.models.conversation import Conversation
from App.models.message import Message
from fastapi.middleware.cors import CORSMiddleware
import uuid
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

class FinancialProfileRequest(BaseModel):
    user_id: str
    family_stat: str | None = None
    job: str | None = None
    income: float | None = None
    investments: float | None = None
    postal_code: str | None = None
    credit: float | None = None
@app.post("/api/v1/profile/financial")
def create_financial_profile(payload: FinancialProfileRequest, db: Session = Depends(get_db)):
    new_profile = FinancialProfile(
        financial_id=str(uuid.uuid4()),
        user_id=payload.user_id,
        family_stat=payload.family_stat, 
        job=payload.job,                   
        income=encrypt_value(payload.income),           
        investments=encrypt_value(payload.investments), 
        postal_code=encrypt_value(payload.postal_code), 
        credit=encrypt_value(payload.credit),            
    )
    db.add(new_profile)
    db.commit()
    return {"status": "profile saved", "financial_id": new_profile.financial_id}

@app.get("/api/v1/profile/financial/{financial_id}")
def get_financial_profile(financial_id: str, db: Session = Depends(get_db)):
    profile = db.query(FinancialProfile).filter(FinancialProfile.financial_id == financial_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "financial_id": profile.financial_id,
        "user_id": profile.user_id,
        "family_stat": profile.family_stat,
        "job": profile.job,
        "income": decrypt_value(profile.income),
        "investments": decrypt_value(profile.investments),
        "postal_code": decrypt_value(profile.postal_code),
        "credit": decrypt_value(profile.credit),
    }

class CreateConversationRequest(BaseModel):
    user_id: str
    title: str
@app.post("/api/v1/chat/conversations")
def create_conversation(payload: CreateConversationRequest, db: Session = Depends(get_db)):
    new_conv = Conversation(
        conv_id=str(uuid.uuid4()),
        user_id=payload.user_id,
        title=payload.title,
        created_at=datetime.utcnow(),
    )
    db.add(new_conv)
    db.commit()
    return {"conv_id": new_conv.conv_id, "title": new_conv.title}
class SendMessageRequest(BaseModel):
    conv_id: str
    content: str
    sender: str  
@app.post("/api/v1/chat/messages")
def send_message(payload: SendMessageRequest, db: Session = Depends(get_db)):
    new_msg = Message(
        msg_id=str(uuid.uuid4()),
        conv_id=payload.conv_id,
        content=payload.content,
        sender=payload.sender,
        timestamp=datetime.utcnow(),
    )
    db.add(new_msg)
    db.commit()
    return {"msg_id": new_msg.msg_id, "content": new_msg.content, "sender": new_msg.sender}