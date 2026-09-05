from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import secrets
import uuid
import random
from App.core.database import get_db
from App.core.security import hash_password, verify_password, create_access_token
from App.core.encryption import encrypt_value, decrypt_value
from App.core.email import send_reset_password_email, send_verification_email
from App.models.user import User
from App.models.financial_profile import FinancialProfile
from App.models.conversation import Conversation
from App.models.message import Message
from App.models.password_reset_token import PasswordResetToken
from App.models.email_verification_code import EmailVerificationCode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"status": "backend is running"}

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str

class ResendCodeRequest(BaseModel):
    email: EmailStr
def generate_and_send_code(user, db):
    code = str(random.randint(100000, 999999))
    code_row = EmailVerificationCode(
        code_id=str(uuid.uuid4()),
        user_id=user.user_id,
        code=code,
        expires_at=datetime.utcnow() + timedelta(minutes=15),
        used=False,
    )
    db.add(code_row)
    db.commit()
    send_verification_email(user.email, code)
@app.post("/api/v1/auth/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = hash_password(payload.password)
    new_user = User(
        user_id=str(uuid.uuid4()),
        name=payload.name,
        email=payload.email,
        password=hashed_pw,
        is_verified=False,
        created_at=datetime.utcnow(),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    generate_and_send_code(new_user, db)
    token = create_access_token({"sub": new_user.user_id})
    return {"access_token": token, "token_type": "bearer"}
@app.post("/api/v1/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Veuillez vérifier votre adresse e-mail avant de vous connecter.")
    token = create_access_token({"sub": user.user_id})
    return {"access_token": token, "token_type": "bearer"}
@app.post("/api/v1/auth/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    code_row = db.query(EmailVerificationCode).filter(
        EmailVerificationCode.user_id == user.user_id,
        EmailVerificationCode.code == payload.code,
        EmailVerificationCode.used == False,
    ).first()
    if not code_row or code_row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Code invalide ou expiré.")
    user.is_verified = True
    code_row.used = True
    db.commit()
    return {"message": "E-mail vérifié avec succès."}
@app.post("/api/v1/auth/resend-code")
def resend_code(payload: ResendCodeRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user and not user.is_verified:
        generate_and_send_code(user, db)
    return {"message": "Si votre compte existe et n'est pas vérifié, un nouveau code a été envoyé."}

class ForgotPasswordRequest(BaseModel):
    email: EmailStr
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
@app.post("/api/v1/auth/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        return {"message": "un lien de réinitialisation a été envoyé."}
    reset_token = secrets.token_urlsafe(32)
    token_row = PasswordResetToken(
        token_id=str(uuid.uuid4()),
        user_id=user.user_id,
        token=reset_token,
        expires_at=datetime.utcnow() + timedelta(minutes=30),
        used=False,
    )
    db.add(token_row)
    db.commit()
    send_reset_password_email(user.email, reset_token)
    return {"message": "un lien de réinitialisation a été envoyé."}
@app.post("/api/v1/auth/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_row = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == payload.token
    ).first()
    if not token_row or token_row.used or token_row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Lien invalide ou expiré.")
    user = db.query(User).filter(User.user_id == token_row.user_id).first()
    user.password = hash_password(payload.new_password)
    token_row.used = True
    db.commit()
    return {"message": "Mot de passe réinitialisé avec succès."}
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
@app.get("/api/v1/profile/{user_id}")
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    financial = db.query(FinancialProfile).filter(FinancialProfile.user_id == user_id).first()
    return {
        "name": user.name,
        "email": user.email,
        "age": user.age,
        "family_stat": financial.family_stat if financial else None,
        "job": financial.job if financial else None,
        "income": decrypt_value(financial.income) if financial and financial.income else None,
        "has_down_payment": financial.has_down_payment if financial else None,
        "down_payment": decrypt_value(financial.down_payment) if financial and financial.down_payment else None,
        "has_loans": financial.has_loans if financial else None,
        "credit": decrypt_value(financial.credit) if financial and financial.credit else None,
        "maingoal": financial.maingoal if financial else None,
        "knowledgelvl": financial.knowledgelvl if financial else None,
        "ethical_priorities": financial.ethical_priorities.split(",") if financial and financial.ethical_priorities else [],
        "networth": decrypt_value(financial.networth) if financial and financial.networth else None,
    }
class ProfileUpdateRequest(BaseModel):
    age: int | None = None
    family_stat: str | None = None
    job: str | None = None
    income: float | None = None
    has_down_payment: bool | None = None
    down_payment: float | None = None
    has_loans: bool | None = None
    credit: float | None = None
    maingoal: str | None = None
    knowledgelvl: str | None = None
    ethical_priorities: list[str] = []
    networth: float | None = None
@app.put("/api/v1/profile/{user_id}")
def update_profile(user_id: str, payload: ProfileUpdateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.age is not None:
        user.age = payload.age
    financial = db.query(FinancialProfile).filter(FinancialProfile.user_id == user_id).first()
    if not financial:
        financial = FinancialProfile(financial_id=str(uuid.uuid4()), user_id=user_id)
        db.add(financial)
    financial.family_stat = payload.family_stat
    financial.job = payload.job
    financial.income = encrypt_value(payload.income)
    financial.has_down_payment = payload.has_down_payment
    financial.down_payment = encrypt_value(payload.down_payment)
    financial.has_loans = payload.has_loans
    financial.credit = encrypt_value(payload.credit)
    financial.maingoal = payload.maingoal
    financial.knowledgelvl = payload.knowledgelvl
    financial.ethical_priorities = ",".join(payload.ethical_priorities)
    financial.networth = encrypt_value(payload.networth)
    db.commit()
    return {"status": "profile updated"}
class OnboardingSubmitRequest(BaseModel):
    user_id: str
    age: int | None = None
    situationFamiliale: str | None = None
    profession: str | None = None
    revenuMensuel: float | None = None
    apportInitial: str | None = None
    montantApport: float | None = None
    montantCredit: float | None = None
    financementsEnCours: str | None = None
    objectifPrincipal: str | None = None
    niveauConnaissance: str | None = None
    prioritesEthiques: list[str] = []
    patrimoineEstime: float | None = None
@app.post("/api/v1/consultation/submit")
def submit_onboarding(payload: OnboardingSubmitRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.age = payload.age
    new_profile = FinancialProfile(
    financial_id=str(uuid.uuid4()),
    user_id=payload.user_id,
    family_stat=payload.situationFamiliale,
    job=payload.profession,
    income=encrypt_value(payload.revenuMensuel),
     has_down_payment=(payload.apportInitial == "Oui"),
     down_payment=encrypt_value(payload.montantApport),
     has_loans=(payload.financementsEnCours == "Oui"),
    credit=encrypt_value(payload.montantCredit),
    maingoal=payload.objectifPrincipal,
    knowledgelvl=payload.niveauConnaissance,
    ethical_priorities=",".join(payload.prioritesEthiques),
    networth=encrypt_value(payload.patrimoineEstime),
    )
    db.add(new_profile)
    db.commit()
    return {"status": "profile saved", "financial_id": new_profile.financial_id}
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