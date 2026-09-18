from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os
import random
import secrets
import uuid

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

# Core & App Imports
from App.core.database import get_db
from App.core.email import send_reset_password_email, send_verification_email
from App.core.encryption import decrypt_value, encrypt_value
from App.core.security import create_access_token, hash_password, verify_password

# Models
from App.models.conversation import Conversation
from App.models.email_verification_code import EmailVerificationCode
from App.models.financial_profile import FinancialProfile
from App.models.message import Message
from App.models.password_reset_token import PasswordResetToken
from App.models.user import User

# AI RAG Imports
from App.AI_RAG.rag.pipeline import load_encoder, answer_question, summarize_conversation, DISCLAIMER

# ------------------------------------------------------------------------------
# 1. LIFESPAN & GLOBAL AI MODEL STORAGE
# ------------------------------------------------------------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading AI models into memory...")
    load_encoder()  # chauffe le cache, résultat ignoré ici volontairement
    yield
# ------------------------------------------------------------------------------
# 2. APP INITIALIZATION & CORS
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

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


# ------------------------------------------------------------------------------
# 3. PYDANTIC SCHEMAS (Defined before route references)
# ------------------------------------------------------------------------------
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

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class FinancialProfileRequest(BaseModel):
    user_id: str
    family_stat: str | None = None
    job: str | None = None
    income: float | None = None
    investments: float | None = None
    postal_code: str | None = None
    credit: float | None = None

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

class CreateConversationRequest(BaseModel):
    user_id: str
    title: str

class SendMessageRequest(BaseModel):
    conv_id: str
    content: str
    sender: str


# ------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 5. AUTH ENDPOINTS
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 6. PROFILE ENDPOINTS
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 7. CHAT & RAG ENDPOINTS
# ------------------------------------------------------------------------------
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


"""
Remplace TOUT le bloc, depuis le premier "SUMMARY_FREQUENCY = 10" jusqu'à
la fin de send_message(), par ce qui suit. Corrige 2 choses :
1. AJOUTE RAG_DATABASE_URL (variable manquante — cause un NameError actuellement)
2. SUPPRIME les définitions dupliquées (SUMMARY_FREQUENCY, RECENT_MESSAGES_LIMIT,
   detect_persona, get_recent_messages n'apparaissaient chacun qu'une fois avant,
   maintenant deux fois dans ton fichier)
"""

import time

# NOUVEAU — manquait, cause un NameError dans send_message() sans ça
RAG_DATABASE_URL = (
    os.getenv("SUPABASE_DATABASE_URL")
    or os.getenv("SUPABASE_DB_URL")
    or os.getenv("DATABASE_URL")
)

SUMMARY_FREQUENCY = 10
RECENT_MESSAGES_LIMIT = 4
MAX_MESSAGE_CHARS = 600
MIN_SECONDS_BETWEEN_GROQ_CALLS = 2.0

_last_groq_call_time = 0.0


def throttle_groq_calls():
    """Espace les appels Groq d'au moins MIN_SECONDS_BETWEEN_GROQ_CALLS."""
    global _last_groq_call_time
    elapsed = time.monotonic() - _last_groq_call_time
    if elapsed < MIN_SECONDS_BETWEEN_GROQ_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_GROQ_CALLS - elapsed)
    _last_groq_call_time = time.monotonic()


def detect_persona(financial_profile) -> str:
    """Mapping basé sur la profession déclarée à l'inscription (Step1.jsx)."""
    if not financial_profile or not financial_profile.job:
        return "general_public"
    profession = financial_profile.job.strip()
    mapping = {
        "Entrepreneur": "entrepreneur",
        "Étudiant": "student",
        "Spécialiste finance": "finance_specialist",
        "Banquier": "banker",
        "Chef d'entreprise": "business_owner",
        "Investisseur": "investor",
        "Universitaire": "academic",
        # "Ouvrier" et "Autre" -> general_public
    }
    return mapping.get(profession, "general_public")


def get_recent_messages(history: list, limit: int = 10, max_chars: int = MAX_MESSAGE_CHARS) -> list:
    """Tronque chaque message à max_chars avant de le renvoyer."""
    trimmed = history[-limit:]
    return [{**m, "content": m["content"][:max_chars]} for m in trimmed]


@app.post("/api/v1/chat/messages")
def send_message(payload: SendMessageRequest, db: Session = Depends(get_db)):
    user_msg = Message(
        msg_id=str(uuid.uuid4()),
        conv_id=payload.conv_id,
        content=payload.content,
        sender="user",
        timestamp=datetime.utcnow(),
    )
    db.add(user_msg)
    db.commit()

    conversation = db.query(Conversation).filter(Conversation.conv_id == payload.conv_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    financial = (
        db.query(FinancialProfile)
        .filter(FinancialProfile.user_id == conversation.user_id)
        .first()
    )
    persona_key = detect_persona(financial)

    prior_messages = (
        db.query(Message)
        .filter(Message.conv_id == payload.conv_id, Message.msg_id != user_msg.msg_id)
        .order_by(Message.timestamp)
        .all()
    )
    recent = get_recent_messages(
        [{"role": m.sender, "content": m.content} for m in prior_messages],
        RECENT_MESSAGES_LIMIT,
    )

    throttle_groq_calls()
    try:
        answer, results = answer_question(
            question=payload.content,
            database_url=RAG_DATABASE_URL,
            persona_name=persona_key,
            conversation_summary=conversation.summary,
            recent_messages=recent,
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        if "429" in error_text or "rate_limit" in error_text or "rate limit" in error_text:
            answer = (
                "Le service est momentanément très sollicité. "
                "Merci de réessayer dans quelques secondes."
            )
        else:
            answer = "Une erreur est survenue lors de la génération de la réponse. Réessayez."
        results = []

    sources = [
        {
            "standard_number": item.get("standard_number"),
            "title": item.get("standard_title"),
            "pages": f"{item.get('page_start')}-{item.get('page_end')}",
        }
        for item in results
    ]

    assistant_msg = Message(
        msg_id=str(uuid.uuid4()),
        conv_id=payload.conv_id,
        content=answer,
        sender="assistant",
        timestamp=datetime.utcnow(),
    )
    db.add(assistant_msg)
    db.commit()

    all_messages = (
        db.query(Message)
        .filter(Message.conv_id == payload.conv_id)
        .order_by(Message.timestamp)
        .all()
    )
    if len(all_messages) > 0 and len(all_messages) % SUMMARY_FREQUENCY == 0:
        recent_for_summary = get_recent_messages(
            [{"role": m.sender, "content": m.content} for m in all_messages], 10
        )
        try:
            throttle_groq_calls()
            conversation.summary = summarize_conversation(conversation.summary, recent_for_summary)
            db.commit()
        except RuntimeError:
            pass

    return {
        "msg_id": assistant_msg.msg_id,
        "content": assistant_msg.content,
        "sender": assistant_msg.sender,
        "disclaimer": DISCLAIMER,
        "sources": sources,
    }