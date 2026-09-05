from sqlalchemy import Column, String, TIMESTAMP, Boolean
from App.core.database import Base

class PasswordResetToken(Base):
    __tablename__ = "password_reset_token"
    token_id = Column(String, primary_key=True)
    user_id = Column(String)
    token = Column(String, unique=True)
    expires_at = Column(TIMESTAMP)
    used = Column(Boolean, default=False)