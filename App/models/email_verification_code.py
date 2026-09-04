from sqlalchemy import Column, String, TIMESTAMP, Boolean
from App.core.database import Base
class EmailVerificationCode(Base):
    __tablename__ = "email_verification_code"
    code_id = Column(String, primary_key=True)
    user_id = Column(String)
    code = Column(String)
    expires_at = Column(TIMESTAMP)
    used = Column(Boolean, default=False)