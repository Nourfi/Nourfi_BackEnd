from sqlalchemy import Column, String, Boolean
from App.core.database import Base

class FinancialProfile(Base):
    __tablename__ = "financial_profile"

    financial_id = Column(String, primary_key=True)
    user_id = Column(String)
    family_stat = Column(String)
    income = Column(String)
    investments = Column(String)
    job = Column(String)
    postal_code = Column(String)
    credit = Column(String)
    down_payment = Column(String)
    has_down_payment = Column(Boolean)
    has_loans = Column(Boolean)
    maingoal = Column(String)
    knowledgelvl = Column(String)
    ethical_priorities = Column(String)
    networth = Column(String)