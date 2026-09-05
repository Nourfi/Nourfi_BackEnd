from sqlalchemy import Column, String
from App.core.database import Base

class FinancialProfile(Base):
    __tablename__ = "financial_profile"  #

    financial_id = Column(String, primary_key=True)
    user_id = Column(String)
    family_stat = Column(String)  
    job = Column(String)          
    income = Column(String)
    investments = Column(String)
    postal_code = Column(String)
    credit = Column(String)