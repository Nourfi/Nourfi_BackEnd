
from xmlrpc.client import Boolean
from sqlalchemy import Column, String, Text, TIMESTAMP, Integer, Boolean
from App.core.database import Base

class User(Base):
    __tablename__ = "User"
    user_id = Column(String, primary_key=True)
    name = Column(Text)
    phone_number = Column(String)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(TIMESTAMP)
    password = Column(String, nullable=False)  
    age = Column(Integer)
    is_verified = Column(Boolean, default=False)