from sqlalchemy import Column, String, Text, TIMESTAMP
from App.core.database import Base

class Message(Base):
    __tablename__ = "message"
    msg_id = Column(String, primary_key=True)
    conv_id = Column(String)  
    content = Column(Text)
    sender = Column(String)   
    timestamp = Column(TIMESTAMP)