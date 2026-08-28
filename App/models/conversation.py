from sqlalchemy import Column, String, TIMESTAMP
from App.core.database import Base
class Conversation(Base):
    __tablename__ = "conversation"
    conv_id = Column(String, primary_key=True)
    user_id = Column(String)  
    title = Column(String)
    created_at = Column(TIMESTAMP)