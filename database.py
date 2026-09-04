from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()
engine = create_engine('sqlite:///sprout.db')

class User(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.now)

class Lecture(Base):
    __tablename__ = 'results'
    lecture_id = Column(String, primary_key=True)
    user_id = Column(String)
    title = Column(String)
    uploaded_files = Column(JSON)
    combined_text = Column(Text)
    summary = Column(String)
    created_at = Column(DateTime, default=datetime.now)

class QuizResult(Base):
    __tablename__ = 'quiz_results'
    result_id = Column(String, primary_key=True)
    lecture_id = Column(String)
    user_id = Column(String)
    quiz_questions = Column(JSON)
    user_answers = Column(JSON)
    performance = Column(JSON)
    feedback = Column(Text)
    study_plan = Column(JSON)
    completed_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
