from database import User, Session
import uuid

def create_user(email, name):
    """"
    create new user if doesn't exist
    """
    session = Session()

    existing = session.query(User).filter_by(email=email).first()
    if existing:
        session.close()
        return existing.user_id

    user_id = str(uuid.uuid4())
    user = User(user_id=user_id, email=email, name=name)
    session.add(user)
    session.commit()
    session.close()

    return user_id

def get_user(user_id):
    """
    fetch user by their id
    """
    session = Session()
    user = session.query(User).filter_by(user_id=user_id).first()
    session.close()
    return user