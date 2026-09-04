from database import Session, Lecture, QuizResult
import uuid
import json

def save_lecture(user_id, title, files, combined_text, summary):
    """
    save processed lecture
    """
    session = Session()

    lecture_id = str(uuid.uuid4())
    lecture = Lecture(
        lecture_id=lecture_id,
        user_id=user_id,
        title=title,
        uploaded_files=[f.name for f in files],
        combined_text=combined_text,
        summary=summary
    )
    session.add(lecture)
    session.commit()
    session.close()

    return lecture_id

def save_quiz_result(user_id, lecture_id, quiz, results, performance, feedback, plan):
    """
    save quiz results and feedback
    """
    session = Session()

    result_id = str(uuid.uuid4())
    quiz_result = QuizResult(
        result_id=result_id,
        lecture_id=lecture_id,
        user_id=user_id,
        quiz_questions=json.dumps(quiz),
        user_answers=json.dumps(results),
        performance=json.dumps(performance),
        feedback=feedback,
        study_plan=json.dumps(plan)
    )
    session.add(quiz_result)
    session.commit()
    session.close()

    return result_id

def get_user_lectures(user_id):
    """"
    get all lectures for user with id in parameter
    """
    session = Session()
    lectures = session.query(Lecture).filter_by(user_id=user_id).order_by(Lecture.created_at.desc()).all()
    session.close()
    return lectures

def get_quiz_results(lecture_id):
    """
    get all quiz results for user with id in parameter
    """
    session = Session()
    results = session.query(QuizResult).filter_by(lecture_id=lecture_id).all()
    session.close()
    return results
