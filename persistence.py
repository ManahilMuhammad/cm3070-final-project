from database import Session, Lecture, QuizResult
import uuid
import json

def load_json(val):
    """
    decode JSON colun value into python data
    unwrap until string stops being returned
    """
    for _ in range(3):
        if not isinstance(val, str):
            break
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            break
    return val

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
        quiz_questions=quiz,
        user_answers=results,
        performance=performance,
        feedback=feedback,
        study_plan=plan
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
