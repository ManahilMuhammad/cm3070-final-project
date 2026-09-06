from database import get_client, LECTURES, RESULTS
from datetime import datetime
import json

def load_json(val):
    """
    decode JSON column value into python data
    unwrap until string stops being returned 
    (for old sqlite rows that may still exist)
    """
    for _ in range(3):
        if not isinstance(val, str):
            break
        try:
            val = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            break
    return val

def fmt_date(ts):
    """
    format supabase timestamp for display
    """
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    return ts.strftime('%b %d, %Y')

def save_lecture(user_id, title, files, combined_text, summary):
    """
    save processed lecture
    """
    sb = get_client()

    res = sb.table(LECTURES).insert({
        'user_id': user_id,
        'title': title,
        'uploaded_files': [f.name for f in files],
        'combined_text': combined_text,
        'summary': summary
    }).execute()

    return res.data[0]['lecture_id']

def save_result(user_id, lecture_id, quiz, results, performance, feedback, plan):
    """
    save quiz results and feedback
    """
    sb = get_client()

    res = sb.table(RESULTS).insert({
        'user_id': user_id,
        'lecture_id': lecture_id,
        'quiz_questions': quiz,
        'user_answers': results,
        'performance': performance,
        'feedback': feedback,
        'study_plan': plan
    }).execute()

    return res.data[0]['result_id']

def update_study_plan(result_id, plan):
    """
    attach learning plan to already saved results
    """
    sb = get_client()
    sb.table(RESULTS).update({'study_plan': plan}).eq('result_id', result_id).execute()

def get_lectures(user_id):
    """
    get all lectures for user with id in parameter
    """
    sb = get_client()

    res = (
        sb.table(LECTURES)
        .select('lecture_id, title, summary, created_at')
        .eq('user_id', user_id)
        .order('created_at', desc=True)
        .execute()
    )

    return res.data

def get_lecture(lecture_id):
    """
    get lecture by id (None if id doesnt exist)
    """
    sb = get_client()

    res = sb.table(LECTURES).select('*').eq('lecture_id', lecture_id).limit(1).execute()

    return res.data[0] if res.data else None

def get_results(lecture_id):
    """
    get all quiz results for user with id in parameter
    """
    sb = get_client()

    res = (
        sb.table(RESULTS)
        .select('*')
        .eq('lecture_id', lecture_id)
        .order('completed_at')
        .execute()
    )

    return res.data 
