"""
authentication using supabase
"""

from database import get_client

def sign_up(email, password, name):
    """
    register new account
    """
    sb = get_client()
    res = sb.auth.sign_up({
        'email': email,
        'password': password,
        'options': {'data': {'name': name}}
    })

    return res.user, res.session is None

def sign_in(email, password):
    """
    sign in existing user
    """
    sb = get_client()
    res = sb.auth.sign_in_with_password({'email': email, 'password': password})
    return res.user

def sign_out():
    """
    end current session
    """
    sb = get_client()
    try:
        sb.auth.sign_out()
    except Exception:
        pass # token expired or revoked so nothing left to end

def get_user():
    """
    fetch signed-in user or None if session gone
    """
    sb = get_client()
    try:
        res = sb.auth.get_user()
    except Exception:
        return None
    return res.user if res else None

def display_name(user):
    """
    name given at sign up with fallback to local part of email
    """
    meta = user.user_metadata or {}
    return meta.get('name') or user.email.split('@')[0]
