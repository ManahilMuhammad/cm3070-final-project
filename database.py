"""
supabase connection layer
"""

import os
import streamlit as st
from supabase import create_client

# table names in supabase project 
LECTURES = 'lectures'
RESULTS = 'results'

def _credential(name):
    """
    read credential from streamlit secrets with fallback to environment
    """
    try:
        val = st.secrets[name]
    except Exception: # no secrets.toml or key not present
        val = None

    return val or os.environ.get(name)

def get_client():
    """
    return browser session's supabase client
    """
    ss = st.session_state

    if 'supabase' not in ss:
        url = _credential('SUPABASE_URL')
        key = _credential('SUPABASE_ANON_KEY')

        if not url or not key:
            st.error('Supabase not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY to .streamlit/secrets.toml as seen in .streamlit/secrets.toml.example.')
            st.stop()

        ss.supabase = create_client(url, key)

    return ss.supabase
