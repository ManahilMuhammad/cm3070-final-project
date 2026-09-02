import re
import subprocess
import sys
import streamlit as st

# text-to-speech will run in a child process bc pyttsx3 does not
# allow stopping the speech midway
_WORKER = (
    "import sys, pyttsx3;"
    "sys.stdin.reconfigure(encoding='utf-8');"
    "engine = pyttsx3.init();"
    "engine.say(sys.stdin.read());"
    "engine.runAndWait()" # this blocks stopping speech midway
)

_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0) # keep the child process off-screen on windows

_PROC = 'tts_proc'

def _clean(text):
    """
    strip markdown so speech only includes words not formatting
    """

    text = re.sub(r'\[[ xX]\]', '', str(text)) # remove checkboxes from learning plan
    text = re.sub(r'[#*`_>|]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def stop():
    """
    stop playback immediately
    """

    proc = st.session_state.get(_PROC)

    if proc and proc.poll() is None:
        proc.terminate()

    st.session_state[_PROC] = None

def speak(text):
    """
    read text to speech, replacing anything playing before
    """

    stop()

    text = _clean(text) # clean text

    if not text:
        return

    proc = subprocess.Popen(
        [sys.executable, '-c', _WORKER],
        stdin=subprocess.PIPE,
        encoding='utf-8',
        creationflags=_NO_WINDOW
    )
    proc.stdin.write(text)
    proc.stdin.close()

    st.session_state[_PROC] = proc

def controls(text, key, label='Read out loud'):
    """
    draw a play and stop button
    """

    play, halt, _ = st.columns([1, 1, 4])

    if play.button(f'{label}', key=f'tts_play_{key}'):
        speak(text)

    if halt.button('Stop', key=f'tts_stop_{key}'):
        stop()
        