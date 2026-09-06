import streamlit as st
import time
from pipeline import (
    transcribe_audio, extract_slides, ocr_notes, describe_figure, # ingestion 
    unload_all, release_llm, # model-related
    fuse, # utilty
    make_summary, # summary
    make_quiz, # quiz
    generate_score, generate_feedback, # feedback
    create_plan, render_study_plan, # learning plan
    extract_from_file, # extraction
    VL_MODEL
)
import tempfile
import cv2
import os
import tts
from auth import sign_in, sign_up, sign_out, display_name
from persistence import (
    save_lecture, save_result, update_study_plan,
    get_lectures, get_lecture, get_results,
    load_json, fmt_date
)
import json

st.title('Sprout')

ss = st.session_state
if 'user_id' not in ss:
    ss.user_id = None
if 'stage' not in ss:
    ss.stage = 'login'
    ss.q_index = 0
    ss.results = []

# LOGIN SCREEN
if ss.user_id is None:
    st.subheader('Sign in to your account')

    login_tab, register_tab = st.tabs(['Sign In', 'Create Account'])

    with login_tab:
        email = st.text_input('Email', key='login-email')
        password = st.text_input('Password', type='password', key='login-password')

        if st.button('Sign In'):
            if not email or not password:
                st.warning('Please enter both email and password.')
            else:
                try:
                    user = sign_in(email, password)
                    ss.user_id = user.id
                    ss.username = display_name(user)
                    ss.stage = 'home'
                    st.rerun()
                except Exception as e:
                    st.error(f'Sign in failed: {e}')

    with register_tab:
        email = st.text_input('Email', key='register-email')
        name = st.text_input('Name', key='register-name')
        password = st.text_input(
            'Password', type='password', key='register-password',
            help='Your password must be at least 6 characters long.'
        )

        if st.button('Create Account'):
            if not email or not password:
                st.warning('You must enter both an email and a password.')
            else:
                try:
                    user, confirm_needed = sign_up(
                        email, password, name or email.split('@')[0]
                    )

                    # UNDECIDED -- if supabase configured to confirm email before continuing
                    if confirm_needed:
                        st.success('Your account has successfully been created! Check your inbox to verify your email address then sign in.')
                    else:
                        ss.user_id = user.id
                        ss.username = display_name(user)
                        ss.stage = 'home'
                        st.rerun()
                except Exception as e:
                    st.error(f'Could not create account! {e}')

    st.stop()

# SIDEBAR
with st.sidebar:
    st.write(f'{ss.username}')

    if st.button('Home'):
        ss.stage = 'home'
        st.rerun()
    if st.button('New Session'):
        ss.stage = 'upload'
        # forget previous attempt
        ss.pop('result_id', None)
        ss.pop('plan_saved', None)
        st.rerun()
    if st.button('Sign Out'):
        sign_out()
        ss.user_id = None
        ss.stage = 'login'
        st.rerun()

# HOME SCREEN
if ss.stage == 'home':
    st.header('Your Sessions')

    sessions = get_lectures(ss.user_id)

    if not sessions:
        st.info('No sessions yet. Start with uploading your study material!')
        if st.button('Upload Materials'):
            ss.stage = 'upload'
            st.rerun()

    else:
        for session in sessions:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f'**{session['title']}**')
                st.caption(f"Uploaded: {fmt_date(session['created_at'])}")

            with col2:
                if st.button('View', key=f'view-{session['lecture_id']}'):
                    ss.selected_lecture_id = session['lecture_id']
                    ss.stage = 'detail'
                    st.rerun()

            with col3:
                if st.button('Delete', key=f"del-{session['lecture_id']}"):
                    continue # TO IMPLEMENT: delete functionality

# SESSION DETAIL SCREEN
if ss.stage == 'detail':
    session_id = ss.selected_lecture_id
    lecture = get_lecture(session_id)

    if lecture is None:
        st.error('Lecture not found!')
        st.stop()

    st.header(lecture['title'])
    st.subheader('Summary')
    st.write(lecture['summary'])

    st.subheader('Past Results')
    results = get_results(session_id)

    if not results:
        st.info('No quiz results available for this lecture.')
    else:
        for i, result in enumerate(results, 1):
            with st.expander(f"Attempt {i} - {fmt_date(result['completed_at'])}"):
                perf = load_json(result['performance']) or []
                for entry in perf:
                    st.write(f"- **{entry['topic']}**: {entry['score']} (confidence: {entry['confidence']:.1%})")

                st.write('**Feedback:**')
                st.write(result['feedback'])

                st.write('**Study plan:**')
                plan = load_json(result['study_plan']) or []
                if not plan:
                    st.caption('No learning plan was generated for this session.')
                for item in plan:
                    st.write(f"- Day {item['day']}: {item['action']}")

# UPLOAD SCREEN
if ss.stage == 'upload':
    st.subheader("Upload your lecture material")
    st.write("Upload one or multiple files (video, audio, slides, images)")

    lecture_title = st.text_input(
        'Give a title to this session: ',
        value='Untitled',
        help="e.g. 'homeostasis lecture 1'"
    )
    
    files = st.file_uploader(
        'Upload files',
        type=['mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'm4a', 'aac', 'flac',
              'pdf', 'pptx', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True  # allow multiple files
    )

    if st.button('Process'):
        if not files:
            st.warning('Please upload at least one file.')
        else:
            try:
                with st.status('Processing your materials...') as status:
                    
                    all_transcripts = []
                    all_slide_text = []
                    all_notes_text = []
                    all_figures = []

                    for file in files:
                        st.info(f"Processing {file.name}...")
                        
                        extracted = extract_from_file(file)

                        # process extracted audio
                        if extracted['audio']:
                            status.update(label='Transcribing audio...') # checkpoint 1: transcription
                            transcript = transcribe_audio(extracted['audio'])
                            all_transcripts.append(transcript)
                            unload_all()
                        
                        # use extracted text directly
                        if extracted['text']:
                            status.update(label='Extracting slide text...') # checkpoint 2: slide text extraction
                            all_slide_text.append(extracted['text'])
                        
                        # describe extracted images (figures/diagrams from PDF/PPTX)
                        if extracted['images']:
                            status.update(label='Describing figures...') # checkpoint 3: figure description
                            for img in extracted['images']:
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    img.save(tmp.name)
                                    description = describe_figure(tmp.name)
                                    all_figures.append(description)
                                os.remove(tmp.name)
                            release_llm(model=VL_MODEL, end_of_phase=False)

                        # apply OCR to extracted slide frames
                        if extracted['notes']:
                            status.update(label='Reading notes...') # checkpoint 4: OCR
                            for note in extracted['notes']:
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    tmp_path = tmp.name
                                    
                                try:
                                    note.convert("RGB").save(tmp_path, format="PNG")
                                    notes_text = ocr_notes(tmp_path)
                                    all_notes_text.append(notes_text)

                                finally:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                    
                            unload_all()

                    status.update(label='Combining extracted content...') # checkpoint 5: combining extracted text
                    ss.combined = fuse(
                        " ".join(all_transcripts),
                        " ".join(all_slide_text),
                        " ".join(all_notes_text),
                        " ".join(all_figures)
                    )
                    
                    if not ss.combined.strip():
                        st.warning('No content extracted. Check your files.')
                        st.stop()
                    
                    status.update(label='Generating summary...') # checkpoint 6: creating summary
                    ss.summary = make_summary(ss.combined[:12000])
        
                    status.update(label='Generating quiz...') # checkpoint 7: creating quiz
                    ss.quiz = make_quiz(ss.summary[:6000]) or []

                status.update(label='Processing complete!', state='complete') # checkpoint 8: complete
                    
                release_llm(unload=False) # keep llama3.2 for scoring/feedback/plan

                # save lecture
                ss.lecture_id = save_lecture(
                    ss.user_id,
                    lecture_title,
                    files,
                    ss.combined,
                    ss.summary
                )

                ss.stage = 'summary'
                st.rerun()

            except Exception as e:
                st.error(f'Processing failed: {type(e).__name__}: {e}')
                st.exception(e)

# SUMMARY SCREEN
elif ss.stage == 'summary':
    st.header('Summary')
    st.markdown(ss.summary)

    tts.controls(ss.summary, key='summary') # text-to-speech options

    if st.button('Start quiz'):
        tts.stop() # stop text-to-speech
        ss.stage = 'quiz'
        ss.q_start = time.time() 
        st.rerun()

# QUIZ SCREEN
elif ss.stage == 'quiz':
    q = ss.quiz[ss.q_index]
    st.header(f'Question {ss.q_index + 1} of {len(ss.quiz)}')
    st.write(q['question'])

    # true-false
    if q['type'] == 'true-false':
        answer = st.radio('Choose: ', ['True', 'False'], key=f"q{ss.q_index}")

    # MCQ
    elif q.get('options'):
        answer = st.radio('Choose:', q['options'], key=f'q{ss.q_index}')

    # fill-in-the-blank / short-answer
    else:
        answer = st.text_input('Your answer:', key=f'q{ss.q_index}')

    # read out questions and answers to choose from
    spoken = q['question'].replace('_____', 'blank') # replace the spaces with the word 'blank' in text-to-speech
    if q['type'] == 'true-false':
        spoken += '. Is this true or false?' # read out true/false options
    elif q.get('options'):
        spoken += '. Your options are: ' + ', '.join(f'{i}. {o}' for i, o in enumerate(q['options'], 1)) # read out mcq options

    @st.fragment
    def tts_section():
        tts.controls(spoken, key=f'q{ss.q_index}', label='Read question') # text-to-speech

    tts_section() # calling here so clicking button does not reset whole app

    if st.button('Submit'):
        tts.stop() # stop text-to-speech
        elapsed = time.time() - ss.q_start

        ss.results.append({
            'question': q['question'], 'type': q['type'], 'topic': q['topic'],
            'correct_answer': q['answer'], 'user_answer': answer,
            'time': elapsed, 'num_words': len(q['question'].split()),
        })

        ss.q_index += 1

        # if not last question then record time and show next one
        if ss.q_index < len(ss.quiz):
            ss.q_start = time.time()          
            st.rerun()

        # if last question then show result
        else:
            ss.stage = 'results'
            st.rerun()

# RESULTS SCREEN
elif ss.stage == 'results':
    st.header('Your Results')

    # initialise if missing
    if 'performance' not in ss:
        ss.performance = None
    if 'feedback' not in ss:
        ss.feedback = None
    if 'plan' not in ss:
        ss.plan = None

    if ss.performance is None:
        with st.status('Preparing your results...', expanded=True) as status:
            status.update(label='Scoring your answers...') # checkpoint 1: scoring
            ss.performance = generate_score(ss.results)
            release_llm(unload=True, end_of_phase=False) # free llama but dont save run

            status.update(label='Generating feedback...') # checkpoint 2: feedback generation
            ss.feedback = generate_feedback(ss.performance, ss.summary)
            release_llm(unload=True, end_of_phase=False) # free llama but dont save run

            status.update(label='Done!', state='complete') # checkpoint 3: coplete

        release_llm(unload=False) # keep llama3.2 warm for the learning plan

    # FEEDBACK
    st.subheader('Feedback')
    if ss.feedback:
        st.write(ss.feedback)
        tts.controls(ss.feedback, key='feedback') # text-to-speech
    else:
        st.info('Generating feedback...')

    # LEARNING PLAN
    st.subheader('Your Personalised Learning Plan')

    # create plan if not already done
    if ss.plan == None:

        # ask student how long they want the plan to be
        duration = st.slider(
            'How many days would you like your learning plan to span?',
            min_value=2, max_value=31, value=15,
        )

        if st.button('Generate my learning plan'):
            tts.stop() # stop text-to-speech
            with st.status('Generating your learning plan...', expanded=True) as status:
                status.update(label='Generating learning plan...')
                ss.plan = create_plan(ss.performance, ss.summary, duration_days=duration)
                status.update(label='Done!', state='complete')

            release_llm(unload=True, end_of_phase=True) # release llama3.2

            st.rerun()

    # display plan
    else:
        render_study_plan(ss.plan)

    # block reruns on every interaction so write attempt once
    # and patch learning plan in once it is generated
    if 'result_id' not in ss:
        ss.result_id = save_result(
            ss.user_id,
            ss.lecture_id,
            ss.quiz,
            ss.results,
            ss.performance,
            ss.feedback,
            ss.plan
        )
    elif ss.plan is not None and not ss.get('plan_saved'):
        update_study_plan(ss.result_id, ss.plan)
        ss.plan_saved = True