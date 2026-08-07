import instrumentation as inst

@inst.timed("fusion of extracted text")
def fuse(transcript, slides, notes, figure):
    """
    combines all pieces of extracted text together
    """

    parts = []
    components = [
        {'name': "TRANSCRIPT", 'content': transcript}, 
        {'name': "SLIDE", 'content': slides}, 
        {'name': "NOTES", 'content': notes}, 
        {'name': "FIGURE", 'content': figure}
    ]

    for component in components:
        content = component['content'].strip()
        if content and content.lower() != "none":
            parts.append(f"[{component['name']}]\n{content}")

    fusion = '\n\n'.join(parts)

    return fusion