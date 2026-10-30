"""Choose truthful keyframes for a shortened, captioned laptop demo recording.

Only the caller's actual captured frames are selected. Long unchanged waits
collapse to one representative frame; meaningful changes retain the last frame.
"""


def _caption(phase, statuses):
    if phase == 'previous_result':
        return 'A previous run is visible; the next upload must clear it.'
    if phase == 'unsupported_rejected':
        return 'Unsupported file rejected. Old input and results are cleared.'
    if phase == 'new_input_selected':
        return 'Choose a new page and the models to run sequentially.'
    if phase == 'markdown_inspection':
        return 'Inspect the real parsed Markdown; the full text is downloadable.'
    if phase == 'json_inspection':
        return 'Inspect structured JSON; raw output is downloadable.'
    if phase in ('final_comparison', 'comparison_complete'):
        return 'One synthetic smoke page on this laptop, not a benchmark or all-input guarantee.'
    if phase != 'live_run':
        return 'Live browser capture from the laptop GPU run.'
    states = [str(x).lower() for x in statuses]
    if any('fail' in x for x in states):
        return 'A model failed. Inspect its reported error; no output is substituted.'
    if states and all(x == 'done' for x in states):
        return 'All three models finished. Compare warm page latency and memory.'
    if len(states) >= 3 and 'loading' in states[2]:
        return 'W4A16 loads after W8A16 releases the GPU; waiting shortened.'
    if len(states) >= 3 and 'parsing' in states[2]:
        return 'W4A16 parses the page and saves Markdown and JSON.'
    if len(states) >= 2 and 'loading' in states[1]:
        return 'W8A16 loads after BF16 releases the GPU; waiting shortened.'
    if len(states) >= 2 and 'parsing' in states[1]:
        return 'W8A16 parses the same page; measured latency is page-only.'
    if states and 'loading' in states[0]:
        return 'BF16 checkpoint is checked and loaded; waiting shortened.'
    if states and 'parsing' in states[0]:
        return 'BF16 parses the page and saves real outputs.'
    return 'The selected models run one at a time on the same GPU.'


def concat_manifest(clips, paths):
    """Build FFmpeg concat-demuxer input and hold the final still correctly."""
    if not clips:
        raise ValueError('No clips')
    lines = []
    for clip in clips:
        path = paths[clip['index']]
        if not path.is_absolute() or clip['seconds'] <= 0:
            raise ValueError('Clips need absolute paths and positive durations')
        escaped = str(path).replace("'", "'\\''")
        lines.extend((f"file '{escaped}'", f"duration {clip['seconds']:.3f}"))
    lines.append(lines[-2])
    return '\n'.join(lines) + '\n'


def build_storyboard(records, *, difference, max_seconds=55):
    """Group adjacent semantic states; return keyframe indices and durations.

    `difference(start, end)` is a caller-provided perceptual score; a score
    above 3 retains a late frame within an otherwise unchanged status group.
    """
    if not records:
        raise ValueError('No recorded frames')
    segments = []
    for index, record in enumerate(records):
        phase = record['phase']
        key = (phase, tuple(record.get('statuses', ())) if phase == 'live_run' else ())
        if not segments or segments[-1]['key'] != key:
            segments.append({'key': key, 'first': index, 'last': index})
        else:
            segments[-1]['last'] = index
    clips = []
    for segment in segments:
        phase, statuses = segment['key']
        duration = {
            'previous_result': 2, 'unsupported_rejected': 3,
            'new_input_selected': 3, 'markdown_inspection': 4,
            'json_inspection': 4, 'final_comparison': 6,
            'comparison_complete': 2,
        }.get(phase, 3)
        first, last = segment['first'], segment['last']
        indices = [first]
        if last > first and difference(first, last) > 3:
            indices.append(last)
        caption = _caption(phase, statuses)
        for chosen in indices:
            clips.append({'index': chosen, 'seconds': duration / len(indices),
                          'caption': caption, 'stage': phase})
    total = sum(clip['seconds'] for clip in clips)
    if total > max_seconds:
        for clip in clips:
            clip['seconds'] = round(clip['seconds'] * max_seconds / total, 3)
    return clips
