import unittest
from pathlib import Path

from capstone_demo.video_storyboard import build_storyboard, concat_manifest


class StoryboardTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'phase': 'previous_result', 'statuses': ['DONE', 'DONE', 'DONE']},
            {'phase': 'unsupported_rejected', 'statuses': []},
            {'phase': 'new_input_selected', 'statuses': []},
            {'phase': 'live_run', 'statuses': ['VERIFYING + LOADING', 'WAITING', 'WAITING']},
            {'phase': 'live_run', 'statuses': ['VERIFYING + LOADING', 'WAITING', 'WAITING']},
            {'phase': 'live_run', 'statuses': ['PARSING PAGE', 'WAITING', 'WAITING']},
            {'phase': 'live_run', 'statuses': ['DONE', 'VERIFYING + LOADING', 'WAITING']},
            {'phase': 'live_run', 'statuses': ['DONE', 'DONE', 'VERIFYING + LOADING']},
            {'phase': 'live_run', 'statuses': ['DONE', 'DONE', 'DONE']},
            {'phase': 'markdown_inspection', 'statuses': ['DONE', 'DONE', 'DONE']},
            {'phase': 'json_inspection', 'statuses': ['DONE', 'DONE', 'DONE']},
            {'phase': 'final_comparison', 'statuses': ['DONE', 'DONE', 'DONE']},
        ]

    def test_static_waits_are_collapsed_and_scope_is_explicit(self):
        clips = build_storyboard(self.records, difference=lambda a, b: 0, max_seconds=55)
        self.assertEqual([c['index'] for c in clips], [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
        self.assertLess(sum(c['seconds'] for c in clips), 60)
        self.assertIn('waiting shortened', clips[3]['caption'].lower())
        self.assertIn('not a benchmark', clips[-1]['caption'].lower())
        self.assertTrue(all(c['seconds'] > 0 for c in clips))

    def test_a_visible_change_inside_one_stage_keeps_a_late_frame(self):
        frames = self.records[:3] + [self.records[3]] * 8 + self.records[-1:]
        clips = build_storyboard(frames, difference=lambda a, b: 8 if b - a > 3 else 0, max_seconds=55)
        indices = [c['index'] for c in clips]
        self.assertIn(3, indices)
        self.assertIn(10, indices)
        self.assertLess(sum(c['seconds'] for c in clips), 60)
    def test_concat_manifest_preserves_clip_durations_and_final_frame(self):
        clips = [{'index': 0, 'seconds': 2.0}, {'index': 1, 'seconds': 3.0}]
        paths = [Path('/tmp/frame 01.png'), Path('/tmp/frame02.png')]
        self.assertEqual(
            concat_manifest(clips, paths),
            "file '/tmp/frame 01.png'\nduration 2.000\n"
            "file '/tmp/frame02.png'\nduration 3.000\n"
            "file '/tmp/frame02.png'\n",
        )


if __name__ == '__main__':
    unittest.main()
