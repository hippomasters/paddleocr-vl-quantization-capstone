import unittest
from capstone_demo.paths import STATIC_DIR


class PresentationThemeTests(unittest.TestCase):
    def test_demo_uses_attached_nmims_presentation_palette(self):
        css = (STATIC_DIR / 'style.css').read_text().lower()
        for color in ('#1f3a5f', '#2e7d8c', '#e0a526', '#f4f6f8', '#d5dae0'):
            self.assertIn(color, css)
        self.assertNotIn('#116f62', css)
        self.assertNotIn('#168575', css)
        self.assertIn('times new roman', css)


if __name__ == '__main__':
    unittest.main()
