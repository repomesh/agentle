import sys
import unittest
from pathlib import Path

AGENTLE_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTLE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTLE_ROOT))

from agentle.agents.channels.providers.whatsapp_cloud._whatsapp_markdown import (
    to_whatsapp_markdown,
)


class WhatsAppMarkdownTests(unittest.TestCase):
    def test_double_star_bold_becomes_single(self):
        self.assertEqual(
            to_whatsapp_markdown("**Exame: Periapical**"), "*Exame: Periapical*"
        )

    def test_underscore_bold_becomes_single_star(self):
        self.assertEqual(to_whatsapp_markdown("__importante__"), "*importante*")

    def test_bold_italic_triple_star(self):
        self.assertEqual(to_whatsapp_markdown("***muito***"), "*muito*")

    def test_headings_become_bold_lines(self):
        self.assertEqual(to_whatsapp_markdown("## Horários"), "*Horários*")
        self.assertEqual(to_whatsapp_markdown("# Título"), "*Título*")
        self.assertEqual(to_whatsapp_markdown("### Sexta-feira ###"), "*Sexta-feira*")

    def test_strikethrough(self):
        self.assertEqual(to_whatsapp_markdown("~~cancelado~~"), "~cancelado~")

    def test_links(self):
        self.assertEqual(
            to_whatsapp_markdown("[Mapa](https://maps.app.goo.gl/abc)"),
            "Mapa (https://maps.app.goo.gl/abc)",
        )
        self.assertEqual(
            to_whatsapp_markdown("[https://x.com](https://x.com)"), "https://x.com"
        )

    def test_single_markers_left_untouched(self):
        # single * is already WhatsApp bold; single _ already WhatsApp italic
        self.assertEqual(to_whatsapp_markdown("*ok*"), "*ok*")
        self.assertEqual(to_whatsapp_markdown("_ok_"), "_ok_")

    def test_multiline_and_multiple_bolds(self):
        src = "**Exame: Periapical**\n**Data: 27/03/2026 às 10:15**"
        self.assertEqual(
            to_whatsapp_markdown(src),
            "*Exame: Periapical*\n*Data: 27/03/2026 às 10:15*",
        )

    def test_plain_text_unchanged(self):
        text = "Bom dia, Arthur! Em que posso ajudar?"
        self.assertEqual(to_whatsapp_markdown(text), text)

    def test_non_string_and_empty(self):
        self.assertEqual(to_whatsapp_markdown(""), "")
        self.assertIsNone(to_whatsapp_markdown(None))


if __name__ == "__main__":
    unittest.main()
