"""Translation lookup, and the table itself.

The table test is the one that matters for a pull request: a Turkish string
whose placeholder was renamed raises KeyError at the moment the message is
shown, which is exactly when nobody is watching a terminal.
"""

import ast
import pathlib
import string
import unittest
from unittest import mock

import i18n
from tests.support import DikteTest

SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def translated_strings():
    """Every literal handed to t(), as {string: 'file:line'}.

    Read out of the source rather than kept as a second list, because a list
    is one more thing to forget to add to. A string built at run time is
    invisible here and has to be looked after by hand; a name() call is left
    out, since a proper noun is inflected rather than translated.
    """
    found = {}
    for path in sorted(SOURCE_DIR.glob("*.py")):
        if path.name == "i18n.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            if not isinstance(called, ast.Name) or called.id != "t":
                continue
            if node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found[node.args[0].value] = f"{path.name}:{node.lineno}"
    return found


class Resolve(unittest.TestCase):
    def test_an_explicit_language_wins(self):
        with mock.patch.dict("os.environ", {"LANG": "tr_TR.UTF-8"}):
            self.assertEqual(i18n.resolve("en"), "en")
            self.assertEqual(i18n.resolve("tr"), "tr")

    def test_auto_reads_the_locale(self):
        with mock.patch.dict("os.environ", {"LANG": "tr_TR.UTF-8"}, clear=True):
            self.assertEqual(i18n.resolve("auto"), "tr")
        with mock.patch.dict("os.environ", {"LANG": "en_GB.UTF-8"}, clear=True):
            self.assertEqual(i18n.resolve("auto"), "en")

    def test_lc_all_outranks_lang(self):
        with mock.patch.dict("os.environ",
                             {"LC_ALL": "tr_TR.UTF-8", "LANG": "en_GB.UTF-8"},
                             clear=True):
            self.assertEqual(i18n.resolve("auto"), "tr")

    def test_no_locale_at_all_falls_back_to_english(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(i18n.resolve("auto"), "en")

    def test_an_unknown_code_is_not_taken_at_its_word(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(i18n.resolve("de"), "en")


class Translate(DikteTest):
    def test_english_returns_the_source_string(self):
        self.assertEqual(i18n.t("Quit"), "Quit")

    def test_turkish_looks_the_string_up(self):
        i18n.set_language("tr")
        self.assertEqual(i18n.t("Quit"), "Çık")

    def test_an_untranslated_string_falls_through(self):
        i18n.set_language("tr")
        self.assertEqual(i18n.t("Nobody translated this"), "Nobody translated this")

    def test_placeholders_are_filled_in_both_languages(self):
        self.assertEqual(i18n.t("Unknown key: {key}", key="f13"), "Unknown key: f13")
        i18n.set_language("tr")
        self.assertIn("f13", i18n.t("Unknown key: {key}", key="f13"))

    def test_a_string_with_no_arguments_is_not_formatted(self):
        # Braces in the text itself must survive when nothing is passed in.
        self.assertEqual(i18n.t("{not a placeholder}"), "{not a placeholder}")

    def test_a_placeholder_may_be_called_anything(self):
        """Including the names of t()'s own parameters, which is why they are
        positional-only: worker.py says {text}, and that has to work."""
        self.assertEqual(i18n.t("Discarded: {text}", text="hello"), "Discarded: hello")
        self.assertEqual(i18n.name("Claude", case="dative"), "Claude")


class Names(DikteTest):
    def test_english_leaves_the_name_alone(self):
        self.assertEqual(i18n.name("Claude", "dative"), "Claude")

    def test_turkish_inflects_by_the_vowels_of_the_name(self):
        i18n.set_language("tr")
        self.assertEqual(i18n.name("Claude", "dative"), "Claude'a")
        self.assertEqual(i18n.name("Codex", "dative"), "Codex'e")
        self.assertEqual(i18n.name("OpenRouter", "accusative"), "OpenRouter'ı")

    def test_no_case_asked_for(self):
        i18n.set_language("tr")
        self.assertEqual(i18n.name("Claude"), "Claude")

    def test_an_unlisted_name_or_case_comes_back_unchanged(self):
        i18n.set_language("tr")
        self.assertEqual(i18n.name("Ollama", "dative"), "Ollama")
        self.assertEqual(i18n.name("Claude", "ablative"), "Claude")


class Table(unittest.TestCase):
    """The Turkish table against the English strings it stands in for."""

    def test_every_translation_keeps_the_placeholders_of_its_source(self):
        for source, translated in i18n.TR.items():
            with self.subTest(source=source[:50]):
                self.assertEqual(
                    placeholders(source), placeholders(translated),
                    "the Turkish string does not take the same arguments",
                )

    def test_nothing_is_translated_to_an_empty_string(self):
        for source, translated in i18n.TR.items():
            with self.subTest(source=source[:50]):
                self.assertTrue(translated.strip())

    def test_every_translation_is_formattable(self):
        """Whatever the table holds, .format() must not blow up on it."""
        for source, translated in i18n.TR.items():
            names = placeholders(translated)
            if not names:
                continue
            with self.subTest(source=source[:50]):
                translated.format(**{key: "x" for key in names})

    def test_every_string_the_code_shows_has_a_turkish_one(self):
        """The half of a feature that is easy to forget.

        A window built in English works, so nothing fails and nothing is
        noticed: the strings pile up until a whole tab is one language and the
        rest another. Reading the calls out of the source is what turns that
        into something the suite can say out loud.
        """
        missing = sorted(
            source for source, where in translated_strings().items()
            if source not in i18n.TR
        )
        self.assertEqual(missing, [], "no Turkish for: " + "; ".join(
            f"{text[:60]!r}" for text in missing[:8]
        ))


if __name__ == "__main__":
    unittest.main()
