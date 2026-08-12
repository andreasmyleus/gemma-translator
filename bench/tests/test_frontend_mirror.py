import unittest

from bench.frontend_mirror import (
    build_llm_payload,
    parse_translation,
    split_text_into_speech_chunks,
    system_prompt,
)


class TestSplitTextIntoSpeechChunks(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(
            split_text_into_speech_chunks("Var ligger stationen?"),
            ["Var ligger stationen?"],
        )

    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(split_text_into_speech_chunks(""), [])

    def test_splits_on_word_boundary_under_limit(self):
        text = " ".join(["ord"] * 10)  # 10 * 4 - 1 = 39 tecken
        chunks = split_text_into_speech_chunks(text, limit=20)
        self.assertEqual(chunks, ["ord ord ord ord ord", "ord ord ord ord ord"])
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)

    def test_word_longer_than_limit_becomes_its_own_chunk(self):
        # Speglar JS-beteendet: ett enskilt ord delas aldrig, även om det
        # spränger gränsen. Porten måste bete sig likadant.
        chunks = split_text_into_speech_chunks("kort " + "x" * 30, limit=10)
        self.assertEqual(chunks, ["kort", "x" * 30])

    def test_collapses_runs_of_whitespace(self):
        self.assertEqual(
            split_text_into_speech_chunks("ett   två\n\ttre"),
            ["ett två tre"],
        )


class TestSystemPrompt(unittest.TestCase):
    def test_uses_first_word_of_language_names(self):
        prompt = system_prompt("Swedish (Source)", "English (Translation)")
        self.assertIn("from Swedish into English", prompt)
        self.assertNotIn("(Source)", prompt)

    def test_demands_bare_json_object(self):
        prompt = system_prompt("Swedish", "English")
        self.assertIn('"translation"', prompt)


class TestBuildLlmPayload(unittest.TestCase):
    def test_system_message_precedes_user_message(self):
        payload = build_llm_payload("hej", "gemma4-e2b", "SYS")
        self.assertEqual(payload["model"], "gemma4-e2b")
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hej"},
            ],
        )

    def test_blank_system_prompt_is_omitted(self):
        payload = build_llm_payload("hej", "gemma4-e2b", "   ")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hej"}])


class TestParseTranslation(unittest.TestCase):
    def test_extracts_field_from_bare_json(self):
        self.assertEqual(parse_translation('{"translation": "Where is it?"}'), "Where is it?")

    def test_tolerates_json_code_fence(self):
        raw = '```json\n{"translation": "Where is it?"}\n```'
        self.assertEqual(parse_translation(raw), "Where is it?")

    def test_falls_back_to_raw_text_when_not_json(self):
        self.assertEqual(parse_translation("Where is it?"), "Where is it?")


if __name__ == "__main__":
    unittest.main()
