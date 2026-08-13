# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from bench.frontend_mirror import (
    build_llm_payload,
    first_sentence_end,
    looks_like_json_envelope,
    parse_translation,
    split_text_into_speech_chunks,
    system_prompt,
    system_prompt_json,
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
    """`system_prompt` är bench:s default och MÅSTE vara produktens prompt.

    Var den något annat mätte varje enarmskörning — och varje arm A — en
    prompt appen inte skickar. Det är inte en detalj: promptlängden avgör om
    en fixtur hamnar på fel sida kontextklippan vid ~725 tecken.
    """

    def test_uses_first_word_of_language_names(self):
        prompt = system_prompt("Swedish (Source)", "English (Translation)")
        self.assertIn("from Swedish into English", prompt)
        self.assertNotIn("(Source)", prompt)

    def test_default_prompt_is_the_plain_one_the_product_sends(self):
        prompt = system_prompt("Swedish", "English")
        self.assertNotIn("JSON", prompt)
        self.assertNotIn('translation":', prompt)

    def test_stays_short(self):
        # Prompten prefillas vid varje anrop; håll den kort.
        self.assertLess(len(system_prompt("Swedish", "English")), 200)


class TestSystemPromptJson(unittest.TestCase):
    def test_uses_first_word_of_language_names(self):
        prompt = system_prompt_json("Swedish (Source)", "English (Translation)")
        self.assertIn("from Swedish into English", prompt)
        self.assertNotIn("(Source)", prompt)

    def test_demands_bare_json_object(self):
        self.assertIn('"translation"', system_prompt_json("Swedish", "English"))


class TestFirstSentenceEnd(unittest.TestCase):
    """Speglar TranslatorApp.speakCompleteSentences.

    Regressionen som fanns: bench använde `endswith`, som bara avfyrar när ett
    delta *slutar* på skiljetecken. Produkten använder `lastIndexOf` och
    avfyrar på första delta som *innehåller* ett. På den riktiga
    sv-multi-översättningen gav det 165 mot 52 tecken, alltså
    first_sentence == hela svaret, alltså "strömning är värdelös".
    """

    SV_MULTI = (
        "I would need to book a room for two nights. Could you also tell me if "
        "breakfast is included in the price? I also wonder where the nearest ATM is."
    )

    def test_fires_on_a_delta_that_only_contains_punctuation(self):
        # Delta slutar mitt i nästa mening — endswith hade missat den här.
        self.assertEqual(first_sentence_end("Hej där. Sedan"), 8)

    def test_matches_the_products_position_on_the_real_sv_multi_output(self):
        upto = first_sentence_end(self.SV_MULTI[:60])
        self.assertEqual(upto, 43)
        self.assertEqual(
            self.SV_MULTI[:upto], "I would need to book a room for two nights."
        )

    def test_no_punctuation_yields_none(self):
        self.assertIsNone(first_sentence_end("Ingen mening än"))

    def test_takes_the_last_complete_sentence_in_the_delta(self):
        # lastIndexOf, inte indexOf: allt färdigt talas i ett svep.
        self.assertEqual(first_sentence_end("Ett. Två! Tre"), 9)

    def test_already_spoken_prefix_is_not_repeated(self):
        self.assertIsNone(first_sentence_end("Ett. Två", spoken_chars=4))
        self.assertEqual(first_sentence_end("Ett. Två.", spoken_chars=4), 9)

    def test_whitespace_only_remainder_is_not_spoken(self):
        # Appens `if (!ready) return`.
        self.assertIsNone(first_sentence_end("Ett.   ", spoken_chars=4))

    def test_json_envelope_is_never_spoken_partially(self):
        # Appen skickar inga partiella deltan alls för ett wrapper-svar, så
        # bench får inte påstå att en mening kunde talas då.
        self.assertIsNone(first_sentence_end('{"translation": "Hej.'))
        self.assertIsNone(first_sentence_end('```json\n{"translation": "Hej.'))


class TestLooksLikeJsonEnvelope(unittest.TestCase):
    def test_detects_object_and_fence_starts(self):
        self.assertTrue(looks_like_json_envelope('  {"translation"'))
        self.assertTrue(looks_like_json_envelope("```json"))

    def test_plain_text_is_not_an_envelope(self):
        self.assertFalse(looks_like_json_envelope("Where is the station?"))


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

    # Nedan speglar JS:ens `parsed.translation || ""`. Verifierat mot node med
    # translateText-blocket i api.js:126-144 kopierat rakt av.
    def test_null_translation_becomes_empty_string(self):
        self.assertEqual(parse_translation('{"translation": null}'), "")

    def test_falsy_translation_values_become_empty_string(self):
        self.assertEqual(parse_translation('{"translation": ""}'), "")
        self.assertEqual(parse_translation('{"translation": 0}'), "")
        self.assertEqual(parse_translation('{"translation": false}'), "")

    def test_missing_key_becomes_empty_string(self):
        self.assertEqual(parse_translation('{"other": "x"}'), "")

    def test_valid_json_that_is_not_an_object_becomes_empty_string(self):
        # Property-access på ett tal, en array eller en sträng ger undefined
        # i JS, och `undefined || ""` normaliserar till "".
        self.assertEqual(parse_translation("5"), "")
        self.assertEqual(parse_translation("[1,2,3]"), "")
        self.assertEqual(parse_translation('"hello"'), "")

    def test_bare_null_falls_back_to_raw_text(self):
        # Specialfall: `null.translation` kastar TypeError inne i JS:ens
        # try-block, så appen hamnar i catch och visar rå text — till skillnad
        # från 5/[1,2,3]/"hello" ovan, som ger "".
        self.assertEqual(parse_translation("null"), "null")

    def test_nonstandard_json_constants_fall_back_to_raw_text(self):
        # Pythons json godtar NaN/Infinity, JSON.parse gör det inte.
        self.assertEqual(parse_translation("NaN"), "NaN")
        self.assertEqual(parse_translation('{"translation": NaN}'), '{"translation": NaN}')


if __name__ == "__main__":
    unittest.main()
