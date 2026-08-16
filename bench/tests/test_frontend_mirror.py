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
    ends_with_continuation_cue,
    far_end_sample_index,
    first_sentence_end,
    is_backchannel,
    is_repair_utterance,
    is_speakable,
    looks_like_json_envelope,
    normalize_stt_text,
    parse_translation,
    route_spoken_turn,
    split_text_into_speech_chunks,
    strip_repair_cue,
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

    def test_punctuation_only_yields_no_chunks(self):
        self.assertEqual(split_text_into_speech_chunks("."), [])
        self.assertEqual(split_text_into_speech_chunks("..."), [])
        self.assertEqual(split_text_into_speech_chunks("?!"), [])

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


class TestConversationHelpers(unittest.TestCase):
    def test_backchannel_is_filler_not_an_answer(self):
        self.assertTrue(is_backchannel("mm"))
        self.assertTrue(is_backchannel("Mhm."))
        self.assertTrue(is_backchannel("uh"))
        self.assertTrue(is_backchannel("öh"))
        self.assertFalse(is_backchannel("ja"))
        self.assertFalse(is_backchannel("yes"))
        self.assertFalse(is_backchannel("Var ligger stationen?"))

    def test_trailing_conjunction_holds_the_turn(self):
        self.assertTrue(ends_with_continuation_cue("Jag vill ha kaffe och"))
        self.assertTrue(ends_with_continuation_cue("I want coffee and"))
        self.assertTrue(ends_with_continuation_cue("Quiero café y"))
        self.assertFalse(ends_with_continuation_cue("Var ligger stationen?"))
        self.assertFalse(ends_with_continuation_cue("ja"))

    def test_finished_sentences_are_not_held_as_continuations(self):
        # "that" / "så" / "att" are normal sentence endings, not trail-offs.
        # Holding them added ~1.5s of dead air to ordinary turns.
        self.assertFalse(ends_with_continuation_cue("I need that"))
        self.assertFalse(ends_with_continuation_cue("Ja, så."))
        self.assertFalse(ends_with_continuation_cue("Jag vet att"))
        self.assertFalse(ends_with_continuation_cue("Thanks for that"))

    def test_repair_cue_strips_the_preamble(self):
        self.assertTrue(is_repair_utterance("Nej, var ligger toaletten?"))
        self.assertTrue(is_repair_utterance("I mean the station"))
        self.assertFalse(is_repair_utterance("Var ligger stationen?"))
        self.assertEqual(
            strip_repair_cue("Nej, var ligger toaletten?"),
            "var ligger toaletten?",
        )
        self.assertEqual(strip_repair_cue("Nej"), "Nej")

    def test_whisper_nospeech_tokens_are_stripped(self):
        self.assertEqual(normalize_stt_text("<|nospeech|> ."), ".")
        self.assertEqual(normalize_stt_text("<|nospeech|>"), "")
        self.assertFalse(is_speakable(normalize_stt_text("<|nospeech|> .")))
        self.assertEqual(
            normalize_stt_text("Hej <|nospeech|> där"),
            "Hej där",
        )


class TestRouteSpokenTurn(unittest.TestCase):
    """Two people pick a language each and just talk — no Enter required.

    STT says which of the two languages was spoken; the turn is attributed
    to that lane and translated into the other.
    """

    SV = {"code": "sv", "name": "Swedish", "ttsLang": "sv"}
    EN = {"code": "en", "name": "English", "ttsLang": "en"}
    FI = {"code": "fi", "name": "Finnish", "ttsLang": "fi"}

    def test_other_lane_language_flips_speaker_and_direction(self):
        routed = route_spoken_turn("en", 1, self.SV, self.EN, self.SV, self.EN)
        self.assertEqual(routed["lane"], 2)
        self.assertEqual(routed["src"]["code"], "en")
        self.assertEqual(routed["dst"]["code"], "sv")
        self.assertTrue(routed["flipped"])

    def test_expected_language_keeps_the_active_lane(self):
        routed = route_spoken_turn("sv", 1, self.SV, self.EN, self.SV, self.EN)
        self.assertEqual(routed["lane"], 1)
        self.assertEqual(routed["src"]["code"], "sv")
        self.assertEqual(routed["dst"]["code"], "en")
        self.assertFalse(routed["flipped"])

    def test_unknown_or_third_language_keeps_the_lane_prior(self):
        routed = route_spoken_turn("de", 1, self.SV, self.EN, self.SV, self.EN)
        self.assertEqual(routed["lane"], 1)
        self.assertEqual(routed["src"]["code"], "sv")
        self.assertFalse(routed["flipped"])
        routed = route_spoken_turn("fi", 2, self.EN, self.SV, self.SV, self.EN)
        self.assertEqual(routed["lane"], 2)
        self.assertFalse(routed["flipped"])

    def test_region_suffix_is_ignored(self):
        routed = route_spoken_turn("en-US", 1, self.SV, self.EN, self.SV, self.EN)
        self.assertEqual(routed["lane"], 2)
        self.assertTrue(routed["flipped"])


class TestFarEndSampleIndex(unittest.TestCase):
    """AEC far-end index must convert mic time into TTS sample time.

    Piper is 22050 Hz; the AudioContext is typically 48000 Hz. Adding the
    mic frame index as if the rates matched reads the reference ~2× too
    fast, so TTS leaked into the next capture.
    """

    def test_same_rate_is_just_offset_plus_index(self):
        self.assertEqual(far_end_sample_index(0.0, 10, 48000, 48000), 10)
        self.assertEqual(far_end_sample_index(0.5, 0, 48000, 48000), 24000)

    def test_piper_rate_against_48k_mic_scales_the_index(self):
        # 1 ms into playback, 48th mic sample of the frame → still ~1 ms of TTS.
        idx = far_end_sample_index(0.001, 48, 22050, 48000)
        self.assertEqual(idx, 44)  # floor(0.001*22050 + 48*22050/48000) = 44

    def test_old_plus_i_formula_is_wrong_across_rates(self):
        elapsed = 0.001
        i = 48
        tts_rate = 22050
        wrong = int(elapsed * tts_rate) + i
        right = far_end_sample_index(elapsed, i, tts_rate, 48000)
        self.assertNotEqual(wrong, right)


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
