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

import json
import unittest
from unittest import mock

from bench import runner
from bench.runner import RunResult


def sse(**choice):
    return "data: " + json.dumps({"choices": [choice]})


class FakeResponse:
    """Precis så mycket av requests.Response som _post_llm_streaming rör."""

    def __init__(self, lines):
        self._lines = lines
        self.encoding = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def run_streaming(lines):
    with mock.patch.object(runner.requests, "post", return_value=FakeResponse(lines)):
        return runner._post_llm_streaming(
            "http://api", "http://llm", "gemma4-e2b", "hej", "sv", "en"
        )


class TestStreamCompleteness(unittest.TestCase):
    """En avbruten ström ger halv översättning MEN korta tider — en falsk vinst
    som annars bokförs som en lyckad, snabb körning."""

    def test_complete_stream_is_accepted(self):
        translation, total_ms, first_ms = run_streaming(
            [
                sse(delta={"content": "Hi."}),
                sse(delta={"content": " There."}),
                sse(delta={}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )
        self.assertEqual(translation, "Hi. There.")
        self.assertGreater(total_ms, 0)
        self.assertLessEqual(first_ms, total_ms)

    def test_stream_without_done_but_with_finish_reason_is_accepted(self):
        translation, _, _ = run_streaming(
            [sse(delta={"content": "Hi."}), sse(delta={}, finish_reason="stop")]
        )
        self.assertEqual(translation, "Hi.")

    def test_truncated_stream_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trunkerat"):
            run_streaming([sse(delta={"content": "Hi. Half a sen"})])

    def test_first_sentence_is_detected_positionally(self):
        # Deltat slutar mitt i mening två; en endswith-detektor hade missat den
        # första meningen helt och rapporterat first_sentence == total.
        _, total_ms, first_ms = run_streaming(
            [
                sse(delta={"content": "Hi. And th"}),
                sse(delta={"content": "en some more text here."}),
                "data: [DONE]",
            ]
        )
        self.assertLess(first_ms, total_ms)


class TestTimeToFirstAudio(unittest.TestCase):
    """time_to_first_audio_ms must switch its LLM term on llm_first_sentence_ms,
    or a streaming measurement silently degrades into measuring the full
    response and the whole point of Task 15 is lost.
    """

    def test_non_streaming_run_uses_full_llm_time(self):
        # llm_first_sentence_ms defaults to 0.0 (never measured): the
        # non-streaming arm must fall back to the full llm_ms.
        result = RunResult(fixture_id="sv-short", stt_ms=100.0, llm_ms=1000.0, tts_first_ms=50.0)
        self.assertEqual(result.time_to_first_audio_ms, 1150.0)

    def test_streaming_run_uses_first_sentence_time_not_full_response(self):
        result = RunResult(
            fixture_id="sv-multi",
            stt_ms=100.0,
            llm_ms=3000.0,
            llm_first_sentence_ms=900.0,
            tts_first_ms=50.0,
        )
        self.assertEqual(result.time_to_first_audio_ms, 1050.0)


if __name__ == "__main__":
    unittest.main()
