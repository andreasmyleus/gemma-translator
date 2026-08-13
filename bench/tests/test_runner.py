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

from bench.runner import RunResult


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
