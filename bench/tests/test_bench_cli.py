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

"""Vakter runt CLI:t: portkontroll, udda repetitioner, flaggkombinationer."""

import socket
import sys
import unittest
from contextlib import closing
from unittest import mock

from bench import bench
from bench.frontend_mirror import system_prompt


def free_port():
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestAssertPortFree(unittest.TestCase):
    def test_free_port_passes(self):
        bench.assert_port_free(free_port())

    def test_occupied_port_fails_loudly(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with self.assertRaises(SystemExit) as caught:
                bench.assert_port_free(port)
            self.assertIn(str(port), str(caught.exception))
        finally:
            listener.close()

    def test_start_backend_refuses_before_spawning_a_child(self):
        """Det här är C1: utan kontrollen svarade den kvarglömda lyssnaren på
        beredskapstestet medan barnet dog på "Address already in use", och arm B
        kördes mot arm A:s process — alla kvoter ~1,000, alltså exakt vad "ingen
        effekt" ser ut som."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with mock.patch.object(bench.subprocess, "Popen") as popen:
                with self.assertRaises(SystemExit):
                    bench.start_backend(port)
            popen.assert_not_called()
        finally:
            listener.close()


class TestArgumentGuards(unittest.TestCase):
    def run_main(self, *argv):
        with mock.patch.object(sys, "argv", ["bench", *argv]):
            with self.assertRaises(SystemExit) as caught:
                bench.main()
        return str(caught.exception)

    def test_even_repeats_are_refused(self):
        # ABBA: repetition 0 kastas, så ett jämnt antal ger en arm en
        # ordningsposition mer än den andra.
        self.assertIn("udda", self.run_main("--label", "x", "--repeats", "4"))

    def test_zero_repeats_are_refused(self):
        self.assertIn("minst 1", self.run_main("--label", "x", "--repeats", "0"))

    def test_odd_repeats_pass_the_guard(self):
        # Går vidare till load_fixtures, som vi stoppar här: poängen är bara att
        # grinden inte fällde.
        with mock.patch.object(bench, "load_fixtures", side_effect=RuntimeError("nådde fixturerna")):
            with mock.patch.object(sys, "argv", ["bench", "--label", "x", "--repeats", "5"]):
                with self.assertRaisesRegex(RuntimeError, "nådde fixturerna"):
                    bench.main()

    def test_stream_is_refused_together_with_an_ab_flag(self):
        # run_ab låser arm A till icke-strömmande, så --stream hade tyst
        # ignorerats för arm A.
        message = self.run_main("--label", "x", "--stream", "--ab-stream")
        self.assertIn("--ab-stream", message)
        self.assertIn("--stream", self.run_main("--label", "x", "--stream", "--ab", "STT_VAD=1"))


class TestPromptSelection(unittest.TestCase):
    def test_default_prompt_is_the_products(self):
        self.assertEqual(bench.DEFAULT_PROMPT, "plain")
        self.assertIs(bench.PROMPT_VARIANTS["plain"], system_prompt)

    def test_json_variant_is_still_available_for_reproduction(self):
        self.assertIn("json", bench.PROMPT_VARIANTS)
        self.assertIn(
            '"translation"', bench.PROMPT_VARIANTS["json"]("Swedish", "English")
        )


if __name__ == "__main__":
    unittest.main()
