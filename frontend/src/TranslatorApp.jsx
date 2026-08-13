/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, { useState, useEffect, useRef, useCallback } from "react"
import LanguageLane from "./components/LanguageLane"
import ResponseDrawer from "./components/ResponseDrawer"
import Visualizer from "./components/Visualizer"
import { useAudioRecorder } from "./hooks/useAudioRecorder"
import {
  transcribeAudio,
  translateTextStreaming,
  splitTextIntoSpeechChunks,
} from "./utils/api"
import { playBlip } from "./utils/audio-blip"

// Core orchestrator for the two-person kiosk translator.
// Flow: hold a key → record mic (useAudioRecorder) → POST /api/stt (Whisper)
// → LLM translation via /proxy (Gemma, strict-JSON prompt) → /api/tts playback.

// Languages offered on each lane's revolver; ttsLang selects the backend voice.
const AVAILABLE_LANGUAGES = [
  { code: "sv", name: "Swedish", voice: "tts", ttsLang: "sv" },
  { code: "en", name: "English", voice: "tts", ttsLang: "en" },
  { code: "fi", name: "Finnish", voice: "tts", ttsLang: "fi" },
  { code: "ar", name: "Arabic", voice: "tts", ttsLang: "ar" },
  { code: "es", name: "Spanish", voice: "tts", ttsLang: "es" },
  { code: "ja", name: "Japanese", voice: "tts", ttsLang: "ja" },
  { code: "zh", name: "Chinese", voice: "tts", ttsLang: "zh" },
  { code: "ko", name: "Korean", voice: "tts", ttsLang: "ko" },
]

function TranslatorApp({ config }) {
  // UI State
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [activePerson, setActivePerson] = useState(1)

  // Translation State
  const [transcriptionData, setTranscriptionData] = useState({
    source: "",
    text: "— listening —",
  })
  const [translationData, setTranslationData] = useState({
    target: "",
    text: "— waiting —",
  })
  const [metaText, setMetaText] = useState("")

  // Currently-playing TTS audio element (chunked playback chain)
  const onlineAudioPlayerRef = useRef(null)

  // Timestamps for latency measurement; reset on every key release.
  const timingRef = useRef(null)

  // Language Lanes State
  const [lang1Index, setLang1Index] = useState(0)
  const [lang2Index, setLang2Index] = useState(1)
  const [activeLaneRecording, setActiveLaneRecording] = useState(null) // 1 or 2

  const { isRecording, startRecording, stopRecording, analyser, micError } =
    useAudioRecorder()

  useEffect(() => {
    if (micError) {
      setIsDrawerOpen(true)
      setTranscriptionData({ source: "Microphone", text: "Access Failed" })
      setTranslationData({
        target: "Error",
        text: `${micError} (HTTPS is required when accessing from remote devices)`,
      })
    }
  }, [micError])

  // Sentences waiting to be spoken. Each entry is { player, marks }: the Audio
  // element is already constructed (creating it starts the fetch, so queued
  // chunks download while the current one plays), and `marks` is the timing
  // object that was current when the chunk was queued.
  const ttsQueueRef = useRef({ pending: [], playing: false })

  // Bumped whenever a new recording starts. Everything a translation does
  // afterwards — speaking, writing the panels — is guarded on still owning the
  // current value, so an interrupted translation cannot talk over the next one.
  const generationRef = useRef(0)
  // The in-flight streamed translation, so it can be aborted when superseded.
  const inflightRef = useRef(null)

  // Report one utterance's latency. `marks` is passed in rather than read from
  // timingRef: re-reading the ref here would let a late-firing chunk from a
  // previous utterance stamp `logged` on the *next* utterance's marks, silently
  // suppressing its measurement (see commit 08c6df0).
  const reportLatency = useCallback((marks, audioStarted) => {
    if (!marks || marks.logged) return
    marks.logged = true
    // An unset mark prints as "—", never as a fabricated 0ms: `llm` is stamped
    // at the first token, so it is set well before audio starts, but a failed
    // or skipped stage must not read as instantaneous.
    const since = (mark) =>
      typeof mark === "number" ? `${(mark - marks.keyup) | 0}ms` : "—"
    const audio = audioStarted ? since(performance.now()) : "—"
    console.log(
      `[latency] keyup→stt ${since(marks.stt)} | →llm first token ${since(marks.llm)} ` +
        `| →first audio ${audio}`,
    )
    setMetaText(`STT ${since(marks.stt)} · LLM ${since(marks.llm)} · ljud ${audio}`)
  }, [])

  const stopSpeaking = useCallback(() => {
    ttsQueueRef.current.pending = []
    ttsQueueRef.current.playing = false
    if (onlineAudioPlayerRef.current) {
      onlineAudioPlayerRef.current.pause()
      onlineAudioPlayerRef.current = null
    }
  }, [])

  const pumpTTSQueue = useCallback(() => {
    const queue = ttsQueueRef.current
    if (queue.playing) return
    const entry = queue.pending.shift()
    if (!entry) return
    const { player, marks } = entry
    queue.playing = true
    onlineAudioPlayerRef.current = player
    player.onplaying = () => reportLatency(marks, true)
    player.onended = () => {
      queue.playing = false
      pumpTTSQueue()
    }
    player.onerror = () => {
      queue.playing = false
      stopSpeaking()
      alert("TTS playback failed. Backend server may be offline.")
    }
    player.play().catch((e) => {
      console.error("Audio play error:", e)
      queue.playing = false
      stopSpeaking()
    })
  }, [reportLatency, stopSpeaking])

  // Queue text for playback. Safe to call repeatedly as sentences arrive.
  const enqueueTTS = useCallback(
    (text, targetLang, marks) => {
      if (!text) return
      for (const chunk of splitTextIntoSpeechChunks(text)) {
        const url = `/api/tts?text=${encodeURIComponent(chunk)}&lang=${encodeURIComponent(targetLang)}`
        const player = new Audio(url)
        player.preload = "auto"
        player.volume = 1.0
        ttsQueueRef.current.pending.push({ player, marks })
      }
      pumpTTSQueue()
    },
    [pumpTTSQueue],
  )

  // Rotate a lane's language, skipping the slot held by the other lane
  // (the two lanes may never show the same language).
  const handleRotateLanguage = useCallback(
    (lane, direction) => {
      if (isRecording) return
      const N = AVAILABLE_LANGUAGES.length

      playBlip("language")

      if (lane === 1) {
        let ni = (lang1Index + direction + N) % N
        if (ni === lang2Index) ni = (ni + direction + N) % N
        setLang1Index(ni)
      } else {
        let ni = (lang2Index + direction + N) % N
        if (ni === lang1Index) ni = (ni + direction + N) % N
        setLang2Index(ni)
      }
    },
    [lang1Index, lang2Index, isRecording],
  )

  // Recording triggers
  const handleRecordStart = useCallback(
    async (lane) => {
      if (isRecording) return
      // Supersede whatever the previous utterance is still doing. stopSpeaking
      // alone is not enough: its stream keeps arriving, and every new sentence
      // would find an empty queue and start playing over the new recording.
      generationRef.current += 1
      if (inflightRef.current) inflightRef.current.abort()
      stopSpeaking()

      setActivePerson((prev) => {
        if (prev !== lane) playBlip("speaker")
        return lane
      })
      setActiveLaneRecording(lane)
      playBlip("ping")

      const result = await startRecording()
      if (result === false || result == null) {
        setActiveLaneRecording(null)
        return
      }
      // startRecording may return audio directly if Z was released during mic setup.
      if (result !== true && result.base64Data) {
        setActiveLaneRecording(null)
        processTranslation(lane, result.base64Data)
      }
    },
    [isRecording, stopSpeaking, startRecording],
  )

  const handleRecordStop = useCallback(async () => {
    timingRef.current = { keyup: performance.now() }
    const recordedLane = activeLaneRecording
    const audioData = await stopRecording()
    if (!audioData) return
    setActiveLaneRecording(null)
    processTranslation(recordedLane, audioData.base64Data)
  }, [activeLaneRecording, stopRecording])

  // Translation Pipeline
  const processTranslation = async (lane, base64Data) => {
    // This utterance owns the generation handleRecordStart bumped for it, and
    // its own marks object. Both are captured once: everything below asks
    // "am I still the current utterance?" rather than reading live state.
    const generation = generationRef.current
    const marks = timingRef.current
    const isCurrent = () => generationRef.current === generation
    const controller = new AbortController()
    inflightRef.current = controller

    setIsDrawerOpen(true)

    const src =
      lane === 1
        ? AVAILABLE_LANGUAGES[lang1Index]
        : AVAILABLE_LANGUAGES[lang2Index]
    const dst =
      lane === 1
        ? AVAILABLE_LANGUAGES[lang2Index]
        : AVAILABLE_LANGUAGES[lang1Index]

    setTranscriptionData({
      source: `${src.name} (Source)`,
      text: "Analyzing voice input...",
    })
    setTranslationData({
      target: `${dst.name} (Translation)`,
      text: "Translating...",
    })
    setMetaText("")

    try {
      // 1. Transcription
      setTranscriptionData((prev) => ({ ...prev, text: "Listening..." }))
      const transcribedText = await transcribeAudio(base64Data, src.code)
      if (!isCurrent()) return
      if (marks) marks.stt = performance.now()
      setTranscriptionData((prev) => ({ ...prev, text: transcribedText }))

      if (!transcribedText.trim()) {
        setTranslationData((prev) => ({
          ...prev,
          text: "(No speech detected)",
        }))
        return
      }

      // 2. Translation, streamed — speak each sentence as soon as it lands.
      // `spokenChars` indexes into whatever string was last passed here, so
      // every call must be given the same string; see the reset below.
      let spokenChars = 0
      const speakCompleteSentences = (full, isFinal) => {
        if (!isCurrent()) return
        let upto = full.length
        if (!isFinal) {
          const lastEnd = Math.max(
            full.lastIndexOf("."),
            full.lastIndexOf("!"),
            full.lastIndexOf("?"),
            full.lastIndexOf("…"),
          )
          if (lastEnd < spokenChars) return
          upto = lastEnd + 1
        }
        const ready = full.slice(spokenChars, upto).trim()
        if (!ready) return
        spokenChars = upto
        if (config.enableTts) enqueueTTS(ready, dst.ttsLang, marks)
      }

      const result = await translateTextStreaming(
        transcribedText,
        {
          ...config,
          modelName: config.modelName,
          systemPrompt: `Translate the user's text from ${src.name.split(" ")[0]} into ${dst.name.split(" ")[0]}. Reply with the translation only — no explanations, no alternatives, no quotes, no preamble.`,
        },
        (partial) => {
          if (!isCurrent()) return
          // First token, not "response complete": this is what the drawer's
          // LLM figure means, and it is the only LLM mark that exists before
          // sentence one starts playing.
          if (marks && !marks.llm) marks.llm = performance.now()
          setTranslationData((prev) => ({ ...prev, text: partial }))
          speakCompleteSentences(partial, false)
        },
        controller.signal,
      )

      if (!isCurrent()) return
      if (marks && !marks.llm) marks.llm = performance.now()
      // The streamed text and the parsed translation are only the same string
      // when the model answered in plain text. If they differ — a legacy JSON
      // envelope, which also means no partials were emitted — `spokenChars`
      // counts into a string that is not this one, so slicing with it would
      // cut at unrelated offsets.
      if (result.translation !== result.raw) spokenChars = 0
      setTranslationData((prev) => ({ ...prev, text: result.translation }))
      speakCompleteSentences(result.translation, true)
      // Tokens rapporteras inte av litert-lm (usage saknas i svaret). Med TTS
      // på sätts latensraden av reportLatency när ljudet börjar spela; utan
      // TTS spelas inget, så den skrivs här i stället för att aldrig skrivas.
      if (!config.enableTts) reportLatency(marks, false)
    } catch (err) {
      if (err.name === "AbortError" || !isCurrent()) return
      console.error(err)
      setTranscriptionData((prev) => ({
        ...prev,
        text: prev.text === "Listening..." ? "(Transcription failed)" : prev.text,
      }))
      setTranslationData((prev) => ({ ...prev, text: `Error: ${err.message}` }))
    } finally {
      if (inflightRef.current === controller) inflightRef.current = null
    }
  }

  // Push-to-talk keyboard control (two modes, see README):
  // landscape = one "active person" driven by Space/Z/arrows;
  // vertical   = independent per-lane keys (Z/X for record, arrows and -/+).
  // keydown starts recording, keyup stops — e.repeat guards auto-repeat.
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      const key = e.key.toLowerCase()

      if (config.keyboardMode === "landscape") {
        if (key === " " || e.key === "Spacebar") {
          e.preventDefault()
          if (!isRecording) {
            playBlip("speaker")
            setActivePerson((p) => (p === 1 ? 2 : 1))
          }
        } else if (key === "z") {
          e.preventDefault()
          if (!e.repeat && !isRecording) handleRecordStart(activePerson)
        } else if (e.key === "ArrowLeft") {
          e.preventDefault()
          handleRotateLanguage(activePerson, -1)
        } else if (e.key === "ArrowRight") {
          e.preventDefault()
          handleRotateLanguage(activePerson, 1)
        }
      } else {
        if (key === "z") {
          e.preventDefault()
          if (!e.repeat && !isRecording) handleRecordStart(1)
        } else if (key === "x") {
          e.preventDefault()
          if (!e.repeat && !isRecording) handleRecordStart(2)
        } else if (e.key === "ArrowLeft") {
          e.preventDefault()
          handleRotateLanguage(1, -1)
        } else if (e.key === "ArrowRight") {
          e.preventDefault()
          handleRotateLanguage(1, 1)
        } else if (key === "-" || key === "_") {
          e.preventDefault()
          handleRotateLanguage(2, -1)
        } else if (key === "+" || key === "=") {
          e.preventDefault()
          handleRotateLanguage(2, 1)
        }
      }
    }

    const handleKeyUp = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      const key = e.key.toLowerCase()

      if (config.keyboardMode === "landscape") {
        // Always attempt stop on Z release — the recorder hook no-ops if idle,
        // and marks a pending stop if getUserMedia is still opening.
        if (key === "z") handleRecordStop()
      } else {
        if (key === "z" || key === "x") handleRecordStop()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    window.addEventListener("keyup", handleKeyUp)
    return () => {
      window.removeEventListener("keydown", handleKeyDown)
      window.removeEventListener("keyup", handleKeyUp)
    }
  }, [
    config.keyboardMode,
    isRecording,
    activePerson,
    activeLaneRecording,
    handleRecordStart,
    handleRecordStop,
    handleRotateLanguage,
  ])

  return (
    <div className="translator-envelope">
      <ResponseDrawer
        isActive={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        transcriptionSource={transcriptionData.source}
        transcriptionText={transcriptionData.text}
        translationTarget={translationData.target}
        translationText={translationData.text}
        metaText={metaText}
      />

      <main className="translator-workspace">
        <div className="languages-container">
          <LanguageLane
            laneId={1}
            laneLabel="1"
            languages={AVAILABLE_LANGUAGES}
            currentIndex={lang1Index}
            isRecording={activeLaneRecording === 1}
            isActivePerson={
              config.keyboardMode === "landscape" && activePerson === 1
            }
            recordKeyHint={
              config.keyboardMode === "vertical"
                ? "Z"
                : activePerson === 1
                  ? "Z"
                  : null
            }
            onRotate={(dir) => handleRotateLanguage(1, dir)}
          />
          <LanguageLane
            laneId={2}
            laneLabel="2"
            languages={AVAILABLE_LANGUAGES}
            currentIndex={lang2Index}
            isRecording={activeLaneRecording === 2}
            isActivePerson={
              config.keyboardMode === "landscape" && activePerson === 2
            }
            recordKeyHint={
              config.keyboardMode === "vertical"
                ? "X"
                : activePerson === 2
                  ? "Z"
                  : null
            }
            onRotate={(dir) => handleRotateLanguage(2, dir)}
          />
        </div>

        <div className="keyboard-legend" aria-live="polite">
          {config.keyboardMode === "landscape" ? (
            <>
              <span>
                <kbd>SPC</kbd> person
              </span>
              <span>
                <kbd>Z</kbd> hold talk
              </span>
              <span>
                <kbd>←→</kbd> language
              </span>
            </>
          ) : (
            <>
              <span>
                <kbd>Z</kbd>/<kbd>X</kbd> talk
              </span>
              <span>
                <kbd>←→</kbd> lang 1
              </span>
              <span>
                <kbd>−+</kbd> lang 2
              </span>
            </>
          )}
        </div>

        <Visualizer
          activePerson={activePerson}
          isRecording={isRecording}
          analyser={analyser}
          barsCount={parseInt(config.visualizerBars, 10)}
        />
      </main>
    </div>
  )
}

export default TranslatorApp
