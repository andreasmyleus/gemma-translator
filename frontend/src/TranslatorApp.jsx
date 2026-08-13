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

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import LanguageLane from "./components/LanguageLane"
import TranscriptView from "./components/TranscriptView"
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
// → LLM translation via /proxy (Gemma) → /api/tts playback.
// Listening / translating / text all live in the side-by-side transcript.

// Languages offered on each lane's revolver; ttsLang selects the backend voice.
const AVAILABLE_LANGUAGES = [
  { code: "sv", name: "Swedish", voice: "tts", ttsLang: "sv" },
  { code: "en", name: "English", voice: "tts", ttsLang: "en" },
  { code: "fi", name: "Finnish", voice: "tts", ttsLang: "fi" },
  { code: "es", name: "Spanish", voice: "tts", ttsLang: "es" },
  { code: "fr", name: "French", voice: "tts", ttsLang: "fr" },
]

function TranslatorApp({ config, clearConversationRef }) {
  // UI State
  const [activePerson, setActivePerson] = useState(1)

  // Conversation State: append-only turns. Column headers track the current
  // lane languages so a mid-conversation rotate still places text correctly.
  const [turns, setTurns] = useState([])
  const nextTurnId = useRef(1)
  // Turn created on keydown (listening); processTranslation fills it in.
  const activeTurnIdRef = useRef(null)
  // Lang pair captured at keydown for the active utterance — never re-read live
  // lane indices after await (a rotate mid-pipeline must not retarget STT/TTS).
  const activeUtteranceRef = useRef(null)

  // Currently-playing TTS audio element (chunked playback chain)
  const onlineAudioPlayerRef = useRef(null)

  // Timestamps for latency measurement; reset on every key release.
  const timingRef = useRef(null)

  // Language Lanes State
  const [lang1Index, setLang1Index] = useState(0)
  const [lang2Index, setLang2Index] = useState(1)
  const [activeLaneRecording, setActiveLaneRecording] = useState(null) // 1 or 2

  const columns = useMemo(
    () => ({
      left: {
        code: AVAILABLE_LANGUAGES[lang1Index].code,
        name: AVAILABLE_LANGUAGES[lang1Index].name,
      },
      right: {
        code: AVAILABLE_LANGUAGES[lang2Index].code,
        name: AVAILABLE_LANGUAGES[lang2Index].name,
      },
    }),
    [lang1Index, lang2Index],
  )

  const { isRecording, startRecording, stopRecording, analyser, micError } =
    useAudioRecorder()

  // Sentences waiting to be spoken. Each entry is { player, marks }: the Audio
  // element is already constructed (creating it starts the fetch, so queued
  // chunks download while the current one plays), and `marks` is the timing
  // object that was current when the chunk was queued.
  //
  // `sealed` is what lets an asynchronously-drained queue still honour the
  // onFinished(played) contract: mid-stream an empty queue only means
  // the next sentence hasn't been generated yet, so "finished" is *sealed and
  // drained with nothing playing*. `onFinished` fires exactly once per session.
  const ttsQueueRef = useRef({
    pending: [],
    playing: false,
    sealed: false,
    played: false,
    onFinished: null,
  })

  // Bumped whenever a new recording starts. Everything a translation does
  // afterwards — speaking, writing the panels — is guarded on still owning the
  // current value, so an interrupted translation cannot talk over the next one.
  const generationRef = useRef(0)
  // The in-flight streamed translation, so it can be aborted when superseded.
  const inflightRef = useRef(null)

  // Patch a single turn in place, so a running pipeline never disturbs the
  // turns around it.
  const updateTurn = useCallback((id, patch) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)))
  }, [])

  // Report one utterance's latency. `marks` is passed in rather than read from
  // timingRef: re-reading the ref here would let a late-firing chunk from a
  // previous utterance stamp `logged` on the *next* utterance's marks, silently
  // suppressing its measurement (see commit 08c6df0). `marks.turnId` carries
  // the same binding into the transcript, so the line lands on its own turn.
  const reportLatency = useCallback(
    (marks, audioStarted) => {
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
      const line = `STT ${since(marks.stt)} · LLM ${since(marks.llm)} · ljud ${audio}`
      if (marks.turnId) updateTurn(marks.turnId, { meta: line })
    },
    [updateTurn],
  )

  // Hand the session its one and only verdict for the current playback.
  const settleTTS = useCallback((played) => {
    const queue = ttsQueueRef.current
    const onFinished = queue.onFinished
    queue.onFinished = null
    onFinished?.(played)
  }, [])

  const stopSpeaking = useCallback(() => {
    const queue = ttsQueueRef.current
    queue.pending = []
    queue.playing = false
    queue.sealed = false
    if (onlineAudioPlayerRef.current) {
      onlineAudioPlayerRef.current.pause()
      onlineAudioPlayerRef.current = null
    }
    // Nothing will ever drain now — a superseded or torn-down session reports
    // played=false rather than leaving onFinished hanging forever.
    settleTTS(false)
  }, [settleTTS])

  const pumpTTSQueue = useCallback(() => {
    const queue = ttsQueueRef.current
    if (queue.playing) return
    const entry = queue.pending.shift()
    if (!entry) {
      // Drained. Only a *sealed* queue is finished: before the stream ends an
      // empty queue just means the next sentence isn't generated yet.
      if (queue.sealed) settleTTS(queue.played)
      return
    }
    const { player, marks } = entry
    queue.playing = true
    onlineAudioPlayerRef.current = player
    player.onplaying = () => reportLatency(marks, true)
    player.onended = () => {
      queue.playing = false
      // Reaching the end of a chunk is the only way audio is known to have been
      // heard: every failure path routes through stopSpeaking instead.
      queue.played = true
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
  }, [reportLatency, settleTTS, stopSpeaking])

  // Open a playback session for one utterance. onFinished(played) fires exactly
  // once: true when every queued sentence has been spoken, false on an error,
  // on nothing to speak, or when a newer recording supersedes this one.
  const beginTTSSession = useCallback(
    (onFinished) => {
      settleTTS(false)
      const queue = ttsQueueRef.current
      queue.pending = []
      queue.playing = false
      queue.sealed = false
      queue.played = false
      queue.onFinished = onFinished
    },
    [settleTTS],
  )

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

  // No more sentences are coming. If the queue is already empty this settles
  // the session immediately; otherwise the last onended does.
  const sealTTSQueue = useCallback(() => {
    ttsQueueRef.current.sealed = true
    pumpTTSQueue()
  }, [pumpTTSQueue])

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

  // Mark a turn cancelled if it is still waiting on STT/LLM — used when a
  // newer recording supersedes it, or when processTranslation early-returns
  // after !isCurrent() so the row never stays on "listening"/"translating".
  const cancelTurnIfPending = useCallback(
    (id) => {
      if (id == null) return
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id &&
          (t.status === "transcribing" || t.status === "translating")
            ? { ...t, status: "cancelled", error: null }
            : t,
        ),
      )
    },
    [],
  )

  // Drop a turn that never left listening/transcribing (empty capture, clear
  // during getUserMedia, superseded before processTranslation starts).
  const abandonActiveTurn = useCallback(
    (turnId) => {
      cancelTurnIfPending(turnId)
      setActiveLaneRecording(null)
      if (activeTurnIdRef.current === turnId) activeTurnIdRef.current = null
      if (activeUtteranceRef.current?.turnId === turnId) {
        activeUtteranceRef.current = null
      }
    },
    [cancelTurnIfPending],
  )

  // Recording triggers
  const handleRecordStart = useCallback(
    async (lane) => {
      if (isRecording) return
      // Supersede whatever the previous utterance is still doing. stopSpeaking
      // alone is not enough: its stream keeps arriving, and every new sentence
      // would find an empty queue and start playing over the new recording.
      generationRef.current += 1
      const generation = generationRef.current
      if (inflightRef.current) inflightRef.current.abort()
      stopSpeaking()
      // isRecording is still false while getUserMedia is in flight, so a second
      // key (Z→X) can enter here. stopRecording sets stopRequestedRef so the
      // in-flight start aborts instead of leaving the mic live orphaned.
      void stopRecording()

      setActivePerson((prev) => {
        if (prev !== lane) playBlip("speaker")
        return lane
      })
      setActiveLaneRecording(lane)
      playBlip("ping")

      const src =
        lane === 1
          ? AVAILABLE_LANGUAGES[lang1Index]
          : AVAILABLE_LANGUAGES[lang2Index]
      const dst =
        lane === 1
          ? AVAILABLE_LANGUAGES[lang2Index]
          : AVAILABLE_LANGUAGES[lang1Index]

      // Open the turn immediately so the transcript shows "listening" while
      // the key is held — not only after release. Cancel any still-pending
      // previous rows in the same update so they never stick on listening.
      const turnId = nextTurnId.current++
      activeTurnIdRef.current = turnId
      // Capture generation with the lang pair so clear / a newer hold can
      // invalidate this utterance while startRecording is still awaiting.
      activeUtteranceRef.current = { turnId, lane, src, dst, generation }
      setTurns((prev) => [
        ...prev.map((t) =>
          t.status === "transcribing" || t.status === "translating"
            ? { ...t, status: "cancelled", error: null }
            : t,
        ),
        {
          id: turnId,
          lane,
          sourceLang: src.code,
          targetLang: dst.code,
          sourceText: "",
          targetText: "",
          status: "transcribing",
          error: null,
          meta: "",
        },
      ])

      const result = await startRecording()
      if (generationRef.current !== generation) {
        // Another hold / clear bumped generation while getUserMedia ran — tear
        // down the capture the same way clear does, or the mic stays live with
        // activeUtteranceRef already cleared.
        abandonActiveTurn(turnId)
        void stopRecording()
        return
      }
      if (result === false || result == null) {
        // Early key release during getUserMedia, mic failure, or a superseding
        // start that hit startingRef — do not invent "Microphone failed". Real
        // mic errors still surface via the micError banner from the hook.
        // Always stopRecording: a Z→X race can leave the first capture live
        // after this start failed on startingRef.
        abandonActiveTurn(turnId)
        void stopRecording()
        return
      }
      // startRecording may return audio directly if Z was released during mic setup.
      if (result !== true && result.base64Data) {
        setActiveLaneRecording(null)
        processTranslation(lane, result.base64Data, turnId, src, dst, generation)
      }
    },
    [
      isRecording,
      stopSpeaking,
      startRecording,
      stopRecording,
      lang1Index,
      lang2Index,
      abandonActiveTurn,
    ],
  )

  const handleRecordStop = useCallback(async () => {
    timingRef.current = { keyup: performance.now() }
    const recordedLane = activeLaneRecording
    const turnId = activeTurnIdRef.current
    const utterance = activeUtteranceRef.current
    const generation = utterance?.generation
    const src = utterance?.turnId === turnId ? utterance.src : null
    const dst = utterance?.turnId === turnId ? utterance.dst : null
    const audioData = await stopRecording()
    // Missing keydown binding (superseded / abandoned mid-start) or a newer
    // generation — do not start STT, and never claim generationRef.current.
    if (
      generation == null ||
      turnId == null ||
      !src ||
      !dst ||
      generationRef.current !== generation
    ) {
      abandonActiveTurn(turnId)
      return
    }
    if (!audioData) {
      // Empty capture after the graph was up — cancel rather than leave the
      // keydown turn stuck on "listening".
      abandonActiveTurn(turnId)
      return
    }
    setActiveLaneRecording(null)
    processTranslation(
      recordedLane,
      audioData.base64Data,
      turnId,
      src,
      dst,
      generation,
    )
  }, [activeLaneRecording, stopRecording, abandonActiveTurn])

  // Translation Pipeline. `turnId` / `src` / `dst` / `generation` must come
  // from the keydown utterance — never invent a turn or fall back to live
  // lane indices / generationRef (a post-supersede keyup used to claim the
  // new generation and run STT/TTS with no active utterance).
  const processTranslation = async (
    lane,
    base64Data,
    turnId,
    src,
    dst,
    generation,
  ) => {
    if (
      generation == null ||
      turnId == null ||
      !src ||
      !dst ||
      generationRef.current !== generation
    ) {
      cancelTurnIfPending(turnId)
      return
    }
    const marks = timingRef.current
    const isCurrent = () => generationRef.current === generation
    const controller = new AbortController()
    inflightRef.current = controller

    // Bind the marks to this turn so the latency line lands on the turn that
    // produced it, even if a later chunk is what finally reports it.
    if (marks) marks.turnId = turnId
    if (activeTurnIdRef.current === turnId) activeTurnIdRef.current = null

    const dropIfSuperseded = () => {
      cancelTurnIfPending(turnId)
    }

    try {
      // 1. Transcription
      const transcribedText = await transcribeAudio(base64Data, src.code)
      if (!isCurrent()) {
        dropIfSuperseded()
        return
      }
      if (marks) marks.stt = performance.now()

      if (!transcribedText.trim()) {
        updateTurn(turnId, { status: "empty" })
        return
      }

      updateTurn(turnId, {
        sourceText: transcribedText,
        status: "translating",
      })

      // Playback settles once; we only need it for latency reporting now that
      // the transcript is the sole UI surface.
      if (config.enableTts) {
        beginTTSSession((played) => {
          if (!isCurrent()) return
          if (!played) reportLatency(marks, false)
        })
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
          // First token, not "response complete": this is what the latency
          // LLM figure means, and it is the only LLM mark that exists before
          // sentence one starts playing.
          if (marks && !marks.llm) marks.llm = performance.now()
          updateTurn(turnId, { targetText: partial })
          speakCompleteSentences(partial, false)
        },
        controller.signal,
      )

      if (!isCurrent()) {
        dropIfSuperseded()
        return
      }
      if (marks && !marks.llm) marks.llm = performance.now()
      // The streamed text and the parsed translation are only the same string
      // when the model answered in plain text. If they differ — a legacy JSON
      // envelope, which also means no partials were emitted — `spokenChars`
      // counts into a string that is not this one, so slicing with it would
      // cut at unrelated offsets.
      if (result.translation !== result.raw) spokenChars = 0
      updateTurn(turnId, { targetText: result.translation, status: "done" })
      speakCompleteSentences(result.translation, true)
      // Tokens rapporteras inte av litert-lm (usage saknas i svaret). Med TTS
      // på sätts latensraden av reportLatency när ljudet börjar spela; utan
      // TTS spelas inget, så den skrivs här i stället för att aldrig skrivas.
      if (!config.enableTts) reportLatency(marks, false)
      if (config.enableTts) sealTTSQueue()
    } catch (err) {
      if (err.name === "AbortError" || !isCurrent()) {
        dropIfSuperseded()
        return
      }
      console.error(err)
      updateTurn(turnId, { status: "error", error: err.message })
      sealTTSQueue()
    } finally {
      if (inflightRef.current === controller) inflightRef.current = null
    }
  }

  // Wipe the conversation and stop anything still in flight so a late STT/LLM
  // completion cannot enqueue TTS after clear.
  const handleClearConversation = useCallback(() => {
    generationRef.current += 1
    if (inflightRef.current) inflightRef.current.abort()
    inflightRef.current = null
    activeTurnIdRef.current = null
    activeUtteranceRef.current = null
    stopSpeaking()
    // End a pending getUserMedia / active capture so handleRecordStart's await
    // and handleRecordStop do not proceed into STT after the wipe.
    void stopRecording()
    setTurns([])
  }, [stopSpeaking, stopRecording])

  // The settings overlay lives in App.jsx but the conversation lives here, so
  // App borrows the handler through a ref rather than us lifting the state up.
  useEffect(() => {
    if (!clearConversationRef) return
    clearConversationRef.current = handleClearConversation
    return () => {
      clearConversationRef.current = null
    }
  }, [clearConversationRef, handleClearConversation])

  // Push-to-talk keyboard control (two modes, see README):
  // landscape = one "active person" driven by Enter/Space/arrows;
  // vertical   = independent per-lane keys (Z/X for record, arrows and -/+).
  // keydown starts recording, keyup stops — e.repeat guards auto-repeat.
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      const key = e.key.toLowerCase()

      if (config.keyboardMode === "landscape") {
        if (e.key === "Enter") {
          e.preventDefault()
          if (!isRecording) {
            playBlip("speaker")
            setActivePerson((p) => (p === 1 ? 2 : 1))
          }
        } else if (key === " " || e.key === "Spacebar") {
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
        // Always attempt stop on Space release — the recorder hook no-ops if
        // idle, and marks a pending stop if getUserMedia is still opening.
        if (key === " " || e.key === "Spacebar") handleRecordStop()
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
      <TranscriptView turns={turns} columns={columns} micError={micError} />

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
                  ? "SPC"
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
                  ? "SPC"
                  : null
            }
            onRotate={(dir) => handleRotateLanguage(2, dir)}
          />
        </div>

        <div className="keyboard-legend" aria-live="polite">
          {config.keyboardMode === "landscape" ? (
            <>
              <span>
                <kbd>⏎</kbd> person
              </span>
              <span>
                <kbd>SPC</kbd> hold talk
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
