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
import TranscriptView from "./components/TranscriptView"
import Visualizer from "./components/Visualizer"
import { useAudioRecorder } from "./hooks/useAudioRecorder"
import {
  transcribeAudio,
  translateText,
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

const languageName = (code) =>
  AVAILABLE_LANGUAGES.find((l) => l.code === code)?.name || ""

// How long the drawer lingers when TTS can't tell us playback has ended.
const DRAWER_FALLBACK_MS = 4000

// Drawer copy for the newest turn: placeholders stand in for the stages that
// haven't landed yet.
const drawerSourceText = (turn) => {
  if (!turn) return ""
  if (turn.status === "transcribing") return "Listening..."
  if (turn.status === "error" && !turn.sourceText) return "(Transcription failed)"
  return turn.sourceText
}

const drawerTargetText = (turn) => {
  if (!turn) return ""
  if (turn.status === "error") return `Error: ${turn.error}`
  if (turn.status === "empty") return "(No speech detected)"
  return turn.targetText || "Translating..."
}

function TranslatorApp({ config, clearConversationRef }) {
  // UI State
  const [isDrawerOpen, setIsDrawerOpen] = useState(false)
  const [activePerson, setActivePerson] = useState(1)

  // Conversation State: append-only turns plus the language pair the columns
  // locked to when the conversation started.
  const [turns, setTurns] = useState([])
  const [columns, setColumns] = useState(null)
  const nextTurnId = useRef(1)

  // Currently-playing TTS audio element (chunked playback chain)
  const onlineAudioPlayerRef = useRef(null)
  // Pending drawer auto-dismiss timer (fallback when TTS can't signal the end)
  const dismissTimerRef = useRef(null)

  // Language Lanes State
  const [lang1Index, setLang1Index] = useState(0)
  const [lang2Index, setLang2Index] = useState(1)
  const [activeLaneRecording, setActiveLaneRecording] = useState(null) // 1 or 2

  const { isRecording, startRecording, stopRecording, analyser, micError } =
    useAudioRecorder()

  // A mic failure takes over the drawer; it isn't a turn.
  useEffect(() => {
    if (micError) setIsDrawerOpen(true)
  }, [micError])

  const clearDismissTimer = useCallback(() => {
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current)
      dismissTimerRef.current = null
    }
  }, [])

  useEffect(() => clearDismissTimer, [clearDismissTimer])

  const closeDrawer = useCallback(() => {
    clearDismissTimer()
    setIsDrawerOpen(false)
  }, [clearDismissTimer])

  // Fallback dismissal for when no TTS playback marks the end of the exchange.
  const closeDrawerAfterDelay = useCallback(() => {
    clearDismissTimer()
    dismissTimerRef.current = setTimeout(() => {
      dismissTimerRef.current = null
      setIsDrawerOpen(false)
    }, DRAWER_FALLBACK_MS)
  }, [clearDismissTimer])

  const stopSpeaking = useCallback(() => {
    if (onlineAudioPlayerRef.current) {
      onlineAudioPlayerRef.current.pause()
      onlineAudioPlayerRef.current = null
    }
  }, [])

  // Speak text via /api/tts, splitting into ~180-char chunks and chaining
  // playback so long translations don't overflow a single TTS request.
  // onFinished(played) fires exactly once, so callers can react to the end of
  // playback without double-handling an error that also rejects play().
  const playTTS = useCallback(
    (text, targetLang, onFinished) => {
      let settled = false
      const finish = (played) => {
        if (settled) return
        settled = true
        onFinished?.(played)
      }

      if (!text) return finish(false)
      stopSpeaking()

      const chunks = splitTextIntoSpeechChunks(text)
      if (chunks.length === 0) return finish(false)

      let chunkIndex = 0

      const playNextChunk = () => {
        if (chunkIndex >= chunks.length) {
          stopSpeaking()
          finish(true)
          return
        }
        const ttsUrl = `/api/tts?text=${encodeURIComponent(chunks[chunkIndex])}&lang=${encodeURIComponent(targetLang)}`
        const player = new Audio(ttsUrl)
        player.volume = 1.0
        onlineAudioPlayerRef.current = player

        player.onended = () => {
          chunkIndex++
          playNextChunk()
        }
        player.onerror = () => {
          stopSpeaking()
          finish(false)
          alert("TTS playback failed. Backend server may be offline.")
        }
        player.play().catch((e) => {
          console.error("Audio play error:", e)
          stopSpeaking()
          finish(false)
        })
      }

      playNextChunk()
    },
    [stopSpeaking],
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
      stopSpeaking()
      // Drop any dismissal left over from the previous exchange so timers
      // from consecutive recordings can't stack up and close the new one.
      clearDismissTimer()

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
    [isRecording, stopSpeaking, startRecording, clearDismissTimer],
  )

  const handleRecordStop = useCallback(async () => {
    const recordedLane = activeLaneRecording
    const audioData = await stopRecording()
    if (!audioData) return
    setActiveLaneRecording(null)
    processTranslation(recordedLane, audioData.base64Data)
  }, [activeLaneRecording, stopRecording])

  // Patch a single turn in place, so a running pipeline never disturbs the
  // turns around it.
  const updateTurn = useCallback((id, patch) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)))
  }, [])

  // Translation Pipeline
  const processTranslation = async (lane, base64Data) => {
    setIsDrawerOpen(true)
    clearDismissTimer()

    const src =
      lane === 1
        ? AVAILABLE_LANGUAGES[lang1Index]
        : AVAILABLE_LANGUAGES[lang2Index]
    const dst =
      lane === 1
        ? AVAILABLE_LANGUAGES[lang2Index]
        : AVAILABLE_LANGUAGES[lang1Index]

    const turnId = nextTurnId.current++
    setTurns((prev) => [
      ...prev,
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

    // The columns lock to the pair in play at the first turn and never move.
    setColumns((prev) =>
      prev
        ? prev
        : {
            left: {
              code: AVAILABLE_LANGUAGES[lang1Index].code,
              name: AVAILABLE_LANGUAGES[lang1Index].name,
            },
            right: {
              code: AVAILABLE_LANGUAGES[lang2Index].code,
              name: AVAILABLE_LANGUAGES[lang2Index].name,
            },
          },
    )

    try {
      // 1. Transcription
      const transcribedText = await transcribeAudio(base64Data, src.code)

      if (!transcribedText.trim()) {
        updateTurn(turnId, { status: "empty" })
        closeDrawerAfterDelay()
        return
      }

      updateTurn(turnId, {
        sourceText: transcribedText,
        status: "translating",
      })

      // 2. Translation
      const result = await translateText(transcribedText, {
        ...config,
        modelName: config.modelName,
        systemPrompt: `You are a high-performance translator. Your task is to translate text from ${src.name.split(" ")[0]} into ${dst.name.split(" ")[0]}.\nYou MUST format your response as a valid JSON object matching this structure:\n{\n  "translation": "High-quality, natural translation into ${dst.name.split(" ")[0]}"\n}\nDo NOT return anything else except this JSON object. No Markdown block wraps (no \`\`\`json), no introductory text, no conversational text. Start directly with "{" and end directly with "}".`,
      })

      updateTurn(turnId, {
        targetText: result.translation,
        meta: `Duration: ${result.duration}s | Tokens: ${result.tokens}`,
        status: "done",
      })

      // The drawer hands over to the transcript once the translation has been
      // spoken; without playback a timer stands in for that beat.
      if (config.enableTts) {
        playTTS(result.translation, dst.ttsLang, (played) => {
          if (played) closeDrawer()
          else closeDrawerAfterDelay()
        })
      } else {
        closeDrawerAfterDelay()
      }
    } catch (err) {
      console.error(err)
      updateTurn(turnId, { status: "error", error: err.message })
      closeDrawerAfterDelay()
    }
  }

  // Wipe the conversation; clearing columns lets a fresh language pair lock.
  const handleClearConversation = useCallback(() => {
    stopSpeaking()
    setTurns([])
    setColumns(null)
    closeDrawer()
  }, [stopSpeaking, closeDrawer])

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

  // The drawer is a view of the newest turn, not state of its own.
  const latestTurn = turns[turns.length - 1] ?? null

  const drawerProps = micError
    ? {
        transcriptionSource: "Microphone",
        transcriptionText: "Access Failed",
        translationTarget: "Error",
        translationText: `${micError} (HTTPS is required when accessing from remote devices)`,
        metaText: "",
      }
    : {
        transcriptionSource: latestTurn
          ? `${languageName(latestTurn.sourceLang)} (Source)`
          : "",
        transcriptionText: drawerSourceText(latestTurn),
        translationTarget: latestTurn
          ? `${languageName(latestTurn.targetLang)} (Translation)`
          : "",
        translationText: drawerTargetText(latestTurn),
        metaText: latestTurn?.meta ?? "",
      }

  return (
    <div className="translator-envelope">
      {/* Transcript and drawer share the envelope's first grid row; the drawer
          overlays the transcript rather than displacing it. */}
      <TranscriptView turns={turns} columns={columns} />

      <ResponseDrawer
        isActive={isDrawerOpen}
        onClose={closeDrawer}
        {...drawerProps}
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
