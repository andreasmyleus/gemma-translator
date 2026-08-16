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
import { useVoiceActivity } from "./hooks/useVoiceActivity"
import {
  transcribeAudio,
  translateTextStreaming,
  splitTextIntoSpeechChunks,
  isSpeakable,
  isBackchannel,
  endsWithContinuationCue,
  isRepairUtterance,
  stripRepairCue,
  normalizeSttText,
  routeSpokenTurn,
  CONTINUE_WINDOW_MS,
  REPAIR_WINDOW_MS,
  CONTINUATION_HOLD_MS,
} from "./utils/api"
import { concatSamples, samplesToBase64Pcm } from "./utils/audioHelpers"
import { playBlip } from "./utils/audio-blip"

// Core orchestrator for the two-person kiosk translator.
// Flow: continuous mic + energy VAD → POST /api/stt (Whisper) → LLM via
// /proxy (Gemma) → /api/tts playback. Enter switches the active person;
// listening / translating / text all live in the side-by-side transcript.

// Languages offered on each lane's revolver; ttsLang selects the backend voice.
const AVAILABLE_LANGUAGES = [
  { code: "sv", name: "Swedish", voice: "tts", ttsLang: "sv" },
  { code: "en", name: "English", voice: "tts", ttsLang: "en" },
  { code: "fi", name: "Finnish", voice: "tts", ttsLang: "fi" },
  { code: "es", name: "Spanish", voice: "tts", ttsLang: "es" },
  { code: "fr", name: "French", voice: "tts", ttsLang: "fr" },
]

function langsForLane(lane, lang1Index, lang2Index) {
  const src =
    lane === 1 ? AVAILABLE_LANGUAGES[lang1Index] : AVAILABLE_LANGUAGES[lang2Index]
  const dst =
    lane === 1 ? AVAILABLE_LANGUAGES[lang2Index] : AVAILABLE_LANGUAGES[lang1Index]
  return { src, dst }
}

function prewarmLang(code) {
  if (!code) return
  void fetch(`/api/prewarm?lang=${encodeURIComponent(code)}`).catch(() => {})
}

function TranslatorApp({ config, clearConversationRef }) {
  // UI State
  const [activePerson, setActivePerson] = useState(1)
  const [lang1Index, setLang1Index] = useState(0)
  const [lang2Index, setLang2Index] = useState(1)
  const [activeLaneRecording, setActiveLaneRecording] = useState(null) // 1 or 2

  // Conversation State: append-only turns. Column headers track the current
  // lane languages so a mid-conversation rotate still places text correctly.
  const [turns, setTurns] = useState([])
  const nextTurnId = useRef(1)
  // Turn opened when VAD hears speech; processTranslation fills it in.
  const activeTurnIdRef = useRef(null)
  // Lang pair captured at speech-start — never re-read live lane indices after
  // await (a rotate mid-pipeline must not retarget STT/TTS).
  const activeUtteranceRef = useRef(null)
  // Live mirrors for VAD callbacks (stable handlers read these, not stale clos.).
  const activePersonRef = useRef(1)
  const lang1IndexRef = useRef(0)
  const lang2IndexRef = useRef(1)
  const configRef = useRef(config)
  activePersonRef.current = activePerson
  lang1IndexRef.current = lang1Index
  lang2IndexRef.current = lang2Index
  configRef.current = config

  // Currently-playing TTS: Web Audio BufferSource (so we can feed AEC).
  const onlineAudioPlayerRef = useRef(null)
  // Controllers for in-flight LLM streams, keyed by turn id — a new utterance
  // must not abort an earlier turn that is still translating.
  const inflightByTurnRef = useRef(new Map())

  // Timestamps for latency measurement are owned per utterance (see endUtterance).

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

  // Epoch bumped only on clear — overlapping speech and Enter must not kill
  // a turn that is still being transcribed or spoken.
  const generationRef = useRef(0)

  // Patch a single turn in place, so a running pipeline never disturbs the
  // turns around it.
  const updateTurn = useCallback((id, patch) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch } : t)))
  }, [])

  // Report one utterance's latency. `marks` is passed in rather than shared
  // globally so overlapping turns do not stamp each other's timings.
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

  // Filled once the VAD hook mounts — TTS playback needs the shared context.
  const getAudioContextRef = useRef(() => null)
  const setTtsPlaybackRef = useRef(() => {})
  const clearTtsPlaybackRef = useRef(() => {})
  const setTtsDuckRef = useRef(() => {})
  const getTtsGainNodeRef = useRef(() => null)
  const setSilenceHoldRef = useRef(() => {})
  // Serialize TTS sessions when overlapping turns both want to speak.
  const ttsWaitQueueRef = useRef([])
  const ttsSessionActiveRef = useRef(false)
  // Last turn that may still accept a continuation or repair.
  const openTurnRef = useRef(null)
  const previousTurnRef = useRef(null)
  const continuationTimerRef = useRef(null)
  const pendingTranslateRef = useRef(null)
  const enqueueChainRef = useRef(Promise.resolve())
  const enqueueGenRef = useRef(0)
  const sttInflightRef = useRef(0)
  const interimAbortRef = useRef(null)

  const stopSpeaking = useCallback(() => {
    const queue = ttsQueueRef.current
    queue.pending = []
    queue.playing = false
    queue.sealed = false
    if (onlineAudioPlayerRef.current) {
      try {
        onlineAudioPlayerRef.current.stop()
      } catch (_) {
        /* already stopped */
      }
      onlineAudioPlayerRef.current = null
    }
    clearTtsPlaybackRef.current?.()
    ttsWaitQueueRef.current = []
    ttsSessionActiveRef.current = false
    enqueueGenRef.current += 1
    enqueueChainRef.current = Promise.resolve()
    settleTTS(false)
  }, [settleTTS])

  const pumpTTSQueue = useCallback(() => {
    const queue = ttsQueueRef.current
    if (queue.playing) return
    const entry = queue.pending.shift()
    if (!entry) {
      if (queue.sealed) settleTTS(queue.played)
      return
    }
    const { buffer, marks } = entry
    const ctx = getAudioContextRef.current()
    if (!ctx || !buffer) {
      queue.playing = false
      stopSpeaking()
      return
    }
    queue.playing = true
    const source = ctx.createBufferSource()
    source.buffer = buffer
    const gain = getTtsGainNodeRef.current?.()
    source.connect(gain || ctx.destination)
    onlineAudioPlayerRef.current = source

    const data =
      buffer.numberOfChannels > 0
        ? buffer.getChannelData(0)
        : new Float32Array(0)
    const startedAt = ctx.currentTime + 0.02
    setTtsPlaybackRef.current(data, buffer.sampleRate, startedAt)
    source.onended = () => {
      queue.playing = false
      queue.played = true
      clearTtsPlaybackRef.current?.()
      onlineAudioPlayerRef.current = null
      pumpTTSQueue()
    }
    try {
      source.start(startedAt)
      reportLatency(marks, true)
    } catch (e) {
      console.error("Audio play error:", e)
      queue.playing = false
      stopSpeaking()
    }
  }, [reportLatency, settleTTS, stopSpeaking])

  const beginTTSSession = useCallback(
    (onFinished, meta) => {
      const start = () => {
        const queue = ttsQueueRef.current
        settleTTS(false)
        queue.pending = []
        queue.playing = false
        queue.sealed = false
        queue.played = false
        queue.turnId = meta?.turnId ?? null
        queue.lane = meta?.lane ?? null
        ttsSessionActiveRef.current = true
        queue.onFinished = (played) => {
          ttsSessionActiveRef.current = false
          onFinished?.(played)
          const next = ttsWaitQueueRef.current.shift()
          next?.()
        }
      }
      if (
        ttsSessionActiveRef.current ||
        ttsQueueRef.current.playing ||
        ttsQueueRef.current.pending.length > 0
      ) {
        ttsWaitQueueRef.current.push(start)
      } else {
        start()
      }
    },
    [settleTTS],
  )

  const enqueueTTS = useCallback(
    async (text, targetLang, marks) => {
      if (!text) return
      const gen = enqueueGenRef.current
      const run = async () => {
        if (enqueueGenRef.current !== gen) return
        const ctx = getAudioContextRef.current()
        if (!ctx) return
        for (const chunk of splitTextIntoSpeechChunks(text)) {
          if (enqueueGenRef.current !== gen) return
          const url = `/api/tts?text=${encodeURIComponent(chunk)}&lang=${encodeURIComponent(targetLang)}`
          try {
            const res = await fetch(url)
            if (!res.ok) throw new Error(`TTS HTTP ${res.status}`)
            const raw = await res.arrayBuffer()
            const buffer = await ctx.decodeAudioData(raw.slice(0))
            if (enqueueGenRef.current !== gen) return
            ttsQueueRef.current.pending.push({ buffer, marks })
            pumpTTSQueue()
          } catch (e) {
            console.error("TTS fetch/play failed:", e)
            stopSpeaking()
            return
          }
        }
      }
      enqueueChainRef.current = enqueueChainRef.current.then(run, run)
      return enqueueChainRef.current
    },
    [pumpTTSQueue, stopSpeaking],
  )

  const sealTTSQueue = useCallback(() => {
    ttsQueueRef.current.sealed = true
    pumpTTSQueue()
  }, [pumpTTSQueue])

  // Rotate a lane's language, skipping the slot held by the other lane
  // (the two lanes may never show the same language).
  const handleRotateLanguage = useCallback(
    (lane, direction) => {
      if (activeLaneRecording != null) return
      const N = AVAILABLE_LANGUAGES.length

      playBlip("language")

      if (lane === 1) {
        let ni = (lang1Index + direction + N) % N
        if (ni === lang2Index) ni = (ni + direction + N) % N
        setLang1Index(ni)
        prewarmLang(AVAILABLE_LANGUAGES[ni].code)
      } else {
        let ni = (lang2Index + direction + N) % N
        if (ni === lang1Index) ni = (ni + direction + N) % N
        setLang2Index(ni)
        prewarmLang(AVAILABLE_LANGUAGES[ni].code)
      }
    },
    [lang1Index, lang2Index, activeLaneRecording],
  )

  // Mark a turn cancelled if it is still waiting on STT/LLM. Empty VAD
  // false-starts are removed instead of leaving a dash in the transcript.
  const cancelTurnIfPending = useCallback((id) => {
    if (id == null) return
    setTurns((prev) =>
      prev.flatMap((t) => {
        if (t.id !== id) return [t]
        if (t.status !== "transcribing" && t.status !== "translating") return [t]
        if (!isSpeakable(t.sourceText) && !isSpeakable(t.targetText)) return []
        return [{ ...t, status: "cancelled", error: null }]
      }),
    )
  }, [])

  // Drop a turn that never left listening/transcribing (empty capture, clear
  // during getUserMedia, superseded before processTranslation starts).
  const dropTurn = useCallback((turnId) => {
    if (turnId == null) return
    setTurns((prev) => prev.filter((t) => t.id !== turnId))
    if (activeTurnIdRef.current === turnId) activeTurnIdRef.current = null
    if (activeUtteranceRef.current?.turnId === turnId) {
      activeUtteranceRef.current = null
    }
    if (openTurnRef.current?.turnId === turnId) openTurnRef.current = null
  }, [])

  const abandonActiveTurn = useCallback(
    (turnId) => {
      dropTurn(turnId)
      setActiveLaneRecording(null)
    },
    [dropTurn],
  )

  // Re-arm is a no-op for the happy path now (mic stays armed through STT/LLM),
  // but clear / failed captures still call it explicitly.
  const setArmedRef = useRef((_armed) => {})
  const processTranslationRef = useRef(null)
  const rearmMic = useCallback(() => {
    setArmedRef.current(true)
  }, [])

  const clearContinuationWait = useCallback(() => {
    if (continuationTimerRef.current != null) {
      clearTimeout(continuationTimerRef.current)
      continuationTimerRef.current = null
    }
    pendingTranslateRef.current = null
  }, [])

  const armContinuationWait = useCallback(
    (job) => {
      clearContinuationWait()
      pendingTranslateRef.current = job
      continuationTimerRef.current = setTimeout(() => {
        const run = pendingTranslateRef.current
        pendingTranslateRef.current = null
        continuationTimerRef.current = null
        run?.()
      }, CONTINUE_WINDOW_MS)
    },
    [clearContinuationWait],
  )

  const abortTurnPipeline = useCallback((turnId) => {
    if (turnId == null) return
    const c = inflightByTurnRef.current.get(turnId)
    c?.abort()
    inflightByTurnRef.current.delete(turnId)
    if (ttsQueueRef.current.turnId === turnId) stopSpeaking()
  }, [stopSpeaking])

  // Speech detected. A trailing “och”/“and” (pendingTranslateRef) glues the
  // next burst onto the open turn. Other speech always starts a new row so
  // the other person is not concatenated onto this clip. TTS from the *other*
  // person is ducked, not cut; same-speaker redo stops that turn's stale
  // translation.
  const beginUtterance = useCallback(() => {
    const lane = activePersonRef.current
    const epoch = generationRef.current
    const now = performance.now()
    const open = openTurnRef.current
    const prev = previousTurnRef.current
    const sameOpen =
      open && open.lane === lane && open.generation === epoch && open.turnId != null
    const dtOpen = sameOpen ? now - open.at : Infinity
    const samePrev =
      prev && prev.lane === lane && prev.generation === epoch && prev.turnId != null
    const dtPrev = samePrev ? now - prev.at : Infinity

    const waitingForCue = pendingTranslateRef.current != null
    const ttsBusy =
      ttsQueueRef.current.playing || ttsQueueRef.current.pending.length > 0
    if (ttsBusy) {
      // Only cut TTS when the same speaker is clearly continuing a
      // trailing “och”/“and”. Any other burst — including the other
      // person talking in their language — ducks instead of chopping
      // the translation they are answering.
      if (waitingForCue && ttsQueueRef.current.lane === lane) stopSpeaking()
      else setTtsDuckRef.current?.(true)
    }
    if (
      sameOpen &&
      waitingForCue &&
      dtOpen < CONTINUE_WINDOW_MS + CONTINUATION_HOLD_MS
    ) {
      clearContinuationWait()
      abortTurnPipeline(open.turnId)
      setActiveLaneRecording(lane)
      const utterance = {
        turnId: open.turnId,
        lane,
        src: open.src,
        dst: open.dst,
        generation: epoch,
        merge: "continue",
        samples: open.samples,
      }
      activeTurnIdRef.current = open.turnId
      activeUtteranceRef.current = utterance
      updateTurn(open.turnId, {
        status: "transcribing",
        error: null,
        targetText: "",
      })
      return utterance
    }

    setActiveLaneRecording(lane)
    playBlip("ping")

    const { src, dst } = langsForLane(
      lane,
      lang1IndexRef.current,
      lang2IndexRef.current,
    )
    const turnId = nextTurnId.current++
    const merge =
      samePrev && dtPrev < REPAIR_WINDOW_MS ? "repair-candidate" : null
    const utterance = {
      turnId,
      lane,
      src,
      dst,
      generation: epoch,
      merge,
      samples: null,
    }
    activeTurnIdRef.current = turnId
    activeUtteranceRef.current = utterance
    setTurns((prevTurns) => [
      ...prevTurns,
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
    return utterance
  }, [abortTurnPipeline, clearContinuationWait, stopSpeaking, updateTurn])

  // Silence after speech — or Enter cutting the capture short. Mic stays armed
  // so more speech during STT/LLM can open another turn. `utteranceFromCapture`
  // is the snapshot from speech-start; live refs may already belong to a later
  // turn if the other person started talking while this clip was encoding.
  const endUtterance = useCallback(
    (payload, utteranceFromCapture) => {
      setTtsDuckRef.current?.(false)
      const marks = { keyup: performance.now() }
      const utterance = utteranceFromCapture || activeUtteranceRef.current
      const turnId = utterance?.turnId ?? null
      const generation = utterance?.generation
      const src = utterance?.src
      const dst = utterance?.dst
      const lane = utterance?.lane

      if (
        generation == null ||
        turnId == null ||
        !src ||
        !dst ||
        generationRef.current !== generation
      ) {
        abandonActiveTurn(turnId)
        rearmMic()
        return
      }
      if (!payload?.base64Data) {
        if (utterance.merge === "continue") {
          if (activeTurnIdRef.current === turnId) setActiveLaneRecording(null)
          rearmMic()
          return
        }
        abandonActiveTurn(turnId)
        rearmMic()
        return
      }

      const finish = (base64Data, samples) => {
        if (utterance.merge !== "repair-candidate") {
          openTurnRef.current = {
            turnId,
            lane,
            src,
            dst,
            generation,
            samples,
            at: performance.now(),
          }
        }
        if (activeTurnIdRef.current === turnId) {
          setActiveLaneRecording(null)
        }
        setArmedRef.current(true)
        processTranslationRef.current?.(
          lane,
          base64Data,
          turnId,
          src,
          dst,
          generation,
          marks,
          { merge: utterance.merge, samples },
        )
      }

      if (utterance.merge === "continue" && utterance.samples && payload.samples) {
        const samples = concatSamples(utterance.samples, payload.samples)
        void samplesToBase64Pcm(samples)
          .then((base64Data) => finish(base64Data, samples))
          .catch((err) => {
            console.error("continue concat failed:", err)
            abandonActiveTurn(turnId)
            rearmMic()
          })
        return
      }

      finish(payload.base64Data, payload.samples || null)
    },
    [abandonActiveTurn, rearmMic],
  )

  const handleInterim = useCallback(
    (payload, utterance) => {
      if (!payload?.base64Data || !utterance?.turnId) return
      if (sttInflightRef.current > 0) return
      if (generationRef.current !== utterance.generation) return
      interimAbortRef.current?.abort()
      const ac = new AbortController()
      interimAbortRef.current = ac
      transcribeAudio(payload.base64Data, utterance.src.code, {
        otherLanguage: utterance.dst.code,
        autoLanguage: false,
        signal: ac.signal,
      })
        .then((stt) => {
          if (ac.signal.aborted) return
          const text = normalizeSttText(stt.text || "")
          if (!text || isBackchannel(text) || !isSpeakable(text)) return
          updateTurn(utterance.turnId, { sourceText: text })
        })
        .catch((err) => {
          if (err.name !== "AbortError") console.error("interim STT:", err)
        })
    },
    [updateTurn],
  )

  const {
    isCapturing,
    analyser,
    micError,
    setArmed,
    cancelCapture,
    finishCapture,
    getAudioContext,
    setTtsPlayback,
    clearTtsPlayback,
    setTtsDuck,
    getTtsGainNode,
    setSilenceHold,
  } = useVoiceActivity({
    onSpeechStart: beginUtterance,
    onSpeechEnd: endUtterance,
    onInterim: handleInterim,
    enabled: true,
  })
  setArmedRef.current = setArmed
  getAudioContextRef.current = getAudioContext
  setTtsPlaybackRef.current = setTtsPlayback
  clearTtsPlaybackRef.current = clearTtsPlayback
  setTtsDuckRef.current = setTtsDuck
  getTtsGainNodeRef.current = getTtsGainNode
  setSilenceHoldRef.current = setSilenceHold

  // Translation Pipeline. Epoch must still match (clear conversation); a newer
  // overlapping utterance or an Enter person-switch does *not* change the epoch.
  const processTranslation = async (
    lane,
    base64Data,
    turnId,
    src,
    dst,
    generation,
    marks,
    extra = {},
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
    const isCurrent = () => generationRef.current === generation
    const controller = new AbortController()
    inflightByTurnRef.current.set(turnId, controller)
    const cfg = configRef.current

    if (marks) marks.turnId = turnId
    if (activeTurnIdRef.current === turnId) activeTurnIdRef.current = null

    const dropIfSuperseded = () => {
      cancelTurnIfPending(turnId)
    }

    const runTranslate = async (spokenSrc, spokenDst, transcribedText) => {
      if (!isCurrent()) {
        dropIfSuperseded()
        return
      }
      updateTurn(turnId, {
        sourceText: transcribedText,
        sourceLang: spokenSrc.code,
        targetLang: spokenDst.code,
        status: "translating",
      })

      if (cfg.enableTts) {
        beginTTSSession(
          (played) => {
            if (!isCurrent()) return
            if (!played) reportLatency(marks, false)
          },
          { turnId, lane },
        )
      }

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
        if (!isSpeakable(ready)) {
          spokenChars = upto
          return
        }
        spokenChars = upto
        if (cfg.enableTts) void enqueueTTS(ready, spokenDst.ttsLang, marks)
      }

      const result = await translateTextStreaming(
        transcribedText,
        {
          ...cfg,
          modelName: cfg.modelName,
          systemPrompt: `Translate the user's text from ${spokenSrc.name.split(" ")[0]} into ${spokenDst.name.split(" ")[0]}. Reply with the translation only — no explanations, no alternatives, no quotes, no preamble.`,
        },
        (partial) => {
          if (!isCurrent()) return
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
      if (result.translation !== result.raw) spokenChars = 0
      updateTurn(turnId, { targetText: result.translation, status: "done" })
      speakCompleteSentences(result.translation, true)
      if (!cfg.enableTts) reportLatency(marks, false)
      if (cfg.enableTts) sealTTSQueue()
    }

    let keepInflight = false
    try {
      let transcribedText
      let sttLanguage = src.code
      if (extra.knownText) {
        transcribedText = normalizeSttText(extra.knownText)
        if (marks) marks.stt = performance.now()
      } else {
        sttInflightRef.current += 1
        interimAbortRef.current?.abort()
        const stt = await transcribeAudio(base64Data, src.code, {
          otherLanguage: dst.code,
          autoLanguage: true,
          signal: controller.signal,
        })
        sttInflightRef.current = Math.max(0, sttInflightRef.current - 1)
        transcribedText = normalizeSttText(stt.text || "")
        sttLanguage = stt.language || src.code
      }
      if (!isCurrent()) {
        dropIfSuperseded()
        return
      }
      if (marks && !marks.stt) marks.stt = performance.now()
      if (!transcribedText || !isSpeakable(transcribedText) || isBackchannel(transcribedText)) {
        if (extra.merge === "continue") {
          // Filler after a real sentence: keep the earlier source, translate it.
          const kept = previousTurnRef.current?.turnId === turnId
            ? previousTurnRef.current.sourceText
            : ""
          if (kept) {
            await runTranslate(src, dst, kept)
            return
          }
        }
        dropTurn(turnId)
        return
      }

      if (
        extra.merge === "repair-candidate" &&
        isRepairUtterance(transcribedText) &&
        previousTurnRef.current &&
        previousTurnRef.current.turnId !== turnId &&
        previousTurnRef.current.lane === lane
      ) {
        const prev = previousTurnRef.current
        const repaired = stripRepairCue(transcribedText)
        abandonActiveTurn(turnId)
        abortTurnPipeline(prev.turnId)
        previousTurnRef.current = {
          ...prev,
          sourceText: repaired,
          at: performance.now(),
        }
        openTurnRef.current = {
          turnId: prev.turnId,
          lane: prev.lane,
          src: prev.src,
          dst: prev.dst,
          generation: prev.generation,
          samples: extra.samples || prev.samples,
          at: performance.now(),
        }
        updateTurn(prev.turnId, {
          sourceText: repaired,
          targetText: "",
          status: "translating",
          error: null,
        })
        await processTranslationRef.current?.(
          prev.lane,
          base64Data,
          prev.turnId,
          prev.src,
          prev.dst,
          prev.generation,
          { keyup: performance.now() },
          { merge: null, samples: extra.samples, knownText: repaired },
        )
        return
      }

      const lang1 = AVAILABLE_LANGUAGES[lang1IndexRef.current]
      const lang2 = AVAILABLE_LANGUAGES[lang2IndexRef.current]
      const routed = routeSpokenTurn(
        sttLanguage,
        lane,
        src,
        dst,
        lang1,
        lang2,
      )
      const spokenSrc = routed.src
      const spokenDst = routed.dst
      if (routed.flipped) {
        updateTurn(turnId, {
          lane: routed.lane,
          sourceLang: spokenSrc.code,
          targetLang: spokenDst.code,
        })
        if (activePersonRef.current !== routed.lane) {
          activePersonRef.current = routed.lane
          setActivePerson(routed.lane)
          playBlip("speaker")
        }
        lane = routed.lane
      }

      previousTurnRef.current = {
        turnId,
        lane,
        src: spokenSrc,
        dst: spokenDst,
        generation,
        samples: extra.samples || null,
        sourceText: transcribedText,
        at: performance.now(),
      }
      if (openTurnRef.current?.turnId === turnId) {
        openTurnRef.current = {
          ...openTurnRef.current,
          lane,
          src: spokenSrc,
          dst: spokenDst,
          samples: extra.samples || openTurnRef.current.samples,
        }
      }

      const translateNow = () => {
        if (!isCurrent()) return
        inflightByTurnRef.current.set(turnId, controller)
        void runTranslate(spokenSrc, spokenDst, transcribedText)
          .catch((err) => {
            if (err.name === "AbortError" || !isCurrent()) {
              dropIfSuperseded()
              return
            }
            console.error(err)
            updateTurn(turnId, { status: "error", error: err.message })
            sealTTSQueue()
          })
          .finally(() => {
            inflightByTurnRef.current.delete(turnId)
          })
      }

      if (endsWithContinuationCue(transcribedText)) {
        setSilenceHoldRef.current?.(CONTINUATION_HOLD_MS)
        updateTurn(turnId, {
          sourceText: transcribedText,
          sourceLang: spokenSrc.code,
          targetLang: spokenDst.code,
          status: "transcribing",
        })
        keepInflight = true
        armContinuationWait(translateNow)
        return
      }

      setSilenceHoldRef.current?.(0)
      await runTranslate(spokenSrc, spokenDst, transcribedText)
    } catch (err) {
      sttInflightRef.current = Math.max(0, sttInflightRef.current - 1)
      if (err.name === "AbortError" || !isCurrent()) {
        dropIfSuperseded()
        return
      }
      console.error(err)
      updateTurn(turnId, { status: "error", error: err.message })
      sealTTSQueue()
    } finally {
      if (!keepInflight) inflightByTurnRef.current.delete(turnId)
    }
  }
  processTranslationRef.current = processTranslation

  useEffect(() => {
    prewarmLang(AVAILABLE_LANGUAGES[lang1Index].code)
    prewarmLang(AVAILABLE_LANGUAGES[lang2Index].code)
  }, [lang1Index, lang2Index])

  // Wipe the conversation and stop anything still in flight.
  const handleClearConversation = useCallback(() => {
    generationRef.current += 1
    clearContinuationWait()
    interimAbortRef.current?.abort()
    for (const c of inflightByTurnRef.current.values()) c.abort()
    inflightByTurnRef.current.clear()
    activeTurnIdRef.current = null
    activeUtteranceRef.current = null
    openTurnRef.current = null
    previousTurnRef.current = null
    setTtsDuckRef.current?.(false)
    setSilenceHoldRef.current?.(0)
    stopSpeaking()
    cancelCapture()
    setActiveLaneRecording(null)
    setTurns([])
    rearmMic()
  }, [stopSpeaking, cancelCapture, rearmMic, clearContinuationWait])

  // The settings overlay lives in App.jsx but the conversation lives here, so
  // App borrows the handler through a ref rather than us lifting the state up.
  useEffect(() => {
    if (!clearConversationRef) return
    clearConversationRef.current = handleClearConversation
    return () => {
      clearConversationRef.current = null
    }
  }, [clearConversationRef, handleClearConversation])

  // Enter switches active person; arrows rotate languages. Speech detection
  // replaces Space / Z / X push-to-talk.
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return
      const key = e.key.toLowerCase()

      if (e.key === "Enter") {
        e.preventDefault()
        if (isCapturing) {
          // Person changed mid-utterance — close the capture as a finished
          // turn so STT → Gemma → TTS still run for the speaker, then listen
          // as the other person.
          void finishCapture()
        }
        playBlip("speaker")
        setActivePerson((p) => {
          const next = p === 1 ? 2 : 1
          activePersonRef.current = next
          return next
        })
        return
      }

      if (config.keyboardMode === "landscape") {
        if (e.key === "ArrowLeft") {
          e.preventDefault()
          handleRotateLanguage(activePerson, -1)
        } else if (e.key === "ArrowRight") {
          e.preventDefault()
          handleRotateLanguage(activePerson, 1)
        }
      } else {
        if (e.key === "ArrowLeft") {
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

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [
    config.keyboardMode,
    activePerson,
    isCapturing,
    handleRotateLanguage,
    finishCapture,
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
            isActivePerson={activePerson === 1}
            recordKeyHint={activePerson === 1 ? "MIC" : null}
            onRotate={(dir) => handleRotateLanguage(1, dir)}
          />
          <LanguageLane
            laneId={2}
            laneLabel="2"
            languages={AVAILABLE_LANGUAGES}
            currentIndex={lang2Index}
            isRecording={activeLaneRecording === 2}
            isActivePerson={activePerson === 2}
            recordKeyHint={activePerson === 2 ? "MIC" : null}
            onRotate={(dir) => handleRotateLanguage(2, dir)}
          />
        </div>

        <div className="keyboard-legend" aria-live="polite">
          <span>
            <kbd>⏎</kbd> person
          </span>
          <span>speak to talk</span>
          {config.keyboardMode === "landscape" ? (
            <span>
              <kbd>←→</kbd> language
            </span>
          ) : (
            <>
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
          isRecording={isCapturing}
          analyser={analyser}
          barsCount={parseInt(config.visualizerBars, 10)}
        />
      </main>
    </div>
  )
}

export default TranslatorApp
