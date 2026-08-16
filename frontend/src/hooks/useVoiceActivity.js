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

import { useState, useRef, useCallback, useEffect } from "react"
import { getMergedSamples, resample, samplesToBase64Pcm } from "../utils/audioHelpers"

// Energy-based VAD for continuous kiosk listening, with a short NLMS echo
// canceller so TTS played from the speakers is subtracted from the mic before
// VAD / capture. That lets the user keep talking while the previous turn is
// still being transcribed or spoken without the translation leaking in.

const SPEECH_RMS = 0.015
const SILENCE_MS_SHORT = 560
const SILENCE_MS_LONG = 1250
const MIN_SPEECH_MS = 400
const MAX_UTTERANCE_MS = 15000
const PRE_ROLL_CHUNKS = 4
const INTERIM_MS = 1100
const INTERIM_MIN_MS = 700
const TTS_DUCK_GAIN = 0.12
// ~21 ms of taps at 48 kHz — short speaker→mic path on a kiosk desk.
const AEC_FILTER_LEN = 1024
const AEC_MU = 0.5
const AEC_EPS = 1e-6
const AEC_UPDATE_STRIDE = 4 // update weights every N samples (CPU)

async function samplesToPayload(samples, sampleRate) {
  const targetSampleRate = 16000
  const resampledSamples = resample(samples, sampleRate, targetSampleRate)

  let peak = 0
  for (let i = 0; i < resampledSamples.length; i++) {
    const a = Math.abs(resampledSamples[i])
    if (a > peak) peak = a
  }
  // AEC can theoretically overshoot; Whisper wants samples in [-1, 1].
  if (peak > 1) {
    const inv = 1 / peak
    for (let i = 0; i < resampledSamples.length; i++) {
      resampledSamples[i] *= inv
    }
    peak = 1
  }
  const durationMs = ((resampledSamples.length / targetSampleRate) * 1000) | 0
  let sumSq = 0
  for (let i = 0; i < resampledSamples.length; i++) {
    sumSq += resampledSamples[i] * resampledSamples[i]
  }
  const rms = resampledSamples.length ? Math.sqrt(sumSq / resampledSamples.length) : 0
  console.log(
    `[vad] ${durationMs}ms, peak=${peak.toFixed(3)}, rms=${rms.toFixed(3)}`,
  )

  const base64Data = await samplesToBase64Pcm(resampledSamples)
  return { base64Data, samples: resampledSamples, durationMs }
}

function rmsOf(frame) {
  let sum = 0
  for (let i = 0; i < frame.length; i++) {
    const s = frame[i]
    sum += s * s
  }
  return Math.sqrt(sum / frame.length)
}

/**
 * @param {{
 *   onSpeechStart?: () => unknown,
 *   onSpeechEnd?: (payload: object | null, context?: unknown) => void,
 *   onInterim?: (payload: object, context?: unknown) => void,
 *   enabled?: boolean,
 * }} options
 */
export function useVoiceActivity({
  onSpeechStart,
  onSpeechEnd,
  onInterim,
  enabled = true,
} = {}) {
  const [isListening, setIsListening] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [micError, setMicError] = useState(null)
  const [analyser, setAnalyser] = useState(null)

  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const sourceRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const tapDestRef = useRef(null)
  const streamRef = useRef(null)
  const ttsGainRef = useRef(null)

  const armedRef = useRef(true)
  const capturingRef = useRef(false)
  const utteranceChunksRef = useRef([])
  const preRollRef = useRef([])
  const speechStartedAtRef = useRef(0)
  const lastLoudAtRef = useRef(0)
  const lastLoudRmsRef = useRef(0)
  const sampleRateRef = useRef(16000)
  const silenceHoldMsRef = useRef(0)
  const lastInterimAtRef = useRef(0)
  const interimBusyRef = useRef(false)
  const ttsDuckRef = useRef(1)
  // Bumped on every stop so a late getUserMedia cannot resurrect a torn-down graph.
  const listenGenRef = useRef(0)
  // Snapshot returned by onSpeechStart — passed back on end so a later
  // utterance cannot steal this capture's turn (Enter / overlapping speech).
  const captureContextRef = useRef(null)

  // Far-end (TTS) playback currently going to the speakers — used as the AEC
  // reference so English (or any TTS) is filtered from the mic.
  const ttsPlaybackRef = useRef(null) // { data, sampleRate, startedAt }
  const aecWRef = useRef(new Float32Array(AEC_FILTER_LEN))
  const aecXRef = useRef(new Float32Array(AEC_FILTER_LEN))
  const aecIdxRef = useRef(0)

  const onSpeechStartRef = useRef(onSpeechStart)
  const onSpeechEndRef = useRef(onSpeechEnd)
  const onInterimRef = useRef(onInterim)
  useEffect(() => {
    onSpeechStartRef.current = onSpeechStart
  }, [onSpeechStart])
  useEffect(() => {
    onSpeechEndRef.current = onSpeechEnd
  }, [onSpeechEnd])
  useEffect(() => {
    onInterimRef.current = onInterim
  }, [onInterim])

  const setArmed = useCallback((armed) => {
    armedRef.current = !!armed
  }, [])

  const setSilenceHold = useCallback((ms) => {
    silenceHoldMsRef.current = Math.max(0, ms | 0)
  }, [])

  const getTtsGainNode = useCallback(() => ttsGainRef.current, [])

  const setTtsDuck = useCallback((on) => {
    const g = ttsGainRef.current
    const ctx = audioContextRef.current
    const target = on ? TTS_DUCK_GAIN : 1
    ttsDuckRef.current = target
    if (ttsPlaybackRef.current) ttsPlaybackRef.current.gain = target
    if (!g || !ctx) return
    try {
      g.gain.cancelScheduledValues(ctx.currentTime)
      g.gain.setTargetAtTime(target, ctx.currentTime, 0.04)
    } catch (_) {
      g.gain.value = target
    }
  }, [])

  const resetAec = useCallback(() => {
    aecWRef.current.fill(0)
    aecXRef.current.fill(0)
    aecIdxRef.current = 0
  }, [])

  const clearTtsPlayback = useCallback(() => {
    ttsPlaybackRef.current = null
  }, [])

  /**
   * Register TTS PCM that is about to play through the shared AudioContext.
   * `startedAt` is `audioContext.currentTime` at source.start().
   */
  const setTtsPlayback = useCallback((data, sampleRate, startedAt) => {
    ttsPlaybackRef.current = {
      data,
      sampleRate,
      startedAt,
      gain: ttsDuckRef.current,
    }
    // Fresh adaptation each chunk — room path is short and stable enough.
    aecWRef.current.fill(0)
    aecXRef.current.fill(0)
    aecIdxRef.current = 0
  }, [])

  const getAudioContext = useCallback(() => audioContextRef.current, [])

  const teardownGraph = useCallback(() => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect()
      scriptProcessorRef.current.onaudioprocess = null
      scriptProcessorRef.current = null
    }
    if (tapDestRef.current) {
      tapDestRef.current = null
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    ttsPlaybackRef.current = null
    ttsGainRef.current = null
  }, [])

  const discardCapture = useCallback(() => {
    capturingRef.current = false
    setIsCapturing(false)
    utteranceChunksRef.current = []
    preRollRef.current = []
    speechStartedAtRef.current = 0
    lastLoudAtRef.current = 0
    lastLoudRmsRef.current = 0
    captureContextRef.current = null
  }, [])

  const finishCapture = useCallback(async () => {
    if (!capturingRef.current) return
    capturingRef.current = false
    setIsCapturing(false)

    const chunks = utteranceChunksRef.current
    const context = captureContextRef.current
    utteranceChunksRef.current = []
    preRollRef.current = []
    captureContextRef.current = null
    const startedAt = speechStartedAtRef.current
    speechStartedAtRef.current = 0
    lastLoudAtRef.current = 0

    const durationMs = startedAt ? performance.now() - startedAt : 0
    if (chunks.length === 0 || durationMs < MIN_SPEECH_MS) {
      console.log(`[vad] drop short utterance ${durationMs | 0}ms`)
      onSpeechEndRef.current?.(null, context)
      return
    }

    const merged = getMergedSamples(chunks)
    try {
      const payload = await samplesToPayload(merged, sampleRateRef.current)
      onSpeechEndRef.current?.(payload, context)
    } catch (err) {
      console.error("VAD encode failed:", err)
      onSpeechEndRef.current?.(null, context)
    }
  }, [])

  const cancelCapture = useCallback(() => {
    if (!capturingRef.current) return
    const context = captureContextRef.current
    discardCapture()
    onSpeechEndRef.current?.(null, context)
  }, [discardCapture])

  // Subtract estimated speaker echo from one mic frame. Far-end samples are
  // read from the TTS buffer using the AudioContext clock so they stay aligned
  // with what the speakers are actually playing.
  const cancelEcho = useCallback((near) => {
    const tts = ttsPlaybackRef.current
    const ctx = audioContextRef.current
    // No far-end: skip the filter entirely. Adapting on silence makes the
    // weights wander, and a later TTS chunk then explodes (peak ≫ 1).
    if (!tts || !ctx) return near

    const out = new Float32Array(near.length)
    const w = aecWRef.current
    const xHist = aecXRef.current
    let xIdx = aecIdxRef.current
    let unstable = false

    for (let i = 0; i < near.length; i++) {
      let far = 0
      const t = ctx.currentTime - tts.startedAt
      const idx = Math.floor(t * tts.sampleRate) + i
      if (idx >= 0 && idx < tts.data.length) {
        far = tts.data[idx] * (tts.gain ?? 1)
      }

      xHist[xIdx % AEC_FILTER_LEN] = far
      let y = 0
      let xPow = 0
      for (let k = 0; k < AEC_FILTER_LEN; k++) {
        const xk = xHist[(xIdx - k + AEC_FILTER_LEN) % AEC_FILTER_LEN]
        y += w[k] * xk
        xPow += xk * xk
      }
      let e = near[i] - y
      if (e > 1) e = 1
      else if (e < -1) e = -1
      out[i] = e
      if (Math.abs(y) > 4) unstable = true
      if (xPow > AEC_EPS && i % AEC_UPDATE_STRIDE === 0) {
        const norm = AEC_MU / (AEC_EPS + xPow)
        for (let k = 0; k < AEC_FILTER_LEN; k++) {
          const xk = xHist[(xIdx - k + AEC_FILTER_LEN) % AEC_FILTER_LEN]
          w[k] += norm * e * xk
        }
      }
      xIdx++
    }
    aecIdxRef.current = xIdx
    if (unstable) {
      w.fill(0)
      xHist.fill(0)
      aecIdxRef.current = 0
      return near
    }
    return out
  }, [])

  const handleAudioProcess = useCallback(
    (e) => {
      const inputData = e.inputBuffer.getChannelData(0)
      // Always run AEC so the filter stays adapted while TTS plays, even when
      // we are not capturing — otherwise the first frames of a barge-in are dirty.
      const cleaned = cancelEcho(inputData)
      const rms = rmsOf(cleaned)
      const now = performance.now()

      if (!capturingRef.current) {
        preRollRef.current.push(cleaned)
        if (preRollRef.current.length > PRE_ROLL_CHUNKS) {
          preRollRef.current.shift()
        }
        if (!armedRef.current) return
        if (rms < SPEECH_RMS) return

        capturingRef.current = true
        setIsCapturing(true)
        speechStartedAtRef.current = now
        lastLoudAtRef.current = now
        lastLoudRmsRef.current = rms
        lastInterimAtRef.current = now
        utteranceChunksRef.current = [...preRollRef.current]
        preRollRef.current = []
        captureContextRef.current = onSpeechStartRef.current?.() ?? null
        return
      }

      utteranceChunksRef.current.push(cleaned)
      if (rms >= SPEECH_RMS) {
        lastLoudAtRef.current = now
        lastLoudRmsRef.current = rms
      }

      const utteredMs = now - speechStartedAtRef.current
      const silentMs = now - lastLoudAtRef.current
      const abrupt = lastLoudRmsRef.current >= 0.06
      const silenceNeed = Math.max(
        abrupt ? SILENCE_MS_SHORT : SILENCE_MS_LONG,
        silenceHoldMsRef.current,
      )
      if (utteredMs >= MAX_UTTERANCE_MS || silentMs >= silenceNeed) {
        void finishCapture()
        return
      }

      if (
        onInterimRef.current &&
        utteredMs >= INTERIM_MIN_MS &&
        now - lastInterimAtRef.current >= INTERIM_MS &&
        !interimBusyRef.current
      ) {
        lastInterimAtRef.current = now
        const snap = utteranceChunksRef.current.slice()
        const ctx = captureContextRef.current
        interimBusyRef.current = true
        void (async () => {
          try {
            const merged = getMergedSamples(snap)
            const payload = await samplesToPayload(merged, sampleRateRef.current)
            if (payload) onInterimRef.current?.(payload, ctx)
          } catch (err) {
            console.error("VAD interim encode failed:", err)
          } finally {
            interimBusyRef.current = false
          }
        })()
      }
    },
    [cancelEcho, finishCapture],
  )

  const startListening = useCallback(async () => {
    if (streamRef.current || scriptProcessorRef.current) return true
    setMicError(null)
    const gen = ++listenGenRef.current
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Browser AEC on: mic tap does not go to speakers (zero-gain keep-alive
          // below), so AEC can cancel real TTS from the loudspeaker. Software
          // NLMS above is the backup when browser AEC is weak.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      })
      if (gen !== listenGenRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return false
      }
      streamRef.current = stream

      const AudioContext = window.AudioContext || window.webkitAudioContext
      audioContextRef.current = new AudioContext()
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume()
      }
      if (gen !== listenGenRef.current) {
        teardownGraph()
        if (audioContextRef.current && audioContextRef.current.state !== "closed") {
          await audioContextRef.current.close()
          audioContextRef.current = null
        }
        return false
      }
      sampleRateRef.current = audioContextRef.current.sampleRate
      resetAec()
      console.log(
        `[vad] listening @ ${sampleRateRef.current}Hz, ctx=${audioContextRef.current.state}`,
      )

      const ttsGain = audioContextRef.current.createGain()
      ttsGain.gain.value = ttsDuckRef.current
      ttsGain.connect(audioContextRef.current.destination)
      ttsGainRef.current = ttsGain

      const source = audioContextRef.current.createMediaStreamSource(stream)
      sourceRef.current = source

      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)
      setAnalyser(analyserRef.current)

      const scriptProcessor = audioContextRef.current.createScriptProcessor(
        4096,
        1,
        1,
      )
      scriptProcessorRef.current = scriptProcessor
      scriptProcessor.onaudioprocess = handleAudioProcess

      // Keep ScriptProcessor alive without audible mic bleed. A zero-gain
      // connection to destination is more reliable than MediaStreamDestination
      // alone (some Chromium builds stop firing otherwise).
      const mute = audioContextRef.current.createGain()
      mute.gain.value = 0
      tapDestRef.current = mute
      source.connect(scriptProcessor)
      scriptProcessor.connect(mute)
      mute.connect(audioContextRef.current.destination)

      setIsListening(true)
      return true
    } catch (err) {
      console.error("Error accessing microphone:", err)
      setMicError(
        err.message ||
          "Microphone access failed (HTTPS required for remote devices)",
      )
      teardownGraph()
      setIsListening(false)
      return false
    }
  }, [handleAudioProcess, teardownGraph, resetAec])

  const stopListening = useCallback(async () => {
    listenGenRef.current += 1
    discardCapture()
    teardownGraph()
    setAnalyser(null)
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      await audioContextRef.current.close()
      audioContextRef.current = null
    }
    setIsListening(false)
  }, [discardCapture, teardownGraph])

  useEffect(() => {
    if (!enabled) {
      void stopListening()
      return undefined
    }
    void startListening()
    return () => {
      void stopListening()
    }
  }, [enabled, startListening, stopListening])

  // After a hard reload the AudioContext often stays suspended until a user
  // gesture. Resume on the first click/key so VAD actually receives frames.
  useEffect(() => {
    const resume = () => {
      const ctx = audioContextRef.current
      if (ctx && ctx.state === "suspended") {
        void ctx.resume().then(() => {
          console.log(`[vad] AudioContext resumed → ${ctx.state}`)
        })
      }
    }
    window.addEventListener("pointerdown", resume)
    window.addEventListener("keydown", resume)
    return () => {
      window.removeEventListener("pointerdown", resume)
      window.removeEventListener("keydown", resume)
    }
  }, [])

  useEffect(() => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.onaudioprocess = handleAudioProcess
    }
  }, [handleAudioProcess])

  return {
    isListening,
    isCapturing,
    analyser,
    micError,
    setArmed,
    setSilenceHold,
    setTtsDuck,
    getTtsGainNode,
    cancelCapture,
    finishCapture,
    startListening,
    stopListening,
    getAudioContext,
    setTtsPlayback,
    clearTtsPlayback,
  }
}
