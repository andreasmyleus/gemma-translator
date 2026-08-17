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
import {
  farEndSampleIndex,
  getMergedSamples,
  resample,
  samplesToBase64Pcm,
} from "../utils/audioHelpers"

// Energy-based VAD for continuous kiosk listening, with a short NLMS echo
// canceller so TTS played from the speakers is subtracted from the mic before
// VAD / capture. That lets the user keep talking while the previous turn is
// still being transcribed or spoken without the translation leaking in.

const SPEECH_RMS = 0.015
// End-of-turn hangover. One value, not the old two-tier 560/1250 ms rule.
//
// That rule picked the short tier on `lastLoudRms >= 0.06`, where lastLoudRms
// is the RMS of the last frame *above* SPEECH_RMS — i.e. the tail of the
// fade-out, by construction the quietest speech in the utterance (measured
// 0.02–0.07 against a speech median of 0.15). It therefore fired essentially at
// random depending on where the fade landed relative to a frame boundary: 3 of
// 7 turns in one bench conversation, 0 of 5 in another. Every miss fell back to
// 1250 ms, which is longer than an ordinary conversational gap, so the VAD glued
// the two speakers into one utterance (see bench/conversations.json,
// sv-en-directions: 5 spoken turns collapsed into 2 captures).
//
// 700 ms is measured, not guessed. Across both bench conversations the longest
// pause *inside* an utterance is 256 ms and the shortest gap *between* speakers
// is 1109 ms, so anything in 350–1050 ms separates them perfectly. 700 sits mid-
// band, leaving ~2.7x headroom over the observed intra-utterance pause for real
// human hesitation, which the synthesized fixtures do not reproduce. It is also
// pure latency: this wait is on the front of every single turn.
const SILENCE_MS = 700
const MIN_SPEECH_MS = 400
// Silence after which the utterance is transcribed *speculatively*, while the
// hangover is still running. If nobody speaks again the result is already in
// hand when the capture closes, so STT comes off the critical path and the turn
// starts translating ~(SILENCE_MS - SPECULATIVE_STT_MS) sooner.
//
// 340 ms = 4 frames at 48 kHz, chosen above the longest pause measured *inside*
// an utterance (256 ms across both bench conversations) so an ordinary
// between-words gap does not trigger it. At most one speculative call per
// utterance: a hesitant speaker who resumes costs one wasted transcription, not
// one per pause. The clip also excludes the trailing silence, which measurably
// helps — Whisper hallucinated more on the padded capture (WER 0.60 vs 0.30 on
// the same turn) than on the speech alone.
const SPECULATIVE_STT_MS = 340
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
 *   onEarlyEnd?: (payload: object, context?: unknown) => void,
 *   enabled?: boolean,
 * }} options
 */
export function useVoiceActivity({
  onSpeechStart,
  onSpeechEnd,
  onInterim,
  onEarlyEnd,
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
  const noiseFloorRef = useRef(0.006)
  const sampleRateRef = useRef(16000)
  const silenceHoldMsRef = useRef(0)
  const lastInterimAtRef = useRef(0)
  const interimBusyRef = useRef(false)
  // Speculative-STT bookkeeping, reset per capture. `earlySpeechAfter` is what
  // makes the result safe to reuse: it flips the moment a loud frame arrives
  // after the speculative clip was taken, which means that clip is no longer
  // the whole utterance.
  const earlyFiredRef = useRef(false)
  const earlySpeechAfterRef = useRef(false)
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
  // Delay line kept as two mirrored halves so the tap loop reads a *contiguous*
  // window (xHist[p..p+N-1]) instead of doing `% AEC_FILTER_LEN` on every one of
  // the ~1024 taps per sample. Same maths, roughly half the work — and this
  // loop runs on the main thread inside onaudioprocess, where a Pi has ~85 ms
  // per frame to spare.
  const aecXRef = useRef(new Float32Array(AEC_FILTER_LEN * 2))
  const aecPosRef = useRef(0)
  // Running Σx² over the window, so the NLMS normalisation does not re-sum
  // 1024 taps per sample. Resynced exactly once per frame (see cancelEcho).
  const aecPowRef = useRef(0)
  // Far-end samples that have been exactly zero in a row. Once that covers the
  // whole filter the window is provably zeros, y is 0 and no adaptation can
  // happen — so the frame can be passed through untouched.
  const aecZeroRunRef = useRef(AEC_FILTER_LEN)
  // Scratch far-end frame, grown to the ScriptProcessor's buffer size on first use.
  const aecFarRef = useRef(new Float32Array(0))

  const onSpeechStartRef = useRef(onSpeechStart)
  const onSpeechEndRef = useRef(onSpeechEnd)
  const onInterimRef = useRef(onInterim)
  const onEarlyEndRef = useRef(onEarlyEnd)
  useEffect(() => {
    onSpeechStartRef.current = onSpeechStart
  }, [onSpeechStart])
  useEffect(() => {
    onSpeechEndRef.current = onSpeechEnd
  }, [onSpeechEnd])
  useEffect(() => {
    onInterimRef.current = onInterim
  }, [onInterim])
  useEffect(() => {
    onEarlyEndRef.current = onEarlyEnd
  }, [onEarlyEnd])

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
    aecPosRef.current = 0
    aecPowRef.current = 0
    aecZeroRunRef.current = AEC_FILTER_LEN
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
    aecPosRef.current = 0
    aecPowRef.current = 0
    aecZeroRunRef.current = AEC_FILTER_LEN
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
    if (ttsGainRef.current) {
      ttsGainRef.current.disconnect()
      ttsGainRef.current = null
    }
    ttsPlaybackRef.current = null
  }, [])

  const discardCapture = useCallback(() => {
    capturingRef.current = false
    setIsCapturing(false)
    utteranceChunksRef.current = []
    preRollRef.current = []
    speechStartedAtRef.current = 0
    lastLoudAtRef.current = 0
    earlyFiredRef.current = false
    earlySpeechAfterRef.current = false
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
    const lastLoudAt = lastLoudAtRef.current
    // Usable only if the speculative clip really was the whole utterance.
    const earlyUsable = earlyFiredRef.current && !earlySpeechAfterRef.current
    speechStartedAtRef.current = 0
    lastLoudAtRef.current = 0
    earlyFiredRef.current = false
    earlySpeechAfterRef.current = false

    // Onset → last loud frame, NOT → now. Measuring to `now` always included
    // the full silence hangover, which is longer than MIN_SPEECH_MS, so the
    // short-utterance guard could never fire and every door slam or cough
    // bought itself a full STT call and a junk row in the transcript.
    const durationMs = startedAt ? Math.max(0, lastLoudAt - startedAt) : 0
    if (chunks.length === 0 || durationMs < MIN_SPEECH_MS) {
      console.log(`[vad] drop short utterance ${durationMs | 0}ms`)
      onSpeechEndRef.current?.(null, context)
      return
    }

    const merged = getMergedSamples(chunks)
    try {
      const payload = await samplesToPayload(merged, sampleRateRef.current)
      onSpeechEndRef.current?.({ ...payload, earlyUsable }, context)
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

    const N = AEC_FILTER_LEN
    const frameLen = near.length
    if (aecFarRef.current.length < frameLen) {
      aecFarRef.current = new Float32Array(frameLen)
    }
    const far = aecFarRef.current

    // Resolve the far-end frame first. A registered TTS buffer stays registered
    // until playback ends, so plenty of frames overlap none of it — and a
    // silent frame needs no filter at all.
    const elapsedSec = ctx.currentTime - tts.startedAt
    const micRate = ctx.sampleRate || sampleRateRef.current
    const gain = tts.gain ?? 1
    let farEnergy = 0
    for (let i = 0; i < frameLen; i++) {
      const idx = farEndSampleIndex(elapsedSec, i, tts.sampleRate, micRate)
      const v = idx >= 0 && idx < tts.data.length ? tts.data[idx] * gain : 0
      far[i] = v
      farEnergy += v * v
    }
    if (farEnergy === 0) {
      // Once the zero run covers the whole filter, the window is provably all
      // zeros: y ≡ 0 and Σx² ≡ 0, so the filter is the identity and no
      // adaptation can happen. Skipping is exact, not an approximation.
      if (aecZeroRunRef.current >= N) return near
      aecZeroRunRef.current += frameLen
    } else {
      aecZeroRunRef.current = 0
    }

    const out = new Float32Array(frameLen)
    const w = aecWRef.current
    const xHist = aecXRef.current
    let p = aecPosRef.current
    let unstable = false

    // Exact resync of the running power, once per frame. The incremental
    // update inside the loop is what keeps the per-sample cost down, but
    // accumulating ±x² over millions of samples drifts (and can go negative).
    let pow = 0
    for (let k = 0; k < N; k++) {
      const v = xHist[p + k]
      pow += v * v
    }

    for (let i = 0; i < frameLen; i++) {
      const v = far[i]
      // Walk the cursor backwards and mirror each sample into both halves, so
      // the window is xHist[p .. p+N-1], newest first, contiguous — no modulo
      // in the tap loops.
      p = p === 0 ? N - 1 : p - 1
      const dropped = xHist[p + N]
      xHist[p] = v
      xHist[p + N] = v
      pow += v * v - dropped * dropped
      if (pow < 0) pow = 0

      let y = 0
      for (let k = 0; k < N; k++) y += w[k] * xHist[p + k]
      let e = near[i] - y
      if (e > 1) e = 1
      else if (e < -1) e = -1
      out[i] = e
      if (y > 4 || y < -4) unstable = true
      if (pow > AEC_EPS && i % AEC_UPDATE_STRIDE === 0) {
        const norm = (AEC_MU / (AEC_EPS + pow)) * e
        for (let k = 0; k < N; k++) w[k] += norm * xHist[p + k]
      }
    }
    aecPosRef.current = p
    aecPowRef.current = pow
    if (unstable) {
      w.fill(0)
      xHist.fill(0)
      aecPosRef.current = 0
      aecPowRef.current = 0
      aecZeroRunRef.current = N
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
        const threshold = Math.max(SPEECH_RMS * 0.7, noiseFloorRef.current * 3.5)
        if (rms < threshold) {
          noiseFloorRef.current = noiseFloorRef.current * 0.98 + rms * 0.02
          return
        }

        capturingRef.current = true
        setIsCapturing(true)
        speechStartedAtRef.current = now
        lastLoudAtRef.current = now
        earlyFiredRef.current = false
        earlySpeechAfterRef.current = false
        lastInterimAtRef.current = now
        utteranceChunksRef.current = [...preRollRef.current]
        preRollRef.current = []
        captureContextRef.current = onSpeechStartRef.current?.() ?? null
        return
      }

      utteranceChunksRef.current.push(cleaned)
      if (rms >= SPEECH_RMS) {
        lastLoudAtRef.current = now
        // Speech after the speculative clip was taken means that clip is no
        // longer the whole utterance, so its transcription must not be reused.
        if (earlyFiredRef.current) earlySpeechAfterRef.current = true
      }

      const utteredMs = now - speechStartedAtRef.current
      const silentMs = now - lastLoudAtRef.current

      // Transcribe during the hangover rather than after it. Fires at most once
      // per capture, and only once the utterance already holds real speech.
      if (
        onEarlyEndRef.current &&
        !earlyFiredRef.current &&
        silentMs >= SPECULATIVE_STT_MS &&
        lastLoudAtRef.current - speechStartedAtRef.current >= MIN_SPEECH_MS
      ) {
        earlyFiredRef.current = true
        const snap = utteranceChunksRef.current.slice()
        const ctx = captureContextRef.current
        void (async () => {
          try {
            const payload = await samplesToPayload(
              getMergedSamples(snap),
              sampleRateRef.current,
            )
            onEarlyEndRef.current?.(payload, ctx)
          } catch (err) {
            console.error("VAD speculative encode failed:", err)
          }
        })()
      }
      // silenceHoldMs is the one legitimate extension: a turn ending on a
      // trailing "och"/"and" asks for more patience (see armContinuationWait).
      const silenceNeed = Math.max(SILENCE_MS, silenceHoldMsRef.current)
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

      // Build into locals and only publish to the refs once this call is still
      // the current generation. Assigning first and tearing down via
      // teardownGraph() on a late abort would destroy whichever graph the refs
      // point at by then — possibly a *newer* one that had already started.
      const AudioContext = window.AudioContext || window.webkitAudioContext
      const ctx = new AudioContext()
      if (ctx.state === "suspended") {
        await ctx.resume()
      }
      if (gen !== listenGenRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        if (ctx.state !== "closed") await ctx.close()
        return false
      }
      streamRef.current = stream
      audioContextRef.current = ctx
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
