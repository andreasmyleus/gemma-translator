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
import { getMergedSamples, resample, blobToBase64 } from "../utils/audioHelpers"

async function samplesToPayload(samples, sampleRate) {
  const targetSampleRate = 16000 // Whisper STT expects 16 kHz mono
  const resampledSamples = resample(samples, sampleRate, targetSampleRate)

  let peak = 0
  for (let i = 0; i < resampledSamples.length; i++) {
    const a = Math.abs(resampledSamples[i])
    if (a > peak) peak = a
  }
  console.log(
    `[mic] ${((resampledSamples.length / targetSampleRate) * 1000) | 0}ms, peak=${peak.toFixed(3)}`,
  )

  // Slice the exact view — Float32Array.buffer may be a larger pooled buffer.
  const rawBlob = new Blob(
    [
      resampledSamples.buffer.slice(
        resampledSamples.byteOffset,
        resampledSamples.byteOffset + resampledSamples.byteLength,
      ),
    ],
    { type: "application/octet-stream" },
  )
  const base64Data = await blobToBase64(rawBlob)
  return { rawBlob, base64Data }
}

// Mic capture hook: records raw Float32 PCM via Web Audio, resamples to
// 16 kHz mono, and returns it base64-encoded — the exact payload format
// expected by POST /api/stt (backend/server.py).
// ScriptProcessorNode is deprecated but used deliberately: it needs no
// separately-served AudioWorklet module and is fine for short push-to-talk
// clips on the kiosk's Chromium.
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [micError, setMicError] = useState(null)

  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const sourceRef = useRef(null)
  const scriptProcessorRef = useRef(null)
  const silentGainRef = useRef(null)
  const streamRef = useRef(null)
  const recordedSamplesRef = useRef([])
  // Refs mirror recording state so keyup can stop even if React state is stale
  // (getUserMedia is async; Z can be released before setState flushes).
  const isRecordingRef = useRef(false)
  const startingRef = useRef(false)
  const stopRequestedRef = useRef(false)

  useEffect(() => {
    return () => {
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close()
      }
    }
  }, [])

  const teardownGraph = useCallback(() => {
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect()
      scriptProcessorRef.current.onaudioprocess = null
      scriptProcessorRef.current = null
    }
    if (silentGainRef.current) {
      silentGainRef.current.disconnect()
      silentGainRef.current = null
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
  }, [])

  const finalizeRecording = useCallback(async () => {
    isRecordingRef.current = false
    startingRef.current = false
    setIsRecording(false)

    const actualSampleRate = audioContextRef.current?.sampleRate || 16000
    teardownGraph()

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      await audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (recordedSamplesRef.current.length === 0) {
      console.warn("No audio samples recorded")
      return null
    }

    const mergedSamples = getMergedSamples(recordedSamplesRef.current)
    recordedSamplesRef.current = []
    try {
      return await samplesToPayload(mergedSamples, actualSampleRate)
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return null
    }
  }, [teardownGraph])

  const startRecording = useCallback(async () => {
    if (isRecordingRef.current || startingRef.current) return false
    startingRef.current = true
    stopRequestedRef.current = false
    setMicError(null)
    try {
      // Disable echoCancellation: ScriptProcessor must stay in the audio graph
      // (via a silent gain→destination) to keep callbacks firing, and AEC would
      // treat that loop as echo and zero out the mic.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      })

      if (stopRequestedRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        startingRef.current = false
        return false
      }

      streamRef.current = stream

      const AudioContext = window.AudioContext || window.webkitAudioContext
      audioContextRef.current = new AudioContext()
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume()
      }
      const source = audioContextRef.current.createMediaStreamSource(stream)
      sourceRef.current = source

      // Small FFT — the analyser only feeds the low-res bar visualizer.
      analyserRef.current = audioContextRef.current.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)

      const scriptProcessor = audioContextRef.current.createScriptProcessor(
        4096,
        1,
        1,
      )
      scriptProcessorRef.current = scriptProcessor
      recordedSamplesRef.current = []

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        recordedSamplesRef.current.push(new Float32Array(inputData))
      }

      // Keep the processor alive without playing mic through the speakers
      // (playback would trigger AEC and wipe the capture).
      const silentGain = audioContextRef.current.createGain()
      silentGain.gain.value = 0
      silentGainRef.current = silentGain
      source.connect(scriptProcessor)
      scriptProcessor.connect(silentGain)
      silentGain.connect(audioContextRef.current.destination)

      isRecordingRef.current = true
      startingRef.current = false
      setIsRecording(true)

      // Key released while mic was still opening — finalize whatever we got.
      if (stopRequestedRef.current) {
        return await finalizeRecording()
      }
      return true
    } catch (err) {
      console.error("Error accessing microphone:", err)
      const msg =
        err.message ||
        "Microphone access failed (HTTPS required for remote devices)"
      setMicError(msg)
      startingRef.current = false
      isRecordingRef.current = false
      setIsRecording(false)
      return false
    }
  }, [finalizeRecording])

  const stopRecording = useCallback(async () => {
    // If getUserMedia is still pending, mark stop so startRecording aborts
    // or finalizes as soon as the graph is up.
    if (startingRef.current) {
      stopRequestedRef.current = true
      return null
    }
    if (!isRecordingRef.current) return null
    return finalizeRecording()
  }, [finalizeRecording])

  return {
    isRecording,
    startRecording,
    stopRecording,
    analyser: analyserRef.current,
    micError,
    setMicError,
  }
}
