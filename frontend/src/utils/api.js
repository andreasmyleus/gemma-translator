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

// API client for the Python backend (backend/server.py). LLM calls are
// routed through the backend's /proxy by default so the kiosk stays
// same-origin (no CORS) and works fully offline.

// Normalize a user-entered endpoint into an OpenAI-compatible ".../v1" base.
export function getNormalizedBaseUrl(endpointUrl) {
  let url = endpointUrl.trim()
  if (!url) return "http://localhost:9379/v1"
  url = url.replace(/\/+$/, "")
  if (!url.endsWith("/v1")) {
    url += "/v1"
  }
  return url
}

// Cheap connectivity probe: GET {base}/v1/models.
export async function testConnectionAPI(endpointUrl, useProxy, apiKey) {
  const baseUrl = getNormalizedBaseUrl(endpointUrl)
  const targetUrl = `${baseUrl}/models`

  const headers = {}
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`
  }

  const fetchUrl = useProxy
    ? `/proxy?url=${encodeURIComponent(targetUrl)}`
    : targetUrl

  const response = await fetch(fetchUrl, { method: "GET", headers })
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }
  return true
}

// POST base64 Float32 PCM (16 kHz mono) to the local Whisper STT.
// Returns { text, language }. `language` is the resolved source code (lane
// prior, unless auto-detect is confident the other lane was spoken).
export async function transcribeAudio(
  base64Data,
  sourceLangCode,
  { otherLanguage, autoLanguage = true, signal } = {},
) {
  const response = await fetch("/api/stt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_base64: base64Data,
      language: sourceLangCode,
      other_language: otherLanguage || undefined,
      auto_language: !!autoLanguage,
    }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`STT failed: ${response.status}`)
  }

  const sttData = await response.json()
  return {
    text: sttData.text || "",
    language: sttData.language || sourceLangCode,
  }
}

function generatePayloadJSON(transcribedText, model, systemPrompt) {
  const messages = []
  if (systemPrompt && systemPrompt.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() })
  }
  messages.push({
    role: "user",
    content: transcribedText,
  })

  return JSON.stringify({ model: model || "gemma4-e2b", messages })
}

// Tolerates ``` fences and falls back to the raw reply when the model didn't
// return the JSON envelope the prompt asked for.
function parseTranslation(modelResponse) {
  let cleanJson = modelResponse.trim()
  if (cleanJson.startsWith("```json")) cleanJson = cleanJson.slice(7)
  if (cleanJson.startsWith("```")) cleanJson = cleanJson.slice(3)
  if (cleanJson.endsWith("```")) cleanJson = cleanJson.slice(0, -3)
  try {
    return JSON.parse(cleanJson.trim()).translation || ""
  } catch (e) {
    return modelResponse
  }
}

// Chat-completions request. The system prompt asks for a bare translation
// (no JSON wrapper) to keep prefill and output short, but we still tolerate
// a legacy {"translation": ...} reply (``` fences included) and fall back to
// the raw reply text if parsing fails. The prompt is user-editable in
// Settings, so that tolerance is reachable without a code change — see
// translateTextStreaming for how the streaming path honours it.
export async function translateText(transcribedText, config) {
  const { endpointUrl, useProxy, apiKey, modelName, systemPrompt } = config
  const baseUrl = getNormalizedBaseUrl(endpointUrl)
  const targetUrl = `${baseUrl}/chat/completions`
  const payload = generatePayloadJSON(transcribedText, modelName, systemPrompt)

  const headers = { "Content-Type": "application/json" }
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`
  }

  const fetchUrl = useProxy
    ? `/proxy?url=${encodeURIComponent(targetUrl)}`
    : targetUrl
  const startRequestTime = Date.now()

  const response = await fetch(fetchUrl, {
    method: "POST",
    headers,
    body: payload,
  })
  const requestDuration = ((Date.now() - startRequestTime) / 1000).toFixed(2)

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(
      `API ${response.status}: ${errorText || response.statusText}`,
    )
  }

  const data = await response.json()
  let modelResponse = ""
  if (data.choices && data.choices[0] && data.choices[0].message) {
    modelResponse = data.choices[0].message.content || ""
  } else {
    modelResponse = JSON.stringify(data, null, 2)
  }

  const translationVal = parseTranslation(modelResponse)

  return {
    translation: translationVal,
    duration: requestDuration,
    tokens: data.usage?.total_tokens || 0,
  }
}

// Streaming chat-completions. Calls `onText(fullTextSoFar)` after every delta
// so the caller can start speaking sentence one while the rest generates.
//
// `onText` receives text that is safe to show and speak. A half-received
// legacy envelope (`{"translation": "Hej.`) is neither: it cannot be parsed
// yet, and speaking it verbatim is exactly what the tolerance in
// `parseTranslation` exists to avoid. So when the reply opens like an
// envelope, partials are suppressed entirely and the caller gets the parsed
// text once, in the return value. `raw` is the unparsed accumulated text —
// the caller can compare it with `translation` to tell whether the offsets it
// accumulated while streaming still address the same string.
//
// Pass `signal` (an AbortSignal) to stop a superseded translation.
export async function translateTextStreaming(
  transcribedText,
  config,
  onText,
  signal,
) {
  const { endpointUrl, useProxy, apiKey, modelName, systemPrompt } = config
  const targetUrl = `${getNormalizedBaseUrl(endpointUrl)}/chat/completions`
  const payload = JSON.parse(
    generatePayloadJSON(transcribedText, modelName, systemPrompt),
  )
  payload.stream = true

  const headers = { "Content-Type": "application/json" }
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`
  }
  const fetchUrl = useProxy
    ? `/proxy?url=${encodeURIComponent(targetUrl)}`
    : targetUrl

  const startRequestTime = Date.now()
  const response = await fetch(fetchUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`API ${response.status}: ${errorText || response.statusText}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let text = ""
  // null until the first non-blank character decides it.
  let isEnvelope = null

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // Keep the last, possibly incomplete, line in the buffer.
    const lines = buffer.split("\n")
    buffer = lines.pop()
    for (const line of lines) {
      if (!line.startsWith("data:")) continue
      const data = line.slice(5).trim()
      if (!data || data === "[DONE]") continue
      try {
        const delta = JSON.parse(data).choices?.[0]?.delta?.content
        if (delta) {
          text += delta
          const head = text.trimStart()
          if (isEnvelope === null && head.length > 0) {
            isEnvelope = head.startsWith("{") || head.startsWith("```")
          }
          if (isEnvelope === false && onText) onText(text)
        }
      } catch (e) {
        // A malformed line is not worth aborting the stream over.
      }
    }
  }

  return {
    translation: parseTranslation(text),
    raw: text,
    duration: ((Date.now() - startRequestTime) / 1000).toFixed(2),
  }
}

// True when `text` contains at least one letter. Punctuation-only strings
// ("." "…" "?") are Whisper/Gemma artefacts, not speech — Piper also refuses
// them (empty chunk list → 500).
export function isSpeakable(text) {
  return /\p{L}/u.test(text || "")
}

// Whisper sometimes emits special tokens (`<|nospeech|>`) plus a leftover
// period. Those contain letters so isSpeakable would let them through to
// Gemma/TTS unless they are stripped first.
export function normalizeSttText(text) {
  return (text || "")
    .replace(/<\|[^|]*\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
}

// Same speaker resumes within this window → glue onto the open turn.
export const CONTINUE_WINDOW_MS = 1500
// After that, still fold in "nej, jag menade…" as a repair of the last row.
export const REPAIR_WINDOW_MS = 4000
// Extra VAD silence after a trailing conjunction ("och", "and", …).
export const CONTINUATION_HOLD_MS = 1400

const BACKCHANNEL_RE =
  /^(m+h?m+|uh+|u+m+|öh+|äh+|eh+|euh+|hmm+|huh+|tsk+)$/iu

export function isBackchannel(text) {
  const t = (text || "").trim().replace(/[.,!?…]+$/g, "")
  return t.length > 0 && BACKCHANNEL_RE.test(t)
}

// Only conjunctions people actually trail off on. "så"/"that"/"att" are
// ordinary sentence endings and must not delay translation.
const CONTINUATION_CUES = new Set([
  "och",
  "and",
  "men",
  "but",
  "or",
  "eller",
  "y",
  "et",
  "mais",
])

export function endsWithContinuationCue(text) {
  const words = (text || "")
    .trim()
    .replace(/[,:]+$/g, "")
    .split(/\s+/)
    .filter(Boolean)
  if (words.length === 0) return false
  const last = words[words.length - 1].toLowerCase().replace(/[.,!?…]+$/g, "")
  return CONTINUATION_CUES.has(last)
}

const REPAIR_CUE_RE =
  /^(nej|no|non|wait|alltså|sorry|jag menade|i mean|en fait|o sea)\b[,.!?]*\s*/iu

export function isRepairUtterance(text) {
  return REPAIR_CUE_RE.test((text || "").trim())
}

export function stripRepairCue(text) {
  const t = (text || "").trim()
  const rest = t.replace(REPAIR_CUE_RE, "").trim()
  return rest.length < 3 ? t : rest
}

export function langByCode(languages, code) {
  return languages.find((l) => l.code === code) || null
}

// After STT, put the turn on the lane whose language was actually spoken
// and translate into the other. Enter is only a prior — two people can
// just talk in the two chosen languages.
export function routeSpokenTurn(detectedCode, activeLane, src, dst, _lang1, _lang2) {
  const code = (detectedCode || "").split("-")[0].toLowerCase()
  if (code && code === dst.code) {
    return {
      lane: activeLane === 1 ? 2 : 1,
      src: dst,
      dst: src,
      flipped: true,
    }
  }
  return { lane: activeLane, src, dst, flipped: false }
}

// Word-safe chunking so each /api/tts request stays under ~`limit` chars.
export function splitTextIntoSpeechChunks(text, limit = 180) {
  if (!isSpeakable(text)) return []
  const words = text.split(/\s+/)
  const chunks = []
  let currentChunk = ""
  for (const word of words) {
    if ((currentChunk + " " + word).trim().length <= limit) {
      currentChunk = (currentChunk + " " + word).trim()
    } else {
      if (currentChunk) chunks.push(currentChunk)
      currentChunk = word
    }
  }
  if (currentChunk) chunks.push(currentChunk)
  return chunks.filter(isSpeakable)
}

