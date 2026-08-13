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
export async function transcribeAudio(base64Data, sourceLangCode) {
  const response = await fetch("/api/stt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_base64: base64Data,
      language: sourceLangCode,
    }),
  })

  if (!response.ok) {
    throw new Error(`STT failed: ${response.status}`)
  }

  const sttData = await response.json()
  return sttData.text || ""
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

// Word-safe chunking so each /api/tts request stays under ~`limit` chars.
export function splitTextIntoSpeechChunks(text, limit = 180) {
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
  return chunks
}

