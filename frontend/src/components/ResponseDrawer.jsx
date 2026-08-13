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

import React from "react"

// Result panel: transcription (left bubble) and translation (right bubble)
// in a chat-style layout, with a first-run placeholder before any recording.
// It sits as an overlay above the transcript, so it can be dismissed by
// pressing Escape or clicking anywhere on the panel. Auto-dismissal is not
// handled here — TranslatorApp owns TTS playback and therefore the timing.
export default function ResponseDrawer({
  isActive,
  onClose,
  transcriptionSource,
  transcriptionText,
  translationTarget,
  translationText,
  metaText,
  placeholderText,
}) {
  // Determine if we have any data to show (or if the drawer was explicitly activated)
  const hasData = isActive || transcriptionSource !== ""

  // Escape dismisses the overlay. Registered on window (not on the panel) so
  // it works without the drawer ever holding focus — TranslatorApp's
  // push-to-talk handlers live on window too and must keep receiving keys.
  React.useEffect(() => {
    if (!isActive) return
    const handleKeyDown = (e) => {
      if (e.key !== "Escape") return
      if (typeof onClose === "function") onClose()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isActive, onClose])

  const handleDismiss = () => {
    if (!isActive) return
    if (typeof onClose === "function") onClose()
  }

  // Swallow the mousedown so the click never moves focus (or starts a text
  // selection). If focus landed on a control here, the next Space press would
  // re-activate that control instead of starting a recording.
  const handleMouseDown = (e) => {
    e.preventDefault()
  }

  return (
    <div
      className={`response-drawer ${isActive ? "active" : ""}`}
      id="response-drawer"
      onClick={handleDismiss}
      onMouseDown={handleMouseDown}
    >
      <div className="app-title">GEMLANG-1</div>
      <div
        className="drawer-handle"
        onClick={onClose}
        style={{ display: "none" }}
      ></div>
      <div className="chat-panel">
        {!hasData ? (
          <div className="initial-placeholder">
            {placeholderText || "Select languages, push to talk"}
          </div>
        ) : (
          <>
            <div className="chat-row chat-row-left">
              <div className="chat-bubble bubble-left">
                <div className="bubble-label">
                  {transcriptionSource || "LANG 1"}
                </div>
                <div className="bubble-text">
                  {transcriptionText || "— listening —"}
                </div>
              </div>
            </div>
            <div className="chat-row chat-row-right">
              <div className="chat-bubble bubble-right">
                <div className="bubble-label">
                  {translationTarget || "LANG 2"}
                </div>
                <div className="bubble-text">
                  {translationText || "— waiting —"}
                </div>
              </div>
            </div>
            <div className="chat-meta">{metaText}</div>
          </>
        )}
      </div>
    </div>
  )
}
