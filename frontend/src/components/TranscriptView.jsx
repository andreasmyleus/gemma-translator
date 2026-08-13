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

import React, { useCallback, useLayoutEffect, useRef } from "react"

// How close to the bottom (px) still counts as "stuck to the bottom". Anything
// further up means the user has scrolled back to read and must not be yanked.
const STICK_TOLERANCE = 40

// Which half of a turn belongs in a column: the transcription ("source"), the
// translation ("target"), or nothing at all. The last case is a turn whose
// languages no longer match the current lane pair (e.g. after a rotate) —
// an empty cell keeps the row aligned.
function roleForColumn(turn, code) {
  if (!code) return null
  if (turn.sourceLang === code) return "source"
  if (turn.targetLang === code) return "target"
  return null
}

// An error belongs to the stage that failed: transcription while there is no
// source text yet, translation once the source text has landed.
function errorRole(turn) {
  return turn.sourceText ? "target" : "source"
}

// The conversation history: every turn becomes one grid row with a cell per
// language, so the two columns stay aligned line by line.
export default function TranscriptView({
  turns = [],
  columns = null,
  micError = null,
}) {
  const scrollRef = useRef(null)
  // Starts true so the very first turns scroll into view.
  const stuckToBottomRef = useRef(true)

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    stuckToBottomRef.current = distanceFromBottom <= STICK_TOLERANCE
  }, [])

  // Follow the newest turn only while the user is parked at the bottom. `turns`
  // is a new array both when a turn is appended and when a live turn is updated
  // in place, so a growing row keeps its tail in view too. useLayoutEffect runs
  // after the DOM is patched but before paint, so the jump is never visible.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el || !stuckToBottomRef.current) return
    el.scrollTop = el.scrollHeight
  }, [turns])

  const renderCell = (turn, side, column) => {
    const code = column ? column.code : null
    const role = roleForColumn(turn, code)

    // No matching language: keep the div so the rows stay aligned, but put
    // nothing inside it.
    if (!role) {
      return <div className={`transcript-cell transcript-cell-${side}`} />
    }

    const isSource = role === "source"
    const text = isSource ? turn.sourceText : turn.targetText
    const showError =
      turn.status === "error" && turn.error && errorRole(turn) === role
    // "transcribing" waits on the source half, "translating" on the target one.
    // Cancelled turns must not keep showing listening/translating.
    const showPending =
      !showError &&
      turn.status !== "cancelled" &&
      !text &&
      ((isSource && turn.status === "transcribing") ||
        (!isSource && turn.status === "translating"))
    // No speech detected: mark the source side so the user can tell the
    // recording registered. No translation was attempted, so the target side
    // stays blank.
    const showMuted = turn.status === "empty" && isSource
    // Superseded / aborted before anything landed — muted dash, not pending.
    const showCancelled =
      turn.status === "cancelled" && !text && isSource && !showError

    return (
      <div className={`transcript-cell transcript-cell-${side}`}>
        <span className="transcript-speaker">{turn.lane}</span>
        <span className="transcript-lang-tag">{String(code).toUpperCase()}</span>
        {text ? <span className="transcript-text">{text}</span> : null}
        {showPending ? (
          <span className="transcript-pending">
            {isSource ? "— listening —" : "— translating —"}
          </span>
        ) : null}
        {showMuted ? (
          <span className="transcript-muted">— no speech detected —</span>
        ) : null}
        {showCancelled ? <span className="transcript-muted">—</span> : null}
        {showError ? (
          <span className="transcript-error">{turn.error}</span>
        ) : null}
      </div>
    )
  }

  return (
    <div className="transcript-view">
      {/* Headers track the current lane languages. */}
      {columns ? (
        <div className="transcript-headers">
          <div className="transcript-header transcript-header-left">
            {columns.left.name}
          </div>
          <div className="transcript-header transcript-header-right">
            {columns.right.name}
          </div>
        </div>
      ) : null}
      <div
        className="transcript-scroll"
        ref={scrollRef}
        onScroll={handleScroll}
      >
        <div className="transcript-grid">
          {turns.map((turn) => (
            <div
              key={turn.id}
              className={`transcript-row transcript-row-lane${turn.lane === 2 ? "2" : "1"} transcript-row-${turn.status}`}
            >
              {renderCell(turn, "left", columns ? columns.left : null)}
              {renderCell(turn, "right", columns ? columns.right : null)}
            </div>
          ))}
        </div>
      </div>
      {micError ? (
        <div className="transcript-error-banner">
          Microphone: {micError} (HTTPS is required from remote devices)
        </div>
      ) : null}
      {turns.length === 0 && !micError ? (
        <div className="transcript-empty">Select languages, push to talk</div>
      ) : null}
    </div>
  )
}
