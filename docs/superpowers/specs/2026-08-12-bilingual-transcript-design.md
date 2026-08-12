# Bilingual conversation transcript

Date: 2026-08-12
Status: approved, ready for implementation

## Problem

The translator shows only the utterance that just happened. `TranslatorApp.jsx`
keeps two state objects, `transcriptionData` and `translationData`, and
overwrites both on every recording. Once the next person speaks, the previous
exchange is gone.

Two people holding a conversation through the device cannot look back at what
was said, and neither of them can read the whole conversation in their own
language — each only ever sees one line at a time.

## Goal

Show the full conversation as two parallel columns, one per language. Every
utterance already produces text in both languages (the transcription and its
translation), so each turn contributes one row to *both* columns. A Swedish
speaker reads the left column top to bottom and follows the entire discussion in
Swedish; an English speaker does the same on the right.

The layout must stop being pinned to the 480x320 kiosk frame and use the space a
desktop browser actually offers, while remaining usable on the Raspberry Pi
display.

## Decisions

Three forks were settled during design:

1. **Responsive, one view** — not a separate desktop mode. The transcript is
   valuable on the Pi too (as a single scrolling column), and two views would
   mean two sources of truth about the same conversation.
2. **Columns lock at the first turn** — the pair of languages in play when the
   conversation starts defines the columns for its lifetime. Changing a lane's
   language mid-conversation is the edge case, not the main flow, and a locked
   header never lies about what the column contains.
3. **The drawer becomes an overlay** — it keeps its large, arm's-length-readable
   text for the moment right after someone speaks, then fades to reveal the
   transcript behind it.

## Architecture

### Data model

`TranslatorApp.jsx` replaces `transcriptionData` / `translationData` with a
single append-only list plus the locked column pair:

```js
const [turns, setTurns] = useState([])
const [columns, setColumns] = useState(null)
```

A turn:

```js
{
  id,          // monotonic counter from a useRef, not a timestamp
  lane,        // 1 | 2 — which lane spoke
  sourceLang,  // language code, e.g. "sv"
  targetLang,  // language code, e.g. "en"
  sourceText,  // transcription; "" until Whisper returns
  targetText,  // translation; "" until Gemma returns
  status,      // "transcribing" | "translating" | "done" | "empty" | "error"
  error,       // string | null
  meta,        // "Duration: 1.2s | Tokens: 48" — as today, or ""
}
```

`columns` is `{ left: {code, name}, right: {code, name} }`, set once when the
first turn is created: `left` is lane 1's language, `right` is lane 2's. Left and
right therefore match the physical lanes, not who happened to speak first.

**The drawer's text is derived, not stored.** `turns[turns.length - 1]` supplies
everything `ResponseDrawer` renders. This removes the duplicate state that exists
today and is why the change makes `TranslatorApp.jsx` simpler rather than larger.

### Data flow

`handleRecordStop` currently sets both state objects to placeholder strings, then
overwrites them as each stage completes. Instead it appends one turn with
`status: "transcribing"` and updates that turn by `id` as the pipeline advances:

```
append turn (transcribing)
  -> Whisper returns    -> sourceText set, status "translating"
     -> empty result    -> status "empty", no translation attempted
  -> Gemma returns      -> targetText set, meta set, status "done"
  -> TTS plays translation (unchanged)
```

Any throw sets `status: "error"` and `error` on that turn. Errors stay attached
to the turn they belong to, so a failed exchange remains visible in the history
instead of blanking the display.

Because turns are updated in place by id, the transcript shows a live row with a
pending indicator rather than having finished rows pop in.

### Rendering a turn into columns

For each turn, per column:

- if `sourceLang` matches the column's language -> render `sourceText`
- else if `targetLang` matches -> render `targetText`
- else -> render nothing (empty cell)

The third case only occurs when someone rotates to a language outside the locked
pair. That turn shows in whichever column it matches and leaves the other empty,
which is the agreed behaviour.

### Components

**New: `frontend/src/components/TranscriptView.jsx`**

Props: `turns`, `columns`. No other responsibilities — it renders history and
owns its scroll behaviour, nothing else.

- **>= 900px**: CSS grid, `grid-template-columns: 1fr 1fr`, sticky language
  headers. Each turn emits two cells in the same grid row so the two languages
  stay aligned line by line. That alignment is the whole point of the feature.
- **< 900px** (including 480x320): one column; each turn stacks source above
  translation with a language tag.
- Speaker attribution: lane number badge, using the existing lane one / lane two
  colours from `style.css`.
- Auto-scroll to the newest turn **only when the user is already at the bottom**.
  If they have scrolled up to read, leave the scroll position alone.
- Empty state before the first turn: the existing "Select languages, push to
  talk" placeholder moves here.

**Changed: `frontend/src/components/ResponseDrawer.jsx`**

Renders above `TranscriptView`. It already opens when recording starts. New
behaviour: it dismisses itself once the exchange has landed — when TTS playback
ends, which is the conversation's natural beat — with a timer fallback (about
4 seconds after `status: "done"`) when TTS is disabled. Escape or a click
dismisses it manually. Dismissal must not move focus, or the window key handlers
stop receiving events.

**Changed: `frontend/src/components/SettingsOverlay.jsx`**

Adds a "Clear conversation" button that resets `turns` **and** `columns`, so a
fresh language pair can lock. It sits with the other action controls and follows
the existing `overlay-btn` styling.

### Layout: unpinning the kiosk frame

`style.css` currently hardcodes `#root { width: 480px; height: 320px }` and
`.translator-envelope { width: 480px; height: 320px }`. The app is locked to the
kiosk rectangle no matter how large the window is; this is the single change that
makes the whole feature possible.

The fixed sizes become the **small-screen** case, and above the breakpoint the
envelope grows to the viewport. The existing `@media (min-width: 481px)` block
already exists for cursor and border tweaks and is the natural place to start;
the two-column transcript needs its own `@media (min-width: 900px)` block.

The Pi renders at exactly 480x320, below both breakpoints, so its layout is
unchanged by construction.

### Persistence

None. Turns live in memory for the session. The kiosk is a walk-up device and a
conversation left on screen for the next stranger is a privacy problem, not a
feature. Reloading the page starts a new conversation.

## Error handling

- **Transcription fails** — turn gets `status: "error"`, `error` holds the
  message; the row renders the error in place of the source text.
- **No speech detected** — `status: "empty"`; the row shows a muted marker rather
  than an empty pair of cells, so the user knows the recording registered.
- **Translation fails** — source text is kept and shown; the target cell carries
  the error. Half an exchange is more useful than none.
- **Column mismatch** (third language) — not an error; empty cell, as designed.

## Testing

The repository has no test infrastructure — `package.json` defines only `dev`,
`build`, and `preview`. Verification is therefore by driving the running app with
Playwright against Chrome, using `--use-fake-device-for-media-stream`:

1. Record two turns from opposite lanes; assert both columns gain two aligned
   rows and that the text lands in the correct column per language.
2. Rotate a lane's language mid-conversation; assert the column headers do not
   change and the new turn fills only its matching column.
3. Assert auto-scroll follows new turns, and that it does **not** when the view
   is scrolled up.
4. Assert the drawer dismisses itself after an exchange completes.
5. Screenshot at 480x320 and 1440x900 and inspect both.

**Known cost:** every turn runs real Whisper, Gemma, and Piper inference and
takes tens of seconds. An earlier scripted attempt at this timed out at four
minutes. Verification runs in the background and is slow; it is not optional.

## Out of scope

- Persisting or exporting conversations
- More than two columns
- Editing or re-playing individual turns
- Touch controls (the build is keyboard-only today)
