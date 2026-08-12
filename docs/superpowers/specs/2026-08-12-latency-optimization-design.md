# Latensoptimering av översättningskedjan — design

**Datum**: 2026-08-12
**Status**: Godkänd, redo för implementationsplan

## Problem

En push-to-talk-runda tar idag ~7 s. Kedjan är:

```
keyup → resample+base64 → POST /api/stt (faster-whisper small, int8, beam 1)
      → POST /proxy → LiteRT-LM gemma4-e2b (icke-strömmande, JSON-wrapper)
      → GET /api/tts × N (Piper, 180-teckens chunkar) → uppspelning
```

README:s avsnitt "Latency notes & ideas" listar åtta idéer, men **inget är
implementerat** och det finns ingen mätning i koden — bara ett `Date.now()`
runt LLM-anropet i `frontend/src/utils/api.js`. Vi kan alltså varken visa att
en optimering hjälpte eller att den kostade kvalitet.

## Mål

Bygg en mät-harness, ta en baseline, och genomför sedan en serie
optimeringar där **varje enskild förbättring mäts** mot baseline med både
latens och kvalitet.

**Huvudmått**: tid till första ljud (upplevd latens). Strömning räknas som
vinst även om total wall-clock är oförändrad.

**Mätmiljö**: MacBook Air M1, 8 kärnor. Pi 5 ligger utanför scope för den
här omgången.

## Beslut fattade under brainstormingen

| Fråga | Beslut |
| :--- | :--- |
| Var mäta? | Bara på Macen |
| Huvudmått? | Tid till första ljud |
| Kvalitetsbevakning? | Automatisk grind: WER mot facit + diff av översättningen |
| Testljud? | Syntetiserat med Piper ur en textfil — deterministiskt, facit gratis |
| Körmiljö? | Eget venv i den här workspacen, men återanvänd litert-lm på 9379 |
| Mätryggrad? | Headless HTTP-bench + tunn frontend-instrumentering |

## Arkitektur

Två komponenter, tydligt åtskilda:

1. **`bench/`** — fristående mät-harness som talar HTTP med den riktiga
   backenden. Vet inget om frontend. Källa till alla siffror vi styr efter.
2. **Frontend-instrumentering** — ~20 rader `performance.mark` i den
   befintliga appen. Enda stället där "tid till första ljud" är sann på
   riktigt (inkluderar autoplay-fördröjning). Körs manuellt vid milstolpar.

Avvisat: in-process microbench som importerar `transcribe()`/`synthesize()`
direkt. Det skulle missa exakt de lager vi vill åt — proxyns hel-buffring och
`_tts_lock`-serialiseringen är latenskällor som ett microbench visar som noll.

### Komponent 1: `bench/bench.py`

**Ansvar**: kör N fixturer genom hela HTTP-kedjan, mät per steg, jämför mot
en tidigare körning, fäll om kvaliteten sjunkit.

**Beroenden**: en körande backend (startas av bench på egen port) och en
körande litert-lm på 9379. Inga nya Python-paket — Levenshtein skrivs för
hand (~30 rader), allt annat finns redan i `backend/requirements.txt`.

**Gränssnitt**:

```bash
python bench/bench.py --label baseline
python bench/bench.py --label whisper-vad --compare baseline
```

Flaggor: `--label` (obligatorisk, namnger körningen), `--compare` (label att
diffa mot), `--api-port` (default 3100), `--llm-port` (default 9379),
`--repeats` (default 3), `--langs` (default sv,fi,en).

#### Fixturer

`bench/fixtures.json` — meningar per språk i tre längder:

```json
{
  "sv-short": {
    "lang": "sv", "target": "en",
    "text": "Var ligger stationen?"
  },
  "sv-medium": {
    "lang": "sv", "target": "en",
    "text": "Ursäkta, kan du berätta hur jag tar mig till stationen härifrån?"
  },
  "sv-long": {
    "lang": "sv", "target": "en",
    "text": "Jag skulle behöva boka ett rum för två nätter, helst med utsikt mot havet, och jag undrar också om frukosten ingår i priset eller om den kostar extra."
  }
}
```

Målet är ~1 s, ~3 s och ~8 s syntetiserat tal per språk (sv, fi, en) — nio
fixturer. Texten *är* WER-facit.

Bench syntetiserar varje fixtur med Piper (samma röstkarta som backenden
använder), resamplar till 16 kHz mono med `np.interp`, och cachar till
`bench/fixtures/<id>.wav`. `*.wav` är redan gitignorerat. Saknas filen
genereras den; finns den återanvänds den. Efter generering assertar bench att
varje klipps längd ligger inom ±50 % av sin målängd — annars har någon
redigerat texten till fel storlek och mätningen skulle bli missvisande.

#### Mätpunkter

Per fixtur och repetition:

| Mått | Från | Till |
| :--- | :--- | :--- |
| `stt_ms` | POST /api/stt skickas | svaret läst |
| `llm_ms` | POST /proxy skickas | svaret läst |
| `tts_first_ms` | GET /api/tts (chunk 1) | svaret läst |
| `tts_rest_ms` | chunk 2 | sista chunken läst |

Härledda:

- `time_to_first_audio = stt_ms + llm_ms + tts_first_ms` ← **huvudsiffran**
- `wall_total = stt_ms + llm_ms + tts_first_ms + tts_rest_ms`

Bench delar texten i TTS-chunkar med samma logik som frontend, så
chunkningsändringar (optimering 4) syns direkt i `tts_first_ms`. Logiken bor
i JS (`splitTextIntoSpeechChunks` i `frontend/src/utils/api.js`) och måste
alltså finnas i en Python-kopia i bench. Det är en känd dubblering: när
optimering 4 ändrar chunkningen måste båda ändras i samma commit, annars
mäter bench något annat än vad appen gör.

Tre repetitioner per fixtur; den första kastas som uppvärmning och medianen
av resten rapporteras, tillsammans med min/max så vi ser spridningen.

#### Kvalitetsgrind

- **WER**: ordvis Levenshtein mellan transkription och fixturtext.
  Normalisering före jämförelse: gemener, skiljetecken bort, whitespace
  kollapsad.
- **Översättning**: strängen sparas per fixtur och jämförs mot baseline.

Utfall (bara när `--compare` angetts; utan den rapporteras WER som ren siffra
och ingen grind körs):

- WER stiger mer än 2 procentenheter absolut mot jämförelsekörningen → rött,
  exit nonzero.
- Översättningen skiljer sig från jämförelsekörningen → gult, flaggas för
  manuell granskning men fäller inte körningen.

Två procentenheter är vald som tröskel för att rymma normal variation mellan
körningar utan att släppa igenom en verklig försämring; den justeras om
baseline visar sig brusigare än så.

#### Utdata

Markdown-tabell till stdout med en rad per fixtur, kolumner för varje
mätpunkt, delta i procent mot jämförelsekörningen, och WER. Under den en
sammanfattningsrad med median `time_to_first_audio` över alla fixturer.

`bench/results/<label>.json` skrivs och **committas** — filerna är små och ger
en historik över hela kampanjen.

#### Felhantering

HTTP-fel på en fixtur loggas som en misslyckad rad, körningen fortsätter, och
scriptet avslutar nonzero. En trasig fixtur ska aldrig dölja siffrorna för de
övriga åtta.

### Komponent 2: Frontend-instrumentering

`performance.mark` på fyra punkter: keyup (i `handleRecordStop`), STT klar,
LLM klar, och `player.onplaying` för första chunken. En konsolrad per runda:

```
[latency] keyup→stt 980ms | →llm 3400ms | →first audio 3600ms
```

Samma siffror ersätter drawerns nuvarande `metaText` (`Duration | Tokens`).

Körs manuellt vid tre milstolpar — baseline, efter STT-jobbet, efter
strömningen — inte per patch.

## Ändringar i befintlig kod

`backend/server.py` läser `PORT` från miljön med 3000 som default. Krävs för
att bench ska kunna köra en egen backend på 3100 parallellt med en annan
workspace-stack på 3000. Enradig ändring, inget beteende ändras utan env-var.

## Optimeringarna

Varje punkt är en commit plus en bench-körning. Ordningen är vald så att
mätningen blir stabil innan vi börjar jaga millisekunder.

### Vad mätning mot den körande litert-lm faktiskt visade

Innan planen skrevs sonderades API:t på 9379. Fyra fynd som ändrar designen:

1. **`max_tokens` ignoreras.** `max_tokens: 5` på "räkna till 50" gav hela
   uppräkningen. Det tänkta taket i steg 0 är alltså en no-op och stryks.
2. **`usage` saknas i svaret** (`null`). Frontendens `Tokens: N` i drawern
   visar därför alltid 0 idag, och bench kan inte mäta tokenantal via API:t —
   outputlängd mäts i tecken i stället.
3. **Strömning fungerar.** `stream: true` ger SSE med
   `chat.completion.chunk` och `delta.content`. Optimering 8 är genomförbar.
4. **Det finns en GPU-variant**: modell-ID `gemma4-e2b,gpu` vid sidan av
   `gemma4-e2b`. Uppmätt på denna Mac med kort systemprompt: 1126–1660 ms på
   CPU mot 579 ms varm på GPU, men 7344 ms på första anropet (modelladdning).
   Det är ungefär en halvering av det dyraste steget och läggs till som
   optimering 9.

Med identisk prompt gav CPU-varianten ordagrant samma svar två körningar i
rad, så genereringen verkar redan vara greedy. Steg 0 krymper därmed till
`temperature: 0` för säkerhets skull, och variansen hanteras i stället av
bench (median av tre, med min/max redovisat).

### Baseline

Full bench-körning med `--label baseline`. Alla siffror härefter jämförs mot
den.

### Optimeringar i ordning

| # | Steg | Åtgärd | Förväntad effekt |
| :-- | :--- | :--- | :--- |
| 1 | STT | `vad_filter=True` | Kortare ljud in i modellen, och hallucinationer på tysta klipp försvinner |
| 2 | STT | Kort mel-padding (ljudlängd + 5 s) i stället för fasta 30 s, bakom env-flagga | Största enskilda STT-vinsten |
| 3 | STT | `cpu_threads=8`, `without_timestamps=True`, `condition_on_previous_text=False` | ~0,2 s, stabilare korta klipp |
| 4 | TTS | Meningsvis chunkning i stället för 180 tecken | Första chunken blir kort → första ljudet mycket tidigare |
| 5 | LLM | Slopa JSON-wrappern, korta systemprompten | Färre prefill- och output-tokens per anrop |
| 6 | TTS | Prefetcha chunk N+1 medan N spelar | Tar bort glappen mellan chunkar |
| 7 | Klient | 16 kHz direkt från `getUserMedia`; ingen `await close()` på kritiska vägen; binär PCM i stället för base64 | Små men gratis; base64 kostar 33 % extra payload |
| 8 | LLM | `stream: true` genom en icke-buffrande proxy → mening 1 till TTS medan resten genereras | Störst upplevd vinst |
| 9 | LLM | Modell-ID `gemma4-e2b,gpu` i stället för `gemma4-e2b`, med förvärmning vid start | Halverar LLM-steget på Mac (579 ms mot 1126–1660 ms uppmätt) |

**Om punkt 9**: GPU-varianten kostar 7,3 s på första anropet medan modellen
laddas, så den kräver ett förvärmningsanrop vid uppstart för att inte straffa
första översättningen. Den är dessutom Mac-specifik och ska **inte** göras
till default i repot utan att ha verifierats på Pi 5 — den läggs in som ett
val i inställningarna och mäts, inte som ny standard.

**Om punkt 4**: idag är första TTS-chunken upp till 180 tecken, vilket för en
normal översättning är *hela texten*. Första ljudet väntar alltså på att allt
syntetiserats. Meningsvis chunkning kan ge nästan lika mycket upplevd vinst
som strömningen, till en bråkdel av jobbet.

**Om punkt 8**: `handle_proxy` gör `response.read()` på hela svaret. Strömning
kräver att proxyn skriver vidare chunk för chunk, och att `translateText`
läser SSE i stället för en JSON-body. Detta är den enda punkten som rör
arkitekturen.

**Om punkt 2**: kort padding läggs bakom en env-flagga med 30 s som default
tills mätningen visar att WER håller. Övriga punkter är säkra nog att gå in
direkt, med kvalitetsgrinden som skyddsnät.

### Utanför scope

Att strömma mic-ljud till STT under inspelningen skulle nästan radera STT
från kritiska vägen, men är en betydligt större ombyggnad. Tas upp igen bara
om siffrorna efter punkt 8 fortfarande skaver.

Verifiering på Pi 5. Alla siffror i den här omgången gäller Macen.

Uppdatering av README:s "Latency notes & ideas" med de uppmätta resultaten —
görs när kampanjen är klar, inte per patch.

## Verifiering

Repot har ingen testsvit och vi inför ingen. Bench-scriptet *är*
verifieringen: det kör den riktiga kedjan, mäter, och fäller på
kvalitetsregression. Utöver det en manuell browserkontroll vid de tre
milstolparna.
