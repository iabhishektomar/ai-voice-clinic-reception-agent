# LiveKit Inbound Voice Agent

An AI-powered phone agent that answers inbound calls, converses naturally using a fully streaming pipeline, and logs every call to Airtable.

**Stack:** LiveKit Agents · Deepgram STT · OpenAI GPT · ElevenLabs TTS · Silero VAD · Twilio SIP · Airtable

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every value before running.

| Variable | Description | Where to get it |
|---|---|---|
| `LIVEKIT_URL` | Your LiveKit Cloud WebSocket URL | [cloud.livekit.io](https://cloud.livekit.io) → Project → Settings → API Keys |
| `LIVEKIT_API_KEY` | LiveKit API Key | Same page as above |
| `LIVEKIT_API_SECRET` | LiveKit API Secret | Same page as above |
| `DEEPGRAM_API_KEY` | Deepgram API key for Speech-to-Text | [console.deepgram.com](https://console.deepgram.com) → API Keys |
| `OPENAI_API_KEY` | OpenAI API key for the LLM | [platform.openai.com](https://platform.openai.com) → API Keys |
| `OPENAI_MODEL` | OpenAI model name (default: `gpt-4o-mini`) | `gpt-4o-mini` (fast/cheap) or `gpt-4o` (smarter) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for Text-to-Speech | [elevenlabs.io](https://elevenlabs.io) → Profile → API Key |
| `ELEVENLABS_VOICE_ID` | Voice ID to use for the agent | [elevenlabs.io/voice-library](https://elevenlabs.io/voice-library) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token (for webhook validation) | [console.twilio.com](https://console.twilio.com) → Account → API Keys |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Same page, starts with `AC...` |
| `AIRTABLE_PAT` | Airtable Personal Access Token | [airtable.com/create/tokens](https://airtable.com/create/tokens) — scopes: `data.records:write`, `schema.bases:read` |
| `AIRTABLE_BASE_ID` | Airtable Base ID | From the URL: `https://airtable.com/appXXXXXXXXXX/...` |
| `AIRTABLE_TABLE_NAME` | Airtable table name (default: `call_logs`) | Must match the table name in your base |
| `MAX_CALL_DURATION_SECONDS` | Hard cap on call length in seconds (default: `600`) | Set based on your cost tolerance |
| `AGENT_SYSTEM_PROMPT` | The agent's personality / instructions | Free text — edit to match your use case |

---

## Airtable Setup

Create a table called `call_logs` (or whatever you set `AIRTABLE_TABLE_NAME` to) with these exact fields:

| Field Name | Field Type |
|---|---|
| `caller_number` | Single line text |
| `duration_seconds` | Number |
| `transcript` | Long text |
| `created_at` | Date (ISO 8601 format) |

---

## ✅ Pre-Launch Setup Checklist

Work through every item below before starting the agent.

### 1 · LiveKit Cloud
- [ ] Create an account at [cloud.livekit.io](https://cloud.livekit.io)
- [ ] Create a new project
- [ ] Copy the **WebSocket URL**, **API Key**, and **API Secret** into `.env`
- [ ] Enable SIP on your project (LiveKit Dashboard → SIP)
- [ ] Create a **SIP Trunk** pointing to Twilio (see LiveKit docs for Twilio SIP)
- [ ] Create a **Dispatch Rule** so that inbound SIP calls are routed to an agent room

### 2 · Twilio
- [ ] Create a Twilio account at [twilio.com](https://twilio.com)
- [ ] Buy a phone number that supports voice
- [ ] Configure the number's Voice settings to use a **SIP Trunk** or **TwiML App** pointing at LiveKit's SIP endpoint
- [ ] Copy your **Account SID** and **Auth Token** into `.env`

### 3 · Deepgram
- [ ] Create a Deepgram account at [deepgram.com](https://deepgram.com)
- [ ] Create an API key with **Speech** permissions
- [ ] Copy the key into `.env` as `DEEPGRAM_API_KEY`

### 4 · OpenAI
- [ ] Create an account at [platform.openai.com](https://platform.openai.com)
- [ ] Generate an API key
- [ ] Copy the key into `.env` as `OPENAI_API_KEY`
- [ ] Ensure your account has billing enabled (add a payment method)

### 5 · ElevenLabs
- [ ] Create an account at [elevenlabs.io](https://elevenlabs.io)
- [ ] Find or create a voice — copy its **Voice ID** into `.env`
- [ ] Copy your API key from Profile → API Key into `.env`

### 6 · Airtable
- [ ] Create a base with a `call_logs` table using the fields listed above
- [ ] Create a **Personal Access Token** at [airtable.com/create/tokens](https://airtable.com/create/tokens) with scopes `data.records:write` and `schema.bases:read`
- [ ] Copy the PAT and Base ID into `.env`

### 7 · VPS / Server Setup
- [ ] Install Python 3.10+ on your VPS (`python3 --version`)
- [ ] Clone this repository to your VPS
- [ ] Create and activate a virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- [ ] Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Copy `.env.example` to `.env` and fill in all values:
  ```bash
  cp .env.example .env
  nano .env
  ```
- [ ] Download the Silero VAD model (runs automatically on first start, or manually):
  ```bash
  python -c "from livekit.plugins import silero; silero.VAD.load()"
  ```

### 8 · Test Locally
```bash
python agent.py dev
```
This connects to LiveKit Cloud in development mode. You can test using a LiveKit SDK sample app or the Sandbox in the LiveKit Dashboard.

### 9 · Run in Production (VPS)
Start the agent as a persistent background process. Three options:

**Option A — systemd (recommended)**

Create `/etc/systemd/system/voice-agent.service`:
```ini
[Unit]
Description=LiveKit Voice Agent
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/AI Voice Agent
EnvironmentFile=/path/to/AI Voice Agent/.env
ExecStart=/path/to/AI Voice Agent/venv/bin/python agent.py start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-agent
sudo systemctl start voice-agent
sudo journalctl -fu voice-agent   # view logs
```

**Option B — tmux**
```bash
tmux new-session -d -s agent 'source venv/bin/activate && python agent.py start'
tmux attach -t agent   # re-attach to view logs
```

**Option C — screen**
```bash
screen -S agent
source venv/bin/activate
python agent.py start
# Ctrl+A D to detach
```

---

## Architecture Overview

```
Inbound Call (PSTN)
       │
   Twilio SIP
       │
LiveKit Cloud (SIP Trunk + Dispatch Rule)
       │
   LiveKit Room
       │
   agent.py (VoicePipelineAgent)
   ┌───────────────────────────────────────┐
   │  Silero VAD → detects speech/silence  │
   │       ↓                               │
   │  Deepgram STT → streaming transcript  │
   │       ↓                               │
   │  OpenAI LLM → streaming tokens       │
   │       ↓                               │
   │  ElevenLabs TTS → streaming audio    │
   └───────────────────────────────────────┘
       │
   Call ends → Airtable call log saved
```

---

## Tuning VAD for Your Use Case

All Silero VAD parameters are in `agent.py` → `prewarm()` function with detailed comments explaining each parameter and the effect of changing it.

Key parameters to adjust if calls feel off:
- **Agent cuts caller off** → increase `min_silence_duration` (e.g., 0.9)
- **Too much silence before agent responds** → decrease `min_silence_duration` (e.g., 0.5)
- **Background noise causes false starts** → increase `min_speech_duration` or `activation_threshold`

---

## Cost Protection

Set `MAX_CALL_DURATION_SECONDS` in `.env` to the maximum number of seconds any single call should last. The agent will politely say goodbye and hang up after this limit, preventing stuck calls from consuming unbounded API credits.

Default: `600` (10 minutes).

---

## Security Notes

- All API keys are loaded from `.env` — nothing is hardcoded in source files.
- `.env` is listed in `.gitignore` and will never be committed.
- Twilio webhook validation is implemented via `validate_twilio_request()` in `agent.py`. When deploying behind a web framework (e.g., FastAPI), pass the URL, POST params, and `X-Twilio-Signature` header to this function and reject requests that fail validation.
