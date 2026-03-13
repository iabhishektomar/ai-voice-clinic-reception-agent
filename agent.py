"""
=============================================================================
  LiveKit Inbound Voice Agent
=============================================================================
  A self-hosted AI phone agent that:
    1. Answers inbound calls routed through Twilio SIP → LiveKit Cloud
    2. Runs a fully streaming STT → LLM → TTS voice pipeline
    3. Logs every call (transcript, duration, caller number) to Airtable

  Stack: LiveKit Agents v1.4 | Deepgram STT | Google Gemini LLM | Cartesia TTS
         Silero VAD | Twilio SIP | Airtable

  Usage:
    python agent.py dev        # local development (connects to LiveKit Cloud)
    python agent.py start      # production mode (long-running worker process)
=============================================================================
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def load_dotenv() -> None:
    """Read a .env file and set its values as environment variables (stdlib only)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:          # don't overwrite existing vars
            os.environ[key] = value


# Load environment variables before any imports that might need them
load_dotenv()

# ── Fix macOS SSL certificate verification ────────────────────────────────────
# Python installed from python.org on macOS doesn't include root CA certs by
# default.  We use certifi's CA bundle so aiohttp / websockets can connect
# to LiveKit Cloud, Deepgram, ElevenLabs, Google, etc. over TLS.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass  # certifi not installed — assume system certs are available

# ── LiveKit Agents core ──────────────────────────────────────────────────────
from livekit import agents, api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    metrics,
    room_io,
)

# ── LiveKit plugins ──────────────────────────────────────────────────────────
from livekit.plugins import cartesia, deepgram, google, silero

# ── Airtable ─────────────────────────────────────────────────────────────────
from pyairtable import Api as AirtableApi

# ── Twilio webhook validation ─────────────────────────────────────────────────
from twilio.request_validator import RequestValidator

# =============================================================================
#  CONFIGURATION — all values come from environment variables
# =============================================================================

# ── LiveKit Cloud credentials ─────────────────────────────────────────────────
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# ── Third-party API keys ──────────────────────────────────────────────────────
DEEPGRAM_API_KEY   = os.environ["DEEPGRAM_API_KEY"]
STT_MODEL          = os.getenv("STT_MODEL", "nova-2")
GOOGLE_API_KEY     = os.environ["GOOGLE_API_KEY"]
GOOGLE_MODEL       = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
CARTESIA_API_KEY    = os.environ["CARTESIA_API_KEY"]
CARTESIA_VOICE_ID   = os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02")

# ── Twilio ────────────────────────────────────────────────────────────────────
TWILIO_AUTH_TOKEN  = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]

# ── Airtable ──────────────────────────────────────────────────────────────────
AIRTABLE_PAT        = os.environ["AIRTABLE_PAT"]
AIRTABLE_BASE_ID    = os.environ["AIRTABLE_BASE_ID"]
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "call_logs")

# ── Call Forwarding ───────────────────────────────────────────────────────────
# The phone number to forward calls to when the caller asks to speak to a human.
# Must be in E.164 format, e.g. +15551234567 or +918529103161
TRANSFER_PHONE_NUMBER = os.getenv("TRANSFER_PHONE_NUMBER", "")

# ── Agent behaviour ───────────────────────────────────────────────────────────
# Maximum call duration in seconds.  After this limit the agent says goodbye
# and disconnects.  This prevents runaway API costs from stuck/looping calls.
MAX_CALL_DURATION_SECONDS = int(os.getenv("MAX_CALL_DURATION_SECONDS", "600"))

# The agent's personality / instructions fed as the system prompt to the LLM.
AGENT_SYSTEM_PROMPT = os.getenv("AGENT_SYSTEM_PROMPT") or """\
You are Milley — a warm, friendly, and professional AI customer-support \
representative for MarsMan, an AI solutions company that helps businesses \
and individuals harness the power of artificial intelligence.

═══ IDENTITY ═══
• Name: Milley
• Tone: natural and conversational — like a real person on the phone, \
not a chatbot.  Warm, casual, and helpful.
• Opening line (EVERY call): "Hi there! I'm Milley from MarsMan. \
How can I help you today?"
• Closing line (EVERY call): "Thanks for calling MarsMan! Have a great \
day — reach out anytime if you need anything."

═══ SPEAKING STYLE ═══
• Talk like a real human on a phone call.  Use SHORT sentences.
• NEVER give long paragraphs.  One or two sentences at a time is ideal.
• Use everyday language — no corporate jargon or stiff phrasing.
• Pause naturally between ideas.  Don't dump all info at once.
• If you have multiple things to say, break them up across turns.
• Match the caller's energy — if they're casual, be casual back.
• Use filler words sparingly (e.g. "sure", "gotcha", "absolutely") \
to sound natural.
• Avoid repeating the same information the caller already knows.

═══ BUSINESS INFORMATION ═══
Company: MarsMan — AI Solution Provider
Business Hours: Monday–Saturday, 9 AM – 7 PM (closed Sundays)
How to get started: phone call, website, or walk-in at our office.
Pricing philosophy: mid-range and affordable; exact quotes depend \
on scope.

Products & Services:
1. Custom AI Chatbots & Voice Agents
2. AI-Powered Business Intelligence
3. Workflow Automation
4. Computer Vision Solutions
5. AI Strategy Consulting

═══ KNOWLEDGE SCOPE ═══
You can help with: general enquiries, product info, pricing guidance, \
onboarding timelines, business hours & contacts.

═══ BEHAVIOR GUIDELINES ═══
• Listen first, then respond briefly.
• Ask one clarifying question at a time — don't overwhelm the caller.
• If you don't know something, say so and offer to connect them.
• Always thank the caller and make them feel valued.

═══ CALL TRANSFER ═══
• If the caller asks to speak with a real person, a human, \
or a manager — use the transfer_to_human tool to forward the call.
• Before transferring, say something brief like: \
"Sure, let me connect you right away."
• Do NOT offer to transfer unless the caller asks for it or is \
clearly frustrated.  Try to help them yourself first.
• After calling the tool, do NOT say anything further — the call will \
be handed off automatically.

═══ HARD LIMITS ═══
• Do NOT invent specific pricing numbers.
• Do NOT promise specific delivery dates.
• Do NOT speak negatively about competitors.
• Do NOT go outside MarsMan's business scope.
• For exact pricing say: "For exact pricing, I'd recommend speaking \
with our team or checking our website."
"""

# =============================================================================
#  LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice-agent")

# =============================================================================
#  AIRTABLE — call log helper
# =============================================================================

def log_call_to_airtable(
    caller_number: str,
    duration_seconds: float,
    transcript: str,
    created_at: datetime,
) -> None:
    """
    Write a completed call record to the 'call_logs' table in Airtable.

    IMPORTANT: This function is always called inside a try/except block so
    that a logging failure never crashes the agent or affects the caller.

    Fields used (must exist in your Airtable table):
      - caller_number   (Single line text)
      - duration_seconds (Number)
      - transcript      (Long text)
      - created_at      (Date — stored as ISO-8601 string)
    """
    try:
        airtable = AirtableApi(AIRTABLE_PAT)
        table = airtable.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

        record = {
            "caller_number":    caller_number,
            "duration_seconds": round(duration_seconds, 1),
            "transcript":       transcript,
            "created_at":       created_at.strftime("%Y-%m-%d"),
        }

        table.create(record)
        logger.info(
            "✅ Call log saved to Airtable | caller=%s duration=%.1fs",
            caller_number,
            duration_seconds,
        )

    except Exception as exc:
        # Log the error but do NOT re-raise — the call has already ended and
        # crashing here would serve no purpose other than hiding the real error.
        logger.error(
            "❌ Failed to save call log to Airtable: %s — "
            "caller=%s duration=%.1fs",
            exc,
            caller_number,
            duration_seconds,
            exc_info=True,
        )

# =============================================================================
#  TWILIO WEBHOOK VALIDATION — helper
# =============================================================================

def validate_twilio_request(url: str, params: dict, signature: str) -> bool:
    """
    Verify that an incoming request genuinely originated from Twilio.

    Twilio signs every webhook request with an HMAC-SHA1 hash of the URL +
    POST parameters using your Auth Token.  Checking this signature prevents
    anyone from spoofing your SIP endpoint.

    Args:
        url:       The full public URL that Twilio called (must match exactly).
        params:    The POST form parameters Twilio sent.
        signature: The value of the 'X-Twilio-Signature' HTTP header.

    Returns:
        True if the request is valid, False otherwise.
    """
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    return validator.validate(url, params, signature)

# =============================================================================
#  PREWARM — load heavy models once per worker process
# =============================================================================

def prewarm(proc):  # noqa: ANN001
    """
    LiveKit calls this once when the worker process starts, before any calls
    arrive.  We use it to download / warm up the Silero VAD model so there
    is no cold-start delay on the first call.
    """
    logger.info("Prewarming Silero VAD model…")
    proc.userdata["vad"] = silero.VAD.load(
        # ── Silero VAD tuning parameters ─────────────────────────────────────
        #
        # min_speech_duration (seconds, default 0.05):
        #   How long audio must look like speech before VAD declares "speech
        #   started".  Increase this if background noise triggers false starts.
        min_speech_duration=0.1,
        #
        # min_silence_duration (seconds, default 0.55):
        #   After the caller stops speaking, the VAD waits this long before
        #   declaring "turn ended" and handing off to the LLM.
        #   LOWER  → agent responds faster but may cut the caller off mid-thought
        #   HIGHER → more tolerant of natural pauses but feels sluggish
        #   Phone calls often benefit from 0.6–0.9 s because mobile networks
        #   can add jitter that creates artificial gaps in speech.
        min_silence_duration=0.7,
        #
        # prefix_padding_duration (seconds, default 0.5):
        #   Audio to prepend before each detected speech chunk.  This ensures
        #   the beginning of the caller's utterance (which fired the VAD) is
        #   included in the STT transcript rather than being clipped.
        prefix_padding_duration=0.5,
        #
        # activation_threshold (0.0–1.0, default 0.5):
        #   Confidence score above which a frame is classified as speech.
        #   LOWER  → more sensitive, picks up quiet voices but also noise
        #   HIGHER → requires clearer speech, may miss soft-spoken callers
        activation_threshold=0.5,
        #
        # max_buffered_speech (seconds, default 60.0):
        #   Maximum audio kept in the rolling speech buffer.  Guards against
        #   extremely long uninterrupted monologues exhausting memory.
        max_buffered_speech=30.0,
        #
        # force_cpu:
        #   Set to True so the model runs on CPU. VPS servers typically do not
        #   have a GPU, and the Silero model is lightweight enough for CPU.
        force_cpu=True,
    )
    logger.info("Silero VAD model ready.")

# =============================================================================
#  VOICE AGENT — defines the agent persona and pipeline
# =============================================================================

class VoiceAssistant(Agent):
    """
    The voice agent definition.

    In LiveKit Agents v1.4, the Agent class defines the persona (instructions)
    and optionally overrides processing nodes (stt_node, llm_node, tts_node).
    """

    def __init__(self, room: rtc.Room, participant_identity: str) -> None:
        super().__init__(
            instructions=AGENT_SYSTEM_PROMPT,
        )
        self._room = room
        self._participant_identity = participant_identity

    @function_tool()
    async def transfer_to_human(
        self,
        context: RunContext,
    ) -> str:
        """Transfer the caller to a human representative.

        Use this tool ONLY when the caller explicitly asks to speak with a
        real person, a human agent, or a manager.  The call will be forwarded
        to the support line and this session will end.
        """
        logger.info("🔧 transfer_to_human tool invoked")

        if not TRANSFER_PHONE_NUMBER:
            logger.warning("⚠️ TRANSFER_PHONE_NUMBER not set")
            return (
                "I'm sorry, call forwarding is not configured at the moment. "
                "Please try calling our office directly."
            )

        try:
            # Prevent the user from interrupting mid-transfer
            context.disallow_interruptions()

            logger.info(
                "📞 Transferring: room=%s, participant=%s, to=%s",
                self._room.name,
                self._participant_identity,
                TRANSFER_PHONE_NUMBER,
            )

            # Perform the SIP cold transfer via LiveKit API
            lk = api.LiveKitAPI()
            await lk.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self._room.name,
                    participant_identity=self._participant_identity,
                    transfer_to=f"tel:{TRANSFER_PHONE_NUMBER}",
                    play_dialtone=True,
                ),
            )
            await lk.aclose()
            logger.info("✅ Call transferred to %s", TRANSFER_PHONE_NUMBER)
            return "Call transferred successfully."

        except Exception as exc:
            logger.error("❌ Call transfer failed: %s", exc, exc_info=True)
            return (
                "I'm sorry, I wasn't able to transfer the call right now. "
                "Please try calling our office directly for assistance."
            )

# =============================================================================
#  ENTRYPOINT — called once per inbound call
# =============================================================================

async def entrypoint(ctx: JobContext) -> None:
    """
    LiveKit calls this coroutine every time a new participant (i.e., caller)
    joins a room.  This is the heart of the agent.

    High-level flow:
      1.  Connect to the LiveKit room.
      2.  Instantiate the streaming STT → LLM → TTS pipeline via AgentSession.
      3.  Listen while enforcing the max-duration safety limit.
      4.  After the call ends, log the transcript to Airtable.
    """

    # ── Call metadata ──────────────────────────────────────────────────────────
    call_start_time = time.monotonic()
    call_started_at = datetime.now(timezone.utc)
    transcript_entries: list[str] = []

    # ── Connect to the LiveKit room ───────────────────────────────────────────
    logger.info("Joining room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the caller's participant to appear in the room.
    participant = await ctx.wait_for_participant()
    logger.info("Caller joined: %s", participant.identity)

    # Extract caller number from participant identity.
    raw_identity = participant.identity or ""
    if raw_identity.startswith("sip:"):
        caller_number = raw_identity.split(":")[1].split("@")[0]
    elif raw_identity.startswith("+") or raw_identity.lstrip("+").isdigit():
        caller_number = raw_identity
    else:
        caller_number = raw_identity or "unknown"
    logger.info("Caller number: %s", caller_number)

    # ── Build the AgentSession with streaming voice pipeline ──────────────────
    #
    #  STT  →  LLM (streaming tokens)  →  TTS (streaming audio)
    #
    session = AgentSession(
        # ── Voice Activity Detection ───────────────────────────────────────────
        vad=ctx.proc.userdata["vad"],

        # ── Speech-to-Text (Deepgram) ──────────────────────────────────────────
        stt=deepgram.STT(
            api_key=DEEPGRAM_API_KEY,
            model=STT_MODEL,
            language="en-US",
        ),

        # ── Language Model (Google Gemini) ─────────────────────────────────────
        llm=google.LLM(
            api_key=GOOGLE_API_KEY,
            model=GOOGLE_MODEL,
        ),

        # ── Text-to-Speech (Cartesia Sonic 3) ──────────────────────────────────
        tts=cartesia.TTS(
            api_key=CARTESIA_API_KEY,
            voice=CARTESIA_VOICE_ID,
            model="sonic-3",
            language="en",
        ),

        # ── Interruption handling ──────────────────────────────────────────────
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )

    # ── Wire up transcript collection via events ──────────────────────────────

    @session.on("user_input_transcribed")
    def on_user_speech(event) -> None:
        if event.transcript and event.is_final:
            text = event.transcript.strip()
            if text:
                logger.info("Caller said: %s", text)
                transcript_entries.append(f"Caller: {text}")

    @session.on("conversation_item_added")
    def on_agent_speech(event) -> None:
        item = event.item
        if item.role == "assistant" and item.text_content:
            text = item.text_content.strip()
            if text:
                logger.info("Agent said:  %s", text)
                transcript_entries.append(f"Agent:  {text}")

    # ── Start the agent ───────────────────────────────────────────────────────
    await session.start(
        agent=VoiceAssistant(room=ctx.room, participant_identity=participant.identity),
        room=ctx.room,
    )

    # Greet the caller
    await session.generate_reply(
        instructions="Greet the caller warmly and ask how you can help them today.",
    )

    # ── Max call duration enforcement ─────────────────────────────────────────

    async def watch_duration() -> None:
        """After MAX_CALL_DURATION_SECONDS, politely end the call."""
        await asyncio.sleep(MAX_CALL_DURATION_SECONDS)
        logger.warning(
            "⏰ Maximum call duration (%ds) reached — disconnecting.",
            MAX_CALL_DURATION_SECONDS,
        )
        await session.say(
            "I'm sorry, we've reached the maximum call duration. "
            "Please call back if you need further assistance. Goodbye!",
            allow_interruptions=False,
        )
        await ctx.room.disconnect()

    async def wait_for_disconnect() -> None:
        """Wait until the caller hangs up or the room is closed."""
        disconnected = asyncio.Event()

        def on_disconnected(*_args) -> None:
            disconnected.set()

        ctx.room.on("disconnected", on_disconnected)
        await disconnected.wait()

    duration_task   = asyncio.create_task(watch_duration())
    disconnect_task = asyncio.create_task(wait_for_disconnect())

    done, pending = await asyncio.wait(
        [duration_task, disconnect_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # ── Calculate call duration ───────────────────────────────────────────────
    call_duration = time.monotonic() - call_start_time
    logger.info(
        "Call ended | caller=%s duration=%.1fs",
        caller_number,
        call_duration,
    )

    # ── Log call to Airtable ──────────────────────────────────────────────────
    full_transcript = "\n".join(transcript_entries)
    logger.info("Full transcript:\n%s", full_transcript or "(no speech detected)")

    log_call_to_airtable(
        caller_number    = caller_number,
        duration_seconds = call_duration,
        transcript       = full_transcript,
        created_at       = call_started_at,
    )

# =============================================================================
#  WORKER BOOTSTRAP
# =============================================================================

if __name__ == "__main__":
    """
    Run the agent worker.

    Development:
        python agent.py dev
        → Connects to LiveKit Cloud, accepts one room at a time.
          Great for local testing via the LiveKit Sandbox.

    Production:
        python agent.py start
        → Runs as a persistent worker that handles multiple concurrent calls.
          Use a process manager (systemd, supervisord, screen, tmux) to keep
          this running continuously on your VPS.
    """
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="voice-agent",
        )
    )
