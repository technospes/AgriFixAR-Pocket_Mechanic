"""
services/transcription_service.py
Audio transcription using Gemini.

Handles:
  - All farm machines (tractor, harvester, thresher, pump, motor, etc.)
  - Indian regional languages (Hindi, Punjabi, Bhojpuri, Haryanvi, Marathi,
    Gujarati, Tamil, Telugu, Kannada, Odia, Bengali + mixed speech)
  - Returns a clean English problem description for downstream diagnosis
"""

from __future__ import annotations
import asyncio
import logging
from pathlib import Path

import google.generativeai as genai

logger = logging.getLogger(__name__)

MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB


async def transcribe_audio_with_gemini(audio_path: Path) -> str:
    """
    Transcribe a farmer's audio complaint about any farm machine.
    Returns a plain English problem description.
    """
    logger.info(f"🎤 Transcribing audio: {audio_path}")

    try:
        if audio_path.stat().st_size > MAX_AUDIO_SIZE:
            logger.warning("Audio file too large — may be truncated by Gemini")

        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(audio_path))
        mime_type = mime_type or "audio/mp4"

        audio_file = genai.upload_file(
            str(audio_path),
            mime_type=mime_type
        )

        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = """You are an expert assistant who understands how Indian farmers speak about problems with their farm machinery and equipment.

The farmer may speak Hindi, Punjabi, Bhojpuri, Haryanvi, Marathi, Gujarati, Tamil, Telugu, Kannada, Odia, Bengali — or a mix. They describe problems using everyday physical words, not technical terms.

Your job: Listen carefully and return a clear, plain English description of the problem. The machine could be any farm equipment — tractor, harvester, thresher, water pump, submersible pump, electric motor, power tiller, rotavator, chaff cutter, generator, or diesel engine.

COMMON PHRASES FOR ALL MACHINES:

STARTING / POWER PROBLEMS (any machine):
- "chal nahi raha / start nahi ho raha" = machine won't start
- "band ho jaata hai / ruk jaata hai" = machine stops by itself
- "bijli nahi aa rahi / self nahi lag raha" = no electricity / electric start failing
- "dheema chal raha hai / power nahi hai / dum nahi hai" = running slow, losing power

TRACTOR / ENGINE:
- "clutch kaam nahi kar raha" = clutch not working
- "gear nahi lag raha / gear phasna" = gear stuck
- "zyada dhuan / kaala dhuan" = excessive smoke
- "tel nikal raha" = oil or fuel leaking
- "pani nikal raha" = coolant leaking
- "awaaz aa rahi / khad khad / thok thok" = knocking/rattling noise
- "belt toot gayi / belt slip kar rahi" = belt broke or slipping
- "engine garam ho raha / overheat" = overheating
- "steering tight / ghoomti nahi" = stiff steering
- "hydraulic nahi uth raha" = hydraulic lift not working

HARVESTER / THRESHER:
- "machine jam ho gayi / anaj phas gayi" = crop jam
- "anaj nahi nikal raha / bhusa mein anaj" = grain not separating
- "anaj kat raha / toot raha" = grain being cracked
- "chalni band / jali jam" = sieve blocked
- "drum jam / cylinder jam" = threshing drum jammed

PUMPS (water pump / submersible):
- "pani nahi aa raha / pani band" = no water output
- "pani kam aa raha / pressure kam" = low flow or pressure
- "motor nahi chali / motor jal gayi" = motor won't start or burned
- "motor gunjti hai par nahi chalti" = hums but won't spin
- "pani ulta aa raha" = water falls back (check valve problem)
- "fuse ud gayi / MCB trip / relay trip" = fuse blown or relay tripping

CHAFF CUTTER:
- "toka nahi kat raha / blade dull ho gayi" = not cutting cleanly
- "toka jam ho gaya" = jammed
- "machine baar baar band hoti hai" = keeps stopping

GENERATOR:
- "bijli nahi aa rahi / current nahi hai" = no electrical output
- "voltage kam / voltage upar neeche" = low or unstable voltage
- "generator band ho jaata hai" = starts then stops

GENERAL:
- "awaaz bhaari / kharkhara" = grinding noise
- "vibration zyada / bahut hilti" = excessive vibration
- "bearing kharaab / ghar ghar" = bearing noise
- "belt dhili / belt nikal jaati" = belt loose or coming off

INSTRUCTIONS:
1. Identify which machine the farmer is talking about.
2. Understand the problem even if speech is unclear or code-switched.
3. Return 1-2 plain English sentences: (a) which machine, (b) what the problem is.
4. Use simple words — no technical jargon.
5. Do NOT include original speech, translations, brackets, or labels.

GOOD EXAMPLES:
- "The farmer's tractor engine starts but stalls immediately with black smoke from exhaust."
- "The submersible pump motor hums when switched on but no water comes from the pipe."
- "The thresher drum has jammed with crop and the machine has stopped."
- "The chaff cutter blades are not cutting cleanly and the machine keeps stopping."
- "The generator starts but there is no electrical output from the sockets."

Return ONLY the plain English problem description. Nothing else."""

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content([audio_file, prompt]),
        )

        transcription = response.text.strip()
        logger.info(f"📝 Transcription: {transcription[:120]}...")

        try:
            genai.delete_file(audio_file.name)
        except Exception:
            pass

        return transcription if transcription else "farm machine problem"

    except Exception as exc:
        logger.error(f"❌ Audio transcription error: {exc}")
        return "farm machine problem"
