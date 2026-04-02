"""
Kokoro TTS Engine
==================
Turns text into speech using the Kokoro-82M model.

TWO-STAGE PIPELINE WITH CONTINUOUS STREAM:
  Stage 1 — Synth thread:
    Takes sentences from speech_queue
    Runs them through Kokoro
    Concatenates all audio chunks per sentence into one array
    Puts the single array into audio_queue
    
  Stage 2 — Playback thread:
    Opens ONE continuous audio stream (never closes between sentences)
    Takes audio arrays from audio_queue
    Writes them directly into the stream
    
  Because the stream never closes and reopens, there is zero gap
  between sentences. The audio flows continuously.
"""

import queue
import threading

import numpy as np
import sounddevice as sd

from config import (
    KOKORO_VOICE, KOKORO_LANG, KOKORO_SAMPLE_RATE, KOKORO_SPEED,
    GRAY, RESET, CYAN, GREEN, YELLOW
)


class KokoroTTS:
    """
    Text-to-speech engine using Kokoro-82M with two-stage pipeline
    and continuous audio stream for gapless playback.
    """

    def __init__(self):
        self.pipeline = None
        self.enabled = False
        self.available = True

        # Stage 1 queue: sentences (text) waiting to be synthesized
        self.speech_queue = queue.Queue()

        # Stage 2 queue: synthesized audio (numpy arrays) waiting to be played
        self.audio_queue = queue.Queue()

        # Background threads
        self.synth_thread = None
        self.playback_thread = None

        # Is the engine running?
        self.running = False

        # Interrupt flag
        self.interrupt_event = threading.Event()

        # Voice settings
        self.voice = KOKORO_VOICE
        self.lang = KOKORO_LANG

    def initialize(self) -> bool:
        """
        Load the Kokoro model. Downloads from HuggingFace on first run (~80MB).
        Returns True if successful.
        """
        try:
            print(f"{CYAN}[TTS] Initializing Kokoro TTS...{RESET}")

            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code=self.lang)

            # Pre-warm: force the voice file to load and caches to fill
            for _ in self.pipeline("Ready.", voice=self.voice, speed=KOKORO_SPEED):
                pass

            print(f"{GREEN}[TTS] ✓ Kokoro loaded (voice: {self.voice}){RESET}")

            # Start the two-stage pipeline threads
            self.running = True

            self.synth_thread = threading.Thread(
                target=self._synth_worker,
                daemon=True,
            )
            self.synth_thread.start()

            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True,
            )
            self.playback_thread.start()

            return True

        except ImportError:
            print(f"{YELLOW}[TTS] Kokoro not installed. To install:{RESET}")
            print(f"{YELLOW}      pip install kokoro soundfile sounddevice numpy{RESET}")
            print(f"{YELLOW}      Also install espeak-ng on your system{RESET}")
            self.available = False
            return False

        except Exception as e:
            print(f"{YELLOW}[TTS] Kokoro init failed: {e}{RESET}")
            self.available = False
            return False

    # ─────────────────────────────────────────
    # Stage 1: Synthesis Thread
    # ─────────────────────────────────────────

    def _synth_worker(self):
        """
        Takes sentences from speech_queue, synthesizes into audio,
        puts audio into audio_queue. Runs ahead of playback.
        """
        while self.running:
            try:
                if self.interrupt_event.is_set():
                    self.interrupt_event.clear()

                text = self.speech_queue.get(timeout=0.5)

                if text is None:
                    self.audio_queue.put(None)
                    break

                if self.interrupt_event.is_set():
                    self.speech_queue.task_done()
                    continue

                self._synthesize(text)
                self.speech_queue.task_done()

            except queue.Empty:
                continue

    def _synthesize(self, text: str):
        """
        Convert one sentence to audio.
        
        Collects ALL audio chunks from Kokoro into one single numpy
        array, then puts that one array into the audio queue. This
        means one queue item = one full sentence of audio, no
        fragmentation.
        """
        if not self.pipeline or not text.strip():
            return

        try:
            generator = self.pipeline(text, voice=self.voice, speed=KOKORO_SPEED)

            # Collect all chunks for this sentence into one list
            chunks = []

            for _graphemes, _phonemes, audio in generator:
                if self.interrupt_event.is_set():
                    return

                if audio is None or len(audio) == 0:
                    continue

                # Convert PyTorch tensor to numpy
                if not isinstance(audio, np.ndarray):
                    audio = audio.cpu().numpy()

                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                chunks.append(audio)

            # Concatenate all chunks into one continuous array
            if chunks:
                full_audio = np.concatenate(chunks)

                # Normalize volume
                max_val = np.abs(full_audio).max()
                if max_val > 1.0:
                    full_audio = full_audio / max_val

                self.audio_queue.put(full_audio)

        except Exception as e:
            print(f"{YELLOW}[TTS] Synthesis error: {e}{RESET}")

    # ─────────────────────────────────────────
    # Stage 2: Playback Thread (Continuous Stream)
    # ─────────────────────────────────────────

    def _playback_worker(self):
        """
        Plays audio through a CONTINUOUS output stream.
        
        Instead of calling sd.play() for each sentence (which opens
        and closes the audio device every time, causing gaps), we open
        ONE stream and write audio directly into it.
        
        stream.write() blocks until the audio has been accepted by the
        sound card buffer, then immediately accepts the next chunk.
        No gap between writes = no gap between sentences.
        """
        # Open one continuous audio stream — stays open the whole time
        stream = sd.OutputStream(
            samplerate=KOKORO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        stream.start()

        try:
            while self.running:
                try:
                    if self.interrupt_event.is_set():
                        self.interrupt_event.clear()

                    audio = self.audio_queue.get(timeout=0.5)

                    if audio is None:
                        break

                    if self.interrupt_event.is_set():
                        self.audio_queue.task_done()
                        continue

                    # Reshape to (samples, 1) for mono output stream
                    # OutputStream expects a 2D array: rows = samples, cols = channels
                    if audio.ndim == 1:
                        audio = audio.reshape(-1, 1)

                    # Write directly into the continuous stream — no gap
                    stream.write(audio)
                    self.audio_queue.task_done()

                except queue.Empty:
                    continue

        finally:
            stream.stop()
            stream.close()

    # ─────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────

    def queue_sentence(self, sentence: str):
        """Add a sentence to the speech queue."""
        if self.enabled and self.pipeline and sentence.strip():
            self.speech_queue.put(sentence)

    def stop(self):
        """Interrupt current speech and clear both queues."""
        self.interrupt_event.set()

        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()

        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        try:
            sd.stop()
        except Exception:
            pass

    def toggle(self, enable: bool) -> bool:
        """
        Turn TTS on or off.
        First time enabling loads the model.
        Returns True if toggle succeeded.
        """
        if enable and not self.pipeline:
            if self.initialize():
                self.enabled = True
                return True
            return False

        self.enabled = enable
        if enable:
            print(f"{GREEN}[TTS] Voice enabled{RESET}")
        else:
            print(f"{GRAY}[TTS] Voice disabled{RESET}")
        return True

    def set_voice(self, voice: str):
        """Change the voice (e.g. 'bm_george', 'af_heart', 'am_adam')."""
        self.voice = voice
        print(f"{CYAN}[TTS] Voice changed to: {voice}{RESET}")

    def wait_for_completion(self):
        """Block until all queued sentences finish speaking."""
        if self.enabled:
            self.speech_queue.join()

    def shutdown(self):
        """Clean up everything."""
        self.running = False
        self.stop()
        self.speech_queue.put(None)
        self.audio_queue.put(None)
        print(f"{CYAN}[TTS] Shut down{RESET}")


# Global Instance
tts_engine = KokoroTTS()