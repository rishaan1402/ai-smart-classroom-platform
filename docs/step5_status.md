# Step 5 status: audio pipeline (2026-09-29)

## What exists
- `audio/io.py`: ffmpeg → 16 kHz mono float32 (settings from opensource-clipping-autopilot's `extract_audio`)
- `audio/vad.py`: Silero VAD via faster-whisper (no extra model)
- `audio/asr.py`: faster-whisper, word timestamps; decode settings from clipping-autopilot's `transcribe_video` (`vad_filter=True`, `condition_on_previous_text=False`, which stops long lectures from losing speech after a hallucinated stretch)
- `audio/teacher.py`: talk ratio, words/min, questions (EN + HI cues), wait time after questions, long pauses, longest monologue, per-minute timeline
- `audio/room.py`: room-mic per-second loudness, voice-band share, spikes, chatter
- `audio/enhance.py`: optional spectral gating (noisereduce). DeepFilterNet doesn't build on Python 3.12 here
- `audio/pipeline.py`: CLI (`--role teacher|room`, `--reuse-transcript` to recompute metrics without ASR)
- `tools/audio_noise_robustness.py`: mixes real classroom noise into clean speech at set SNRs and measures ASR drift
- `tests/unit/test_audio.py`: 12 tests (63 total, all passing)

Test audio: 4 NPTEL lectures under Creative Commons BY (IIT Bombay ×3 incl. 1 Hindi, IIT Guwahati), in `footage/lectures/` with licence metadata.

## Results
| Lecture | Model | Speed on M4 CPU | Language | Talk ratio | Words/min | Questions | Median wait |
|---|---|---|---|---|---|---|---|
| Hindi nutrition intro (9 min) | large-v3 | 4.7× real time | hi (0.83) | 0.90 | 127 | 0 | – |
| Pharmacognosy L1, English (26 min) | small | 0.17× real time | en (1.00) | 0.95 | 144 | 16 (mostly genuine, rhetorical) | 0.4 s |

- **Hindi ASR**: the opening is accurate, including English terms in Devanagari ("MD", "associate professor"), but some middle passages garble. No punctuation, so sentences are split on pauses.
- **English ASR**: clean, with 0.91 mean word confidence.
- **Noise robustness** (Whisper small, 3 min English, real Securus classroom noise added):

| SNR | Drift vs. clean | + enhancement |
|---|---|---|
| 20 dB (lapel-like) | 2.3% | 1.3% |
| 10 dB | 3.0% | 2.3% |
| 5 dB | 3.8% | 3.3% |
| 0 dB | 5.5% | 7.5% |

  Whisper tolerates *additive* noise well, and enhancement helps slightly down to 5 dB but hurts at 0 dB, so it stays off by default. This test doesn't model distance, echo or AGC, which is what made the real CCTV mic untranscribable (0 s speech). A **lapel mic remains required** for teacher transcription.
- **Room mic (Securus)**: a constant −4 dBFS, 98% voice band, no spikes or chatter periods detected. CCTV AGC flattens the level, so even the noise curve carries no signal. **Pilot requirement: a room mic with fixed gain (AGC off).**

## Known limits
1. Question detection is keyword/punctuation based: good for punctuated English, weak for Hindi. The step 6 LLM pass should classify questions (and rhetorical vs. real).
2. CPU speed: large-v3 is 4.7× real time on the Mac, so hours for a school day. The plan's NVIDIA edge (CTranslate2 float16) is the deployment target. For local dev use `small` for English; for Hindi try `large-v3-turbo` or an IndicWhisper fine-tune next.
3. No diarization yet (pyannote needs a Hugging Face token and model licence acceptance). Only needed for student-speech turns on the room mic.
4. Talk ratio ~0.9–0.95 reflects recorded monologue lectures, not a live class. These metrics need real classroom lapel audio to mean anything.

## Next
- Step 6: transcript + timetable → LLM topic segmentation and question classification, joined with `class_engagement.csv` by minute.
- A Hindi model comparison (large-v3 vs. large-v3-turbo vs. IndicWhisper) on the Hindi lecture.
- Get 10 minutes of real lapel audio from a teacher for true classroom metrics.
