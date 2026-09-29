# Step 7 status: dashboard (2026-09-30)

## What exists
**Backend**
- `analytics/store.py`: `LessonStore`, a read-only view over pipeline outputs (lessons, per-second engagement downsampled to ≤ 600 points, topics, questions, transcript, seat grid, per-student history). Strips local file paths; lesson ids are whitelisted and path-checked.
- `api/routes/lessons.py`:
  - `GET /api/lessons`, `/api/lessons/{id}`: any signed-in user.
  - `GET /api/lessons/{id}/seats`: anonymous seats for everyone; student ids only for admin / principal / teacher roles **and** only while consent allows.
  - `GET /api/lessons/{id}/preview`: the **blurred** preview only (H.264, supports range requests for seeking).
  - `GET /api/students/{id}`: teacher / principal / admin only; 404 for both "unknown" and "no consent" (doesn't leak who exists).
  - Every view is appended to the audit log with user id and role; denied student views are logged too.

**Frontend** (`dashboard/src`)
- **Lessons**: cards with length, average engagement, seats / teacher talk, and which sources exist (classroom video, teacher audio).
- **Lesson**: stat tiles; blurred video synced with an engagement curve (topic bands, hover crosshair + tooltip, click to seek, playhead); state-mix bar; seat map (learned positions, sequential engagement fill, consented names for permitted roles, click → student); topics table (click → seek); teacher report (talk ratio, pace, longest monologue, questions with wait times, time by method); transcript synced to playback.
- **Student**: per-lesson engaged share + state mix, with a consent and audit notice and an explicit "estimates, not grades" note. No ranking against classmates.
- Chart colours: the dataviz reference palette's dark steps, validated on the dashboard surface (`#0f172a`): all checks pass, worst adjacent CVD ΔE 8.4.

**Fixes to existing SentinelCore setup found while running it**
- `psycopg[binary]` added: SQLAlchemy 2.1 maps `postgresql://` to psycopg 3, so the API couldn't even import.
- `bcrypt==4.0.1` pinned: passlib 1.7.4 crashes on bcrypt ≥ 4.1, so registration and login returned 500.

**Pipeline fixes**
- Previews are now H.264 + faststart (OpenCV's `mp4v` doesn't play in browsers).
- The step 3 previews of Altima and Lolo predated blurring; they were re-rendered blurred before being served.

Tests: `tests/unit/test_lessons_api.py`, 8 tests (104 total, all passing). Verified end-to-end in the browser against real Postgres/Redis: login, lesson list, video lesson (seek, tooltip, seats), audio lesson (topics, teacher report, transcript), and viewer-role gating via the API (anonymous seats, 403 on student pages, 401 without login, 404 on path traversal, 206 range requests).

## Run locally
```
docker run -d --name aisc-postgres -e POSTGRES_DB=sentinel -e POSTGRES_USER=sentinel -e POSTGRES_PASSWORD=<DB_PASSWORD> -p 5433:5432 postgres:16-alpine
docker run -d --name aisc-redis -p 6380:6379 redis:7-alpine redis-server --requirepass <REDIS_PASSWORD>
set -a && . ./.env && set +a && .venv/bin/alembic upgrade head
# then start "api" and "dashboard" from .claude/launch.json (uvicorn :8000, vite :5173)
```
`.env` (git-ignored) holds the generated dev secrets and two test accounts (admin, viewer). Ports 5433/6380 avoid the Smartvision project's containers on 5432/6379.

## Not built / limits
- **Exam flags page**: exam mode (step 2 of the roadmap) doesn't exist yet, so there's nothing to show.
- **No lesson with both video and teacher audio** exists yet, so the combined view (engagement × topics on one timeline, per-topic engagement column) is exercised by unit tests, not by real data.
- **Student page with real data**: no consented students are enrolled; the page is verified via tests and the no-consent path in the browser.
- **No per-teacher or cross-lesson curriculum view** yet; this needs multiple real lessons per teacher and subject.
- Lessons are read from files, one school per deployment; not tenant-scoped like SentinelCore's Postgres tables. Move to Postgres/ClickHouse when there's more than one school.
- There's no "teacher" role assignment UI; roles are set in the `users` table (SentinelCore gives the first user admin, the rest viewer).
