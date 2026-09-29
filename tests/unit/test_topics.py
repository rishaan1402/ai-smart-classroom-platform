"""Unit tests for topic segmentation merging and the lesson join — fake LLM, no Ollama."""

import pandas as pd

from analytics.lesson import method_summary, topic_table
from audio.asr import Segment, Transcript, Word
from topics.llm import parse_json
from topics.segment import Line, merge_chunk, parse_verdicts, question_candidates, segment_transcript


def lines(n, start=1, step=10.0):
    return [Line(i, (i - 1) * step, i * step, f"sentence {i}.") for i in range(start, start + n)]


def transcript_from(n_sentences=6, step=10.0) -> Transcript:
    segs = []
    for i in range(n_sentences):
        s = i * step
        words = [Word(w, s + j, s + j + 0.5, 0.9) for j, w in enumerate(["sentence", f"{i + 1}."])]
        segs.append(Segment(s, s + step, f"sentence {i + 1}.", -0.1, 0.0, words))
    return Transcript("en", 1.0, n_sentences * step, "test", 0.0, segs)


class FakeClient:
    """Returns canned chunk responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat_json(self, system, user, schema, temperature=0.1):
        self.prompts.append(user)
        return self.responses.pop(0)


def topic(start_line, title, method="explanation", cont=False):
    return {"start_line": start_line, "title": title, "summary": f"{title} summary", "method": method,
            "keywords": [], "continues_previous": cont}


class TestMergeChunk:
    def test_maps_lines_to_times_and_closes_segments(self):
        topics = merge_chunk([], lines(6), {"topics": [topic(1, "Intro"), topic(4, "Cells", "example")]}, 60.0)
        assert [(t.title, t.start_s, t.end_s) for t in topics] == [("Intro", 0.0, 30.0), ("Cells", 30.0, 60.0)]

    def test_first_segment_forced_to_chunk_start_and_bad_lines_dropped(self):
        topics = merge_chunk([], lines(5), {"topics": [topic(3, "Late start"), topic(99, "Out of range")]}, 50.0)
        assert len(topics) == 1 and topics[0].start_s == 0.0

    def test_continues_previous_extends_last_topic(self):
        topics = merge_chunk([], lines(3), {"topics": [topic(1, "Photosynthesis")]}, 60.0)
        topics = merge_chunk(topics, lines(3, start=4), {
            "topics": [topic(4, "Light reactions", cont=True), topic(6, "Summary", "recap")],
        }, 60.0)
        assert [t.title for t in topics] == ["Photosynthesis", "Summary"]
        assert topics[0].end_s == 50.0

    def test_empty_llm_output_gives_placeholder(self):
        topics = merge_chunk([], lines(2), {}, 20.0)
        assert topics[0].title == "Unsegmented"


class TestQuestions:
    def test_candidates_english_use_cues_other_languages_all_lines(self):
        chunk = [Line(1, 0, 1, "Today we study cells."), Line(2, 1, 2, "What is a cell?")]
        assert [ln.number for ln in question_candidates(chunk, "en")] == [2]
        assert len(question_candidates(chunk, "hi")) == 2

    def test_verdicts_drop_non_questions_and_unknown_lines(self):
        cands = [Line(1, 0, 1, "What is a cell?"), Line(2, 5, 6, "Now, heroin.")]
        qs = parse_verdicts(cands, {"verdicts": [
            {"line": 1, "type": "rhetorical", "text_en": "What is a cell?"},
            {"line": 2, "type": "not_a_question", "text_en": ""},
            {"line": 7, "type": "open", "text_en": "?"},
        ]})
        assert [(q.line, q.type, q.time_s) for q in qs] == [(1, "rhetorical", 0)]


class TestShortTopics:
    def test_short_segments_fold_into_neighbours(self):
        from topics.segment import Topic, merge_short_topics
        ts = [Topic(0, 20, "Hello", "", "administration"), Topic(20, 200, "Cells", "", "explanation"),
              Topic(200, 230, "Aside", "", "other"), Topic(230, 400, "Tissues", "", "explanation")]
        out = merge_short_topics(ts, min_s=60)
        assert [(t.title, t.start_s, t.end_s) for t in out] == [("Cells", 0, 230), ("Tissues", 230, 400)]


class TestWaitTimes:
    def test_quick_follow_up_makes_question_rhetorical(self):
        from topics.segment import ClassifiedQuestion, add_wait_times
        tr = transcript_from(6)   # word starts at 0,1,10,11,20,21,…; sentence i ends at i*10+1.5
        ls = [Line(i + 1, i * 10.0, i * 10 + 1.5, f"sentence {i + 1}.") for i in range(6)]
        qs = [ClassifiedQuestion(0.0, 1, "open", "sentence 1.", ""),
              ClassifiedQuestion(20.0, 3, "check_understanding", "sentence 3.", "")]
        add_wait_times(qs, ls, tr)
        assert qs[0].wait_time_s == 8.5 and qs[0].effective_type == "open"

        ls[0] = Line(1, 0.0, 9.5, "sentence 1.")                 # teacher resumes 0.5 s later
        add_wait_times(qs, ls, tr)
        assert qs[0].effective_type == "rhetorical"
        assert qs[1].effective_type == "check_understanding"     # checks keep their type


class TestSegmentTranscript:
    def test_chunks_pass_previous_topic(self):
        client = FakeClient([
            {"topics": [topic(1, "Intro")]},
            {"topics": [topic(4, "Intro", cont=True), topic(5, "Enzymes")]},
        ])
        topics, _, _ = segment_transcript(
            transcript_from(6), client, context="Class 9 Biology", chunk_lines=3, min_topic_s=0,
        )
        assert [t.title for t in topics] == ["Intro", "Enzymes"]
        assert '"Intro"' in client.prompts[1] and "Class 9 Biology" in client.prompts[0]

    def test_llm_failure_keeps_lesson_usable(self):
        class Broken:
            def chat_json(self, *a, **k):
                raise TimeoutError("ollama down")
        topics, qs, _ = segment_transcript(transcript_from(4), Broken(), chunk_lines=10)
        assert len(topics) == 1 and topics[0].title == "Unsegmented"


class TestParseJson:
    def test_plain_and_wrapped(self):
        assert parse_json('{"a": 1}') == {"a": 1}
        assert parse_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
        assert parse_json("no json") is None


class TestLessonJoin:
    def test_topic_table_with_engagement(self):
        topics = [
            {"start_s": 0, "end_s": 30, "title": "Lecture", "method": "explanation", "summary": ""},
            {"start_s": 30, "end_s": 60, "title": "Demo", "method": "demonstration", "summary": ""},
        ]
        questions = [{"time_s": 40.0, "type": "open"}, {"time_s": 45.0, "type": "rhetorical"}]
        engagement = pd.DataFrame({
            "second": range(60),
            "engaged_share": [0.5] * 30 + [0.9] * 30,
            "share_turned_away": [0.4] * 30 + [0.05] * 30,
            "share_head_down": [0.1] * 30 + [0.05] * 30,
        })
        table = topic_table(topics, transcript_from(6), questions, engagement)
        assert list(table["engaged_share"]) == [0.5, 0.9]
        assert list(table["vs_lesson_avg"]) == [-0.2, 0.2]
        assert table.loc[0, "main_disengagement"] == "turned_away"
        assert table.loc[1, "q_open"] == 1 and table.loc[1, "q_rhetorical"] == 1

        methods = method_summary(table)
        assert dict(zip(methods["method"], methods["engaged_share"])) == {"explanation": 0.5, "demonstration": 0.9}

    def test_topic_table_without_engagement(self):
        topics = [{"start_s": 0, "end_s": 60, "title": "All", "method": "explanation", "summary": ""}]
        table = topic_table(topics, transcript_from(6), [], None)
        assert "engaged_share" not in table and table.loc[0, "words_per_min"] == 12.0
