# Labelling guide

Labels are ground truth for measuring the classroom pipeline. Consistency matters more than speed.

**What you see**: one student's region (cyan box) over 3 seconds: left = t−1 s, **middle = the moment to label**, right = t+1 s. Faces are pixelated; judge posture and head direction only. You never see the system's own guess.

| Key | Label | Use when |
|---|---|---|
| 1 | attentive | Head up, facing the lesson (board, teacher, or a classmate who is speaking to the class) |
| 2 | writing | Head down over the desk with a hand working: writing, drawing, reading the book |
| 3 | hand_raised | A hand raised to ask or answer, at any height (small children often raise to face level) |
| 4 | head_down | Head resting on the desk or arms, or slumped, not working |
| 5 | turned_away | Facing clearly away from the lesson (window, behind, sideways) and not talking |
| 6 | interacting | Turned towards a neighbour who is turned back: talking, passing something |
| 7 | phone | Holding or looking at a phone |
| 8 | out_of_seat | Standing or walking away from their seat |
| 9 | cant_tell | Occluded, too blurred or too small to judge. Use it honestly; it's excluded from accuracy, not counted as an error |
| 0 | not_a_student | The box isn't on one student: teacher, empty seat, two people, an object. Counted as a detection error |

Rules of thumb:
- Judge the **middle frame**; use the outer frames only to resolve motion (writing vs. resting, turning vs. glancing).
- Writing vs. head down: if a hand moves across the three frames or holds a pen over paper → writing.
- A brief glance sideways that returns by the third frame → attentive.
- When two labels fit, choose the one a teacher standing at the front would choose.

Process: two labellers on at least 20% of tasks to measure agreement (Cohen's κ ≥ 0.6 is the bar for trusting the labels). Run `python -m labeling.evaluate --lesson <lesson>` for the report.

Consent: for a pilot school, labelling students' footage for model improvement is its own purpose. Add it to the consent form; only lessons whose students consented to it may be exported for labelling.
