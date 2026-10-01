# Evaluation: cctv_securus_2mp_audio

- Labelled tasks: 96 (scored 91; excluded {'cant_tell': 4, 'not_a_student': 1})
- Labellers: claude-provisional
- **State accuracy 64%**, macro-F1 0.53
- **Engaged-vs-not accuracy 79%**; engaged share by labels 78% vs predicted 86%
- Detection errors (not_a_student): 1 of 96

## Per state

| state | precision | recall | F1 | support | predicted |
|---|---|---|---|---|---|
| attentive | 0.75 | 0.79 | 0.77 | 53 | 56 |
| writing | 0.41 | 0.60 | 0.49 | 15 | 22 |
| hand_raised | — | 0.00 | — | 3 | 0 |
| head_down | 0.50 | 0.33 | 0.40 | 12 | 8 |
| turned_away | 0.60 | 0.38 | 0.46 | 8 | 5 |
| interacting | — | — | — | 0 | 0 |
| phone | — | — | — | 0 | 0 |
| out_of_seat | — | — | — | 0 | 0 |

## Confusion matrix (rows = label, columns = prediction)

| label \ pred | attentive | writing | hand_raised | head_down | turned_away |
|---|---|---|---|---|---|
| attentive | 42 | 6 | 0 | 3 | 2 |
| writing | 5 | 9 | 0 | 1 | 0 |
| hand_raised | 2 | 1 | 0 | 0 | 0 |
| head_down | 2 | 6 | 0 | 4 | 0 |
| turned_away | 5 | 0 | 0 | 0 | 3 |

## By person size (resolution bias check)

| boxes | height px | n | accuracy | engaged accuracy |
|---|---|---|---|---|
| small (back) | 59–92 | 31 | 71% | 84% |
| medium | 93–132 | 30 | 60% | 73% |
| large (front) | 136–239 | 30 | 60% | 80% |
