# Evaluation: cctv_securus_2mp_audio

- Labelled tasks: 96 (scored 91; excluded {'cant_tell': 4, 'not_a_student': 1})
- Labellers: claude-provisional
- **State accuracy 58%**, macro-F1 0.50
- **Engaged-vs-not accuracy 73%**; engaged share by labels 78% vs predicted 79%
- Detection errors (not_a_student): 1 of 96

## Per state

| state | precision | recall | F1 | support | predicted |
|---|---|---|---|---|---|
| attentive | 0.75 | 0.74 | 0.74 | 53 | 52 |
| writing | 0.35 | 0.47 | 0.40 | 15 | 20 |
| hand_raised | — | 0.00 | — | 3 | 0 |
| head_down | 0.44 | 0.33 | 0.38 | 12 | 9 |
| turned_away | 0.60 | 0.38 | 0.46 | 8 | 5 |
| interacting | — | — | — | 0 | 0 |
| phone | — | — | — | 0 | 0 |
| out_of_seat | 0.00 | — | — | 0 | 5 |

## Confusion matrix (rows = label, columns = prediction)

| label \ pred | attentive | writing | hand_raised | head_down | turned_away | out_of_seat |
|---|---|---|---|---|---|---|
| attentive | 39 | 6 | 0 | 3 | 2 | 3 |
| writing | 5 | 7 | 0 | 2 | 0 | 1 |
| hand_raised | 1 | 1 | 0 | 0 | 0 | 1 |
| head_down | 2 | 6 | 0 | 4 | 0 | 0 |
| turned_away | 5 | 0 | 0 | 0 | 3 | 0 |
| out_of_seat | 0 | 0 | 0 | 0 | 0 | 0 |

## By person size (resolution bias check)

| boxes | height px | n | accuracy | engaged accuracy |
|---|---|---|---|---|
| small (back) | 59–92 | 31 | 71% | 84% |
| medium | 93–132 | 30 | 53% | 63% |
| large (front) | 136–239 | 30 | 50% | 70% |
