# Evaluation: cctv_altima

- Labelled tasks: 45 (scored 37; excluded {'cant_tell': 4, 'not_a_student': 4})
- Labellers: claude-provisional
- **State accuracy 27%**, macro-F1 0.43
- **Engaged-vs-not accuracy 49%**; engaged share by labels 100% vs predicted 49%
- Detection errors (not_a_student): 4 of 45

## Per state

| state | precision | recall | F1 | support | predicted |
|---|---|---|---|---|---|
| attentive | 0.00 | — | — | 0 | 8 |
| writing | 1.00 | 0.27 | 0.43 | 37 | 10 |
| hand_raised | — | — | — | 0 | 0 |
| head_down | 0.00 | — | — | 0 | 17 |
| turned_away | 0.00 | — | — | 0 | 2 |
| interacting | — | — | — | 0 | 0 |
| phone | — | — | — | 0 | 0 |
| out_of_seat | — | — | — | 0 | 0 |

## Confusion matrix (rows = label, columns = prediction)

| label \ pred | attentive | writing | head_down | turned_away |
|---|---|---|---|---|
| attentive | 0 | 0 | 0 | 0 |
| writing | 8 | 10 | 17 | 2 |
| head_down | 0 | 0 | 0 | 0 |
| turned_away | 0 | 0 | 0 | 0 |

## By person size (resolution bias check)

| boxes | height px | n | accuracy | engaged accuracy |
|---|---|---|---|---|
| small (back) | 61–90 | 13 | 31% | 54% |
| medium | 94–127 | 12 | 42% | 50% |
| large (front) | 128–193 | 12 | 8% | 42% |
