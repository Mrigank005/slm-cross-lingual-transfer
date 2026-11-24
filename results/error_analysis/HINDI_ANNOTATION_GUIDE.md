# Hindi error annotation: extension sheet

File to fill in: `hindi_annotation_extension.csv` (opens in Excel; keep it UTF-8).

## What these rows are

- There are 40 Romanized Hindi test items that the English-tuned QLoRA model (seed 42) got wrong.
- They were drawn at random (`random_state=2026`) from the seed-42 Hindi errors that are not already in `annotation_sample.csv`.
- The 10 Hindi items already annotated for the paper are excluded.
- The sample's mix of error types roughly follows the full error set: mostly negative→neutral and positive→neutral, with a few others.

## What to fill in

| Column                  | What to write                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `failure_category`    | Exactly one category from the list below: the main reason a fluent reader would give for the error.                     |
| `english_translation` | A short English gloss.                                                                                                  |
| `notes`               | Optional. Note whether you think the gold label is itself questionable, and which word or phrase carries the sentiment. |

## Categories

Use the same six categories as the paper's error taxonomy (Appendix E):

| Category                  | Use when                                                                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tokenization_artifact` | The sentiment-bearing word is split into pieces that plausibly lose its meaning, and nothing else explains the error. For Romanized Hindi this should be rare. Use it only when you can name the word. |
| `ambiguous_neutral`     | The tweet is genuinely borderline between neutral and a polar label; a reasonable reader could pick either.                                                                                            |
| `cultural_idiom`        | The sentiment is carried by an idiom, proverb or culturally specific expression whose literal reading is neutral or misleading.                                                                        |
| `context_dependent`     | The sentiment depends on knowing who or what is referred to (a celebrity, politician, event, or an earlier tweet).                                                                                     |
| `code_switching`        | The error is plausibly due to Hindi–English mixing or non-standard Romanized spellings and abbreviations (`h` for *hai*, `bt`, `ldko`, `bhot`).                                             |
| `sarcasm_irony`         | The literal sentiment is the opposite of the intended one.                                                                                                                                             |

## Rules that keep the result usable in the paper

1. **Annotate before looking at the paper's hypothesis.** Ask what a fluent reader sees, not whether the item supports the code-mixing story.
2. **One category per row.** If two apply, pick the one that would fix the error if resolved, and put the other in `notes`.
3. **Don't change the gold label column.** If you disagree with the gold label, say so in `notes`. The paper's Ethics section already says some errors may be label disagreements, so it's useful to count them.
4. **Second annotator, if possible.** Have another fluent Romanized-Hindi reader label 15 or more of the same rows independently, in a copy of the file. Even a small overlap lets the paper report agreement (Cohen's kappa) instead of "no second annotator".
