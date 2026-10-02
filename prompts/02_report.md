# Prompt 02: Check the report's numbers and references

## 1. Task

Check my draft `report.tex` against the saved Question 2 results. Correct numbers and typos, leaving my explanations, interpretations, and reasons for each choice as they are. This check follows stage 4 and my independent verification.

## 2. Context

Start in `ps1/` and read the agent-instructions file, `spec.md`, `tests.md`, `log.md`, `README.md`, my draft, and the relevant files in `results/`. The report source is `ps1/report.tex`; this prompt is `ps1/prompts/02_report.md`.

Before starting, confirm that stage 4 is complete, my independent verification has a dated log entry, and the draft and this prompt were committed together before the run. If a prerequisite or supporting result is missing, report it and stop. Otherwise, give a brief checking plan.

## 3. Checks and constraints

1. **Numerical results and references.** Check every computed number against `results/*.json`, allowing the report's display rounding. Compare parameters and algorithm settings with `spec.md`, and tables with the generated LaTeX files and, where relevant, `results/tables.md`. Each table or figure reference must exist and point to the intended experiment. Replace an explicit table or figure placeholder only when its generated reference is unambiguous.

2. **Explanations and choices.** Edit only numbers, typos, and the explicit placeholders above. Flag unsupported conclusions, substantive errors, unclear references, and missing text for me to revise. Do not fill in missing results or explanations, change my choices, rerun experiments, or alter code, saved results, inputs, or manual work. Follow the agent file's permissions; do not install packages or rewrite Git history.

3. **Report contents.** Check coverage of parts (a)–(h), the citation to my manual derivation, and final test results. The verification section must include two independent checks, one being the manual-kernel comparison, distinguish agent claims from what I checked, and cite the dated entries in `log.md`. Compare these descriptions with the log. Check that the AI-use note identifies the agent, access mode, and actual contributions, including drafting assistance. Flag anything missing; your review does not replace my verification.

## 4. Output format and commit

Append a dated entry to `log.md` with the agent/model version and this prompt. List each correction's location, old and new text, and supporting file/key. Record the checks performed and open concerns, keeping measured evidence distinct from claims. Return the correction count and a readable diff.

Once the check is complete, commit the report corrections and log as `report: checked against results/, N corrections`, using the actual count for `N` and identifying agent authorship. Keep flagged concerns in the log for my revision. This commit records the check, not approval of my explanations.

## 5. Success criteria

Every reported number and artifact reference is checked, corrections can be traced to their evidence, and my explanations remain unchanged. Report the commit hash, correction count, and unresolved concerns, then stop. I will read the diff, resolve the concerns, finish the report, build `report.pdf`, and make the stage-5 commit and final submission tag myself.
