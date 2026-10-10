---
description: Work through PRESENT.md tasks with the tiered agent workflow, update the task files, and loop. Optional arg = max tasks this invocation (default 3).
---

# Batch Loop

You are running the doc-pilot batch loop. One invocation = plan (if needed) → work up to N tasks → update the task files. N = `$ARGUMENTS` if a number was passed, else 3.

State files (repo root): **PRESENT.md** (current batch board), **PREVIOUS.md** (completed log), **FUTURE.md** (upcoming batches), **UPDATES.md** (session log).

## 1. Load state

Read PRESENT.md, FUTURE.md, and the tail of PREVIOUS.md. Run `git status` and `git log --oneline -5` to see where the repo actually is — trust the repo over the files if they disagree, and fix the files.

## 2. Refill if the board is empty

If PRESENT.md has no unchecked tasks:
1. Move the finished batch's tasks into PREVIOUS.md (dated, with commit hashes).
2. Promote the next batch from FUTURE.md. Expand it into concrete tasks **using the `planner` agent (Opus)**: give it the batch goal, the CLAUDE.md constraints, and the current code; it returns ordered tasks. Write them into PRESENT.md, each with a *Verify* line. Remove the promoted batch from FUTURE.md.
3. If FUTURE.md is also empty: report that the plan is exhausted, suggest next directions, and stop the loop (see §5).

## 3. Work tasks (the core)

For each of up to N topmost unchecked tasks, in order — never skip ahead, later tasks assume earlier ones:

1. **Scan (Haiku):** if the task touches existing code, send `fast-reader` to map the relevant files/symbols first. Skip for greenfield tasks with nothing to read.
2. **Plan (Opus):** for multi-file or design-sensitive tasks, get a concrete step plan from `planner`. Skip for tasks whose PRESENT.md description is already an executable spec.
3. **Implement (Sonnet):** send `coder` the task text + scan results + plan. It must run the task's *Verify* line and report honest output. For Claude-API-touching code, tell it to consult the `claude-api` skill — never from memory.
4. **Review (Fable):** for any non-trivial change, run `deep-reviewer` on the diff. Route real findings back to `coder` to fix; re-verify. Skip only for trivial/mechanical tasks.
5. **Commit + push:** one commit per completed task with a "why" message (history is a deliverable), then `git push origin master` so GitHub stays current between runs. **Never commit CLAUDE.md, UPDATES.md, PRESENT/PREVIOUS/FUTURE.md** — Anthony decides on those. Never commit secrets or `.env`; before any push, confirm nothing staged matches a key/secret pattern.
6. **Update the board:** check the task off in PRESENT.md, appending the commit hash.

If a task fails verification after a genuine fix attempt: leave it unchecked, add a `⚠ BLOCKED:` note under it with what's wrong and what's needed, and move to reporting (don't churn).

## 4. Close out the invocation

- Confirm everything committed this invocation is pushed (`git status` shows nothing ahead of origin/master).
- Append a dated entry to UPDATES.md session log (tasks done, decisions made, anything blocked) and refresh its Status/Next-Up sections.
- Final message: lead with what got done (task names + commit hashes), then anything blocked or decided, then what the next invocation will pick up.

## 5. Looping

- Continuous mode is `/loop /batch-loop` — the loop skill re-invokes this command; self-paced.
- Stop the loop (or don't schedule the next iteration) when: the board and FUTURE.md are exhausted, a task is BLOCKED on input only Anthony can give (API key, deploy account, PII decision on eval docs), or the same task fails twice across invocations.
- A stopped loop reports *why* it stopped as the first line.
