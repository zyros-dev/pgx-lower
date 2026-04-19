#!/usr/bin/env python3
"""Update specs/STATUS.md atomically and commit/push to main.

Subcommands:
    claim NN BRANCH        — mark spec NN in_progress (owner from $OWNER or git config)
    release NN             — set spec NN back to available
    in_review NN PR        — mark spec NN in_review with PR number
    complete NN PR         — mark spec NN done with PR number
    block NN REASON        — mark spec NN blocked with a note

All subcommands rebase main, edit STATUS.md, commit, push. They refuse to run
from a git worktree (only from the main checkout) so the canonical state is
in one place.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = REPO_ROOT / "specs" / "STATUS.md"

# Match a single board row. Groups: name, state, owner, branch, pr, notes.
# Allows arbitrary inner spacing; we re-emit with single spaces.
ROW_RE = re.compile(
    r"^\| (?P<name>\d{2}-[\w-]+) \| (?P<state>[\w_]+) \|(?P<owner>[^|]*)\|(?P<branch>[^|]*)\|(?P<pr>[^|]*)\|(?P<notes>[^|]*)\|$",
    re.M,
)


def die(msg: str, code: int = 1) -> "None":
    print(f"spec_status: {msg}", file=sys.stderr)
    sys.exit(code)


def assert_main_checkout() -> None:
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--git-dir"], cwd=REPO_ROOT, text=True
    ).strip()
    if "worktrees" in git_dir:
        die("must run from the main checkout, not a git worktree")


def run(cmd: list[str], **kw) -> None:
    subprocess.check_call(cmd, cwd=REPO_ROOT, **kw)


def find_row(text: str, nn: str) -> re.Match[str]:
    nn = nn.zfill(2)
    for m in ROW_RE.finditer(text):
        if m.group("name").startswith(f"{nn}-"):
            return m
    die(f"spec {nn} not found in STATUS.md")
    raise AssertionError  # unreachable


def replace_row(text: str, m: re.Match[str], **fields) -> str:
    cur = m.groupdict()
    cur.update(fields)
    new_row = (
        f"| {cur['name']} | {cur['state']} | {cur['owner']} | "
        f"{cur['branch']} | {cur['pr']} |{cur['notes']}|"
    )
    return text[: m.start()] + new_row + text[m.end():]


def default_owner() -> str:
    raw = subprocess.check_output(
        ["git", "config", "user.email"], cwd=REPO_ROOT, text=True
    ).strip()
    user = raw.split("@", 1)[0] or "unknown"
    from datetime import date
    return f"{user}-{date.today().strftime('%m%d')}"


def commit_and_push(message: str) -> None:
    run(["git", "pull", "--rebase", "--autostash", "--quiet", "origin", "main"])
    run(["git", "add", str(STATUS.relative_to(REPO_ROOT))])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", "main"])


def cmd_claim(nn: str, branch: str) -> None:
    owner = os.environ.get("OWNER") or default_owner()
    text = STATUS.read_text()
    m = find_row(text, nn)
    if m["state"].strip() not in {"available", "blocked"}:
        die(f"spec {nn} is {m['state'].strip()} (owner: {m['owner'].strip()}); "
            f"release first if you really mean to take it over")
    text = replace_row(text, m,
                       state="in_progress",
                       owner=f" {owner} ",
                       branch=f" {branch} ",
                       pr=" — ")
    STATUS.write_text(text)
    commit_and_push(f"claim spec {nn} ({owner} / {branch})")
    print(f"claimed {nn} as {owner} on {branch}")


def cmd_release(nn: str) -> None:
    text = STATUS.read_text()
    m = find_row(text, nn)
    text = replace_row(text, m,
                       state="available",
                       owner=" — ",
                       branch=" — ",
                       pr=" — ")
    STATUS.write_text(text)
    commit_and_push(f"release spec {nn}")


def cmd_in_review(nn: str, pr: str) -> None:
    text = STATUS.read_text()
    m = find_row(text, nn)
    text = replace_row(text, m,
                       state="in_review",
                       pr=f" #{pr.lstrip('#')} ")
    STATUS.write_text(text)
    commit_and_push(f"spec {nn} in review (#{pr.lstrip('#')})")


def cmd_complete(nn: str, pr: str) -> None:
    text = STATUS.read_text()
    m = find_row(text, nn)
    text = replace_row(text, m,
                       state="done",
                       pr=f" #{pr.lstrip('#')} ")
    STATUS.write_text(text)
    commit_and_push(f"complete spec {nn} (#{pr.lstrip('#')})")


def cmd_block(nn: str, reason: str) -> None:
    text = STATUS.read_text()
    m = find_row(text, nn)
    notes = m["notes"].strip()
    new_notes = f" {reason} " if not notes else f" {reason}; {notes} "
    text = replace_row(text, m,
                       state="blocked",
                       notes=new_notes)
    STATUS.write_text(text)
    commit_and_push(f"block spec {nn}: {reason}")


def main() -> None:
    assert_main_checkout()
    if not STATUS.is_file():
        die(f"{STATUS} missing")
    if len(sys.argv) < 2:
        die(__doc__ or "usage error", code=2)

    sub = sys.argv[1]
    args = sys.argv[2:]
    handlers = {
        "claim":     (2, cmd_claim),
        "release":   (1, cmd_release),
        "in_review": (2, cmd_in_review),
        "complete":  (2, cmd_complete),
        "block":     (2, cmd_block),
    }
    if sub not in handlers:
        die(f"unknown subcommand {sub!r}", code=2)
    arity, fn = handlers[sub]
    if len(args) != arity:
        die(f"{sub} expects {arity} args, got {len(args)}", code=2)
    fn(*args)


if __name__ == "__main__":
    main()
