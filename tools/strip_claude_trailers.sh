#!/usr/bin/env bash
# Remove the Co-Authored-By trailers from every commit message in this repo.
#
# Run it as:      bash tools/strip_claude_trailers.sh
# From cmd.exe, PowerShell, Git Bash or WSL. All four work.
#
# WHAT IT CHANGES. Two lines, deleted from every commit message on every branch
# and tag:
#
#     Co-Authored-By: Claude ...
#     ... Generated with [Claude Code] ...
#
# Nothing else moves. Not one file, not one author, not one date. Every commit
# in this repo is already authored AND committed by Ibrahim Eren Bisen; the
# trailer is the only thing that made GitHub list a second contributor.
#
# WHAT IT COSTS. Every commit SHA from the first rewritten one onward changes,
# so the next push must be a force push. A backup tag is made first.
#
# NO `set -e` HERE, deliberately. The first version had it, and it killed the
# script silently at an empty-list check that returned non-zero - it printed
# three lines, looked like it had worked, and had done nothing. A script that
# stops without saying so is worse than one that fails loudly.

cd "$(dirname "$0")/.." || exit 1

# COUNT ON THE REAL BRANCHES ONLY, never `--all`.
#
# `--all` also walks refs/original/ - the untouched copy filter-branch keeps of
# every rewritten ref - and the backup tags made above. Both hold the OLD
# history on purpose, so both still carry the trailers. Counting them made the
# first run of this script report "trailers before: 106, trailers after: 106"
# and refuse to let the push happen, when main and rebuild were in fact already
# clean. The rewrite had worked and the check said it had not.
count_trailers() {
  local n=0 b
  for b in main rebuild; do
    git rev-parse --verify -q "$b" >/dev/null || continue
    n=$((n + $(git log --format='%B' "$b" 2>/dev/null | grep -ci 'co-authored-by: claude')))
  done
  echo "$n"
}

echo "== repo: $(pwd)"

before=$(count_trailers)
echo "== trailers found: $before"
if [ "$before" = "0" ]; then
  echo "== nothing to do"
  exit 0
fi

short=$(git rev-parse --short HEAD)
git tag -f "backup/before-strip-$short" >/dev/null 2>&1
echo "== backup tag: backup/before-strip-$short"

# filter-branch refuses to run with a dirty tree, and training writes to
# progress/ continuously - so the tree goes dirty again a second after it is
# cleaned. `--skip-worktree` makes git stop noticing those files for the
# duration. They are restored at the end; leaving them held would make git
# silently ignore your later edits to them.
# NULL-delimited. `robot/` holds SolidWorks parts with spaces in their names -
# "Main Body.SLDPRT" and friends - and splitting on whitespace turned each of
# those into `git update-index --skip-worktree robot/Main` and a pile of
# `fatal: Unable to mark file` lines.
#
# RUN THIS UNDER GIT BASH, NOT WSL. WSL's git, looking at the repo through
# /mnt/c, disagrees with Windows git about line endings and permission bits, so
# it reports EVERY tracked file as modified - hundreds of them - and the rewrite
# refuses to start. Git Bash sees the three files training is actually writing.
held_count=0
while IFS= read -r -d '' entry; do
  f=${entry:3}
  [ -n "$f" ] || continue
  git update-index --skip-worktree "$f" 2>/dev/null && held_count=$((held_count + 1))
done < <(git status --porcelain -z | grep -z '^.M' || true)
echo "== holding $held_count file(s) that training is writing"

echo "== rewriting $before trailers across all refs - takes a minute, no output"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter \
  'sed -e "/Co-Authored-By: Claude/d" -e "/Generated with \[Claude Code\]/d"' \
  -- --all
rc=$?

# Restore, whether or not the rewrite worked. Read the flags back out of the
# index rather than replaying the list - if the script is interrupted and re-run,
# this still finds and clears everything that is held.
git ls-files -v | grep '^S ' | cut -c3- | while IFS= read -r f; do
  git update-index --no-skip-worktree "$f" 2>/dev/null
done
echo "== released the held files ($(git ls-files -v | grep -c '^S ') still held)"

if [ "$rc" != "0" ]; then
  echo
  echo "FILTER-BRANCH FAILED, exit $rc. Nothing was published. History is"
  echo "unchanged unless it says otherwise above."
  exit "$rc"
fi

after=$(count_trailers)
echo
echo "== trailers before: $before"
echo "== trailers after:  $after"
echo
if [ "$after" = "0" ]; then
  echo "Done. Now publish it:"
  echo
  echo "    git push --force origin main rebuild"
  echo
  echo "If the result is wrong:"
  echo
  echo "    git reset --hard backup/before-strip-$short"
else
  echo "SOME SURVIVED - do NOT force push. $after left."
fi
