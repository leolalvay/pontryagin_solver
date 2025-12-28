# Git Workflow (pontryagin_solver)

This repository contains both MATLAB and Python implementations. On this machine, we work with **sparse checkout** so we only have the `python/` folder checked out locally.

---

## 0) Key idea: "Push only python"

Git cannot push a single folder. Git pushes **commits**.

In practice, we "push only python" by:
1) editing only `python/`, and
2) staging only `python/` (`git add python`).

---

## 1) One-time setup (Python-only sparse checkout)

Goal: clone the repo via SSH, but **only check out** the `python/` folder locally (so we don't have `matlab/` on disk).

```bash
# Go to your Git projects folder (adjust if needed)
cd ~/Desktop/KAUST/Github

# Clone the repository using SSH:
# --filter=blob:none  -> partial clone (faster, downloads blobs on demand)
# --no-checkout       -> do not populate the working tree yet (we will enable sparse checkout first)
git clone --filter=blob:none --no-checkout git@github.com:leolalvay/pontryagin_solver.git pontryagin_solver

# Enter the repo folder
cd pontryagin_solver

# Enable sparse checkout (cone mode is the simplest folder-based sparse mode)
git sparse-checkout init --cone

# Tell Git to check out ONLY the python folder
git sparse-checkout set python

# Check out the main branch (populate working tree according to sparse rules)
git checkout main
```

Verify:

```bash
# Shows which folders are included in sparse checkout (should show: python)
git sparse-checkout list

# Shows what is actually present on disk (should show: python/ plus root docs)
ls
```

---

## 2) Daily workflow (recommended)

### 2.1 Pull latest changes

Use fast-forward only (safe: no accidental merge commits):

```bash
# Update your local branch ONLY if it can fast-forward
git pull --ff-only
```

If Git refuses because your local branch diverged (you have local commits):

```bash
# Rebase your local commits on top of the updated remote branch
git pull --rebase
```

### 2.2 Work on Python code

Make changes inside `python/`.

```bash
# Show which files changed and whether you're ahead/behind the remote
git status -sb
```

### 2.3 Stage only python changes

```bash
# Stage only the python folder (this is how we avoid touching matlab/ in commits)
git add python
```

### 2.4 Commit

```bash
# Create a commit with the staged changes
git commit -m "Describe the change"
```

### 2.5 Push to GitHub (main)

```bash
# Upload your commits to the remote main branch
git push origin main
```

---

## 3) Safer workflow (branch + PR)

Use this when making larger changes (recommended if collaborating or when changes are risky).

```bash
# Create and switch to a new feature branch
git checkout -b feature/short-name

# Update your branch base before working (fast-forward only)
git pull --ff-only

# edit files under python/
git add python
git commit -m "Describe the change"

# Push the new branch to GitHub and set upstream tracking
git push -u origin feature/short-name
```

Then open a Pull Request on GitHub and merge into `main`.

---

## 4) Confirm what will be pushed

Before pushing, verify what the last commit contains:

```bash
# Show last commit details + list of files included in it
git show --name-only --stat
```

You should mostly see paths starting with `python/` (unless you intentionally changed root files like `.gitignore` or docs).

---

## 5) Python cache artifacts (never commit)

We ignore Python caches/bytecode:
- `__pycache__/`
- `*.pyc`

If they ever become tracked (rare; only if someone committed them in the past), remove them from tracking:

```bash
# Remove cached tracking (keeps files on disk, stops tracking them in git)
git rm -r --cached **/__pycache__
git rm --cached **/*.pyc

# Commit the cleanup
git commit -m "Stop tracking python cache artifacts"
```

---

## 6) Useful commands (quick reference)

```bash
# Show concise status (branch + changed files)
git status -sb

# Show unstaged differences
git diff

# Show staged differences (what will go into the next commit)
git diff --staged

# Show recent history in a compact graph
git log --oneline --decorate --graph -n 15

# Show configured remotes (URLs)
git remote -v

# Show local branches and what they track
git branch -vv

# Show sparse checkout folders (should list: python)
git sparse-checkout list
```
