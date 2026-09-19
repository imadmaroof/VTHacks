# GitHub CSV Runner

Live at `https://<owner>.github.io/<repo>/` once GitHub Pages is enabled for this repo (Settings > Pages > Source: "Deploy from a branch" > Branch `main`, folder `/docs`). It can also be opened directly as a local file, or served locally — see below.

## Configure it

1. Open **Connection and pipeline settings**.
2. Enter a GitHub personal access token, repository owner, and repository name. The token is stored in this browser's localStorage and is sent only to GitHub's API. Use a fine-grained token with access to the repository's contents, including permission to read and write files.
3. Confirm the branch, input path, and output path. The defaults assume the branch is `main`. Set **Input file path** to any `.csv` under `input/` (e.g. `input/data.csv`) and **Output file path** to exactly `output/result.csv` — that's the path the `score-input.yml` workflow writes to.
4. Set the polling interval and timeout, then use **Test connection**.
5. Choose or drag in a CSV and select **Send to GitHub**.

Settings are saved in this browser so they do not need to be retyped on every run. Do not use this page on a shared or untrusted computer while a token is saved.

## Match the GitHub Action

The **Input file path** must exactly match a path under `input/`; the workflow scores whichever `.csv` file it finds there. The **Output file path** must be `output/result.csv`, since that's what `predict.py` writes (it delegates the actual scoring to `score_upload.score_patient_upload()`, the same validated serving path used elsewhere in the repo). Both paths are relative to the repository root, and the selected branch must be the branch watched and updated by the workflow (`main`).

The page records the output file's SHA before upload. It waits until that SHA changes, which avoids displaying an older result. If no output exists yet, the first output file found is treated as the result — so the workflow needs to have run at least once (even via **Actions > Score uploaded CSV > Run workflow**) before the very first poll.

## Troubleshooting

The status log explains common HTTP failures: `401` means the token was rejected, `403` means permissions or rate limiting, and `404` means the repository, branch, or path could not be found. A timeout usually means the Action is still running or failed; check the workflow's GitHub Actions logs and then use **Check now**.

If your browser blocks API requests from a `file://` page, run a tiny local server from this folder instead:

```text
python -m http.server 8000
```

Then open `http://localhost:8000/`.
