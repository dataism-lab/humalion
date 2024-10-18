import base64
import urllib
from pathlib import Path

from github import Github


def github_download_file(download_dir: str, repo: str, path: str, git_token: str | None = None, branch="main") -> Path:
    g = Github(git_token) if git_token else Github()
    path = Path(path)
    repo = g.get_repo(repo)
    content_encoded = repo.get_contents(urllib.parse.quote(str(path)), ref=branch).content
    content = base64.b64decode(content_encoded)
    new_filepath = Path(download_dir) / path.name
    with open(new_filepath, "wb") as f:
        f.write(content)

    return new_filepath
