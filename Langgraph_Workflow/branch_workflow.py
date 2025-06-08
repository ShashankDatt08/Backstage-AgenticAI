import os
import subprocess
import tempfile
import shutil
from typing import TypedDict, Dict, Any


GIT_PAT = os.getenv("GIT_PAT")
GIT_USERNAME = os.getenv("GIT_USERNAME", "x-access-token")
if not GIT_PAT:
    raise RuntimeError("Environment variable GIT_PAT must be set to your Personal Access Token")

class BranchState(TypedDict):
    ticket_key: str
    git_url: str
    base_branch: str
    repository_path: str
    branch_name: str
    status: str
    error_message: str

def run_branch_creation_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. Embed PAT into the HTTPS URL
    2. Clone the repo at data['git_url'] on data['base_branch']
    3. Create + checkout feature/ai-{ticket_key}
    4. Push that branch to origin
    """
    original_url = data["git_url"]
    if original_url.startswith("https://"):
        auth_url = original_url.replace(
            "https://",
            f"https://{GIT_USERNAME}:{GIT_PAT}@"
        )
    else:
        auth_url = original_url

    state: BranchState = {
        "ticket_key": data["ticket_key"],
        "git_url": auth_url,
        "base_branch": data["base_branch"],
        "repository_path": "",
        "branch_name": f"feature/ai-{data['ticket_key']}",
        "status": "Running",
        "error_message": ""
    }

    try:
        temp_dir = tempfile.mkdtemp()
        repo_path = os.path.join(temp_dir, "repo")
        subprocess.run(
            ["git", "clone", "--branch", state["base_branch"], "--depth", "1", state["git_url"], repo_path],
            check=True, capture_output=True, text=True, timeout=120
        )
        state["repository_path"] = repo_path

        subprocess.run(
            ["git", "checkout", "-b", state["branch_name"]],
            cwd=repo_path,
            check=True, capture_output=True, text=True
        )

        subprocess.run(
            ["git", "push", "--set-upstream", "origin", state["branch_name"]],
            cwd=repo_path,
            check=True, capture_output=True, text=True
        )

        state["status"] = "Completed"
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = str(e)
    finally:
        if state["repository_path"]:
            shutil.rmtree(os.path.dirname(state["repository_path"]), ignore_errors=True)

    return {
        "branch": state["branch_name"],
        "status": state["status"],
        "error_message": state["error_message"]
    }
