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
    original_url = data["git_url"].strip()

    if original_url.endswith("/"):
        original_url = original_url[:-1]
    if not original_url.lower().endswith(".git"):
        original_url += ".git"

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
            ["git", "clone", "--depth", "1", state["git_url"], repo_path],
            check=True, capture_output=True, text=True, timeout=120
        )
        state["repository_path"] = repo_path

        try:
            subprocess.run(
                ["git", "checkout", "-b", state["branch_name"], state["base_branch"]],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as cpe:
            alt = "master" if state["base_branch"] == "main" else "main"
            subprocess.run(
                ["git", "checkout", "-b", state["branch_name"], alt],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
            state["base_branch"] = alt

        subprocess.run(
            ["git", "push", "--set-upstream", "origin", state["branch_name"]],
            cwd=repo_path,
            check=True, capture_output=True, text=True
        )

        state["status"] = "Completed"

    except subprocess.CalledProcessError as cpe:
        state["status"] = "Failed"
        state["error_message"] = (
            f"git exited {cpe.returncode}\n"
            f"stdout:\n{cpe.stdout}\n"
            f"stderr:\n{cpe.stderr}"
        )

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