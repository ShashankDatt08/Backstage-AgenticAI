import os
import subprocess
import tempfile
import shutil
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
import logging
import openai
import time
import requests
import ast
import re


openai.api_key = os.getenv("OPENAI_API_KEY")
GIT_USERNAME = os.getenv("GIT_USERNAME")
GIT_PAT = os.getenv("GIT_PAT")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CodeGenerationState(TypedDict):
    session_id: int
    ticket_key: str
    git_url: str
    base_branch: str
    prompt: str
    repository_path: Optional[str]
    target_code: Optional[str]
    status: str
    current_step: str
    generated_code: Optional[str]
    error_message: Optional[str]
    branch_name: Optional[str]
    file_path: Optional[str]
    file_action: Optional[str]
    summary: Optional[str]
    changed_files: List[str]
    pr_url: Optional[str]


def detect_default_branch(git_url: str) -> str:
    try:
        if not GIT_USERNAME or not GIT_PAT:
            raise EnvironmentError("Missing GIT_USERNAME or GIT_PAT")

        original_url = git_url.strip()
        if original_url.endswith("/"):
            original_url = original_url[:-1]
        if not original_url.lower().endswith(".git"):
            original_url += ".git"

        if original_url.startswith("https://"):
            auth_url = original_url.replace("https://", f"https://{GIT_USERNAME}:{GIT_PAT}@")
        else:
            auth_url = original_url

        result = subprocess.run(
            ["git", "ls-remote", "--symref", auth_url, "HEAD"],
            capture_output=True, text=True, check=True, timeout=30
        )
        for line in result.stdout.splitlines():
            if line.startswith("ref:"):
                parts = line.split()
                if len(parts) >= 2:
                    ref = parts[1]
                    return ref.replace("refs/heads/", "")
        return "main"
    except Exception as e:
        logger.warning("Failed to detect default branch: %s", str(e))
        return "main"


def create_feature_branch(state: CodeGenerationState) -> CodeGenerationState:
    try:
        if not GIT_USERNAME or not GIT_PAT:
            raise EnvironmentError("Missing GIT_USERNAME or GIT_PAT environment variable")

        branch_name = f"feature/ai-{state['ticket_key']}"
        original_url = state["git_url"].strip()

        if original_url.endswith("/"):
            original_url = original_url[:-1]
        if not original_url.lower().endswith(".git"):
            original_url += ".git"

        if original_url.startswith("https://"):
            auth_url = original_url.replace("https://", f"https://{GIT_USERNAME}:{GIT_PAT}@")
        else:
            auth_url = original_url

        temp_dir = tempfile.mkdtemp()
        repo_path = os.path.join(temp_dir, "repo")

        logger.info("Cloning repo from %s into %s", auth_url, repo_path)
        subprocess.run(
            ["git", "clone", "--depth", "1", auth_url, repo_path],
            check=True, capture_output=True, text=True, timeout=120
        )

        state["repository_path"] = repo_path

        # Checkout the base branch first
        try:
            subprocess.run(
                ["git", "checkout", state["base_branch"]],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            # fallback to alternate branch
            alt = "master" if state["base_branch"] == "main" else "main"
            subprocess.run(
                ["git", "checkout", alt],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
            state["base_branch"] = alt

        # Now create and switch to the new feature branch from the checked out base branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            check=True, capture_output=True, text=True
        )

        # Push the new branch upstream
        subprocess.run(
            ["git", "push", "--set-upstream", "origin", branch_name],
            cwd=repo_path,
            check=True, capture_output=True, text=True
        )

        state["branch_name"] = branch_name
        state["current_step"] = "branch_created"
        state["status"] = "Completed"
        logger.info("Feature branch '%s' successfully created and pushed.", branch_name)

    except subprocess.CalledProcessError as e:
        logger.error("Git command failed: %s", e.stderr)
        state["status"] = "Failed"
        state["error_message"] = (
            f"git exited {e.returncode}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        )

    except Exception as e:
        logger.error("Unexpected error during branch creation: %s", str(e))
        state["status"] = "Failed"
        state["error_message"] = f"BranchCreationError: {str(e)}"

    return state


def enhanced_repository_analyzer(state: CodeGenerationState) -> CodeGenerationState:
    """Clone the repo and extract a potentially relevant code file based on keywords."""
    try:
        temp = tempfile.mkdtemp()
        repo = os.path.join(temp, "repo")
        subprocess.run(
            ["git", "clone", "--branch", state["base_branch"], "--depth", "1", state["git_url"], repo],
            check=True, capture_output=True, text=True, timeout=60
        )
        state["repository_path"] = repo

        match_file = None
        for word in state["prompt"].split():
            if word.endswith((".java", ".ts", ".tsx", ".py", ".js", ".sql", ".yaml")):
                match_file = word
                break
            if word.lower().endswith("service"):
                word_clean = word if word.endswith("Service") else word.capitalize()
                match_file = f"{word_clean}.java"
                break

        if match_file:
            for root, _, files in os.walk(repo):
                if match_file in files:
                    path = os.path.join(root, match_file)
                    state["target_code"] = open(path, "r", encoding="utf-8").read()
                    break

        state["current_step"] = "repository_analyzed"
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = str(e)
    return state


def truncate_code(code: str, max_chars: int = 1500) -> str:
    lines = code.splitlines()
    truncated = ""
    for line in lines:
        if len(truncated) + len(line) > max_chars:
            break
        truncated += line + "\n"
    return truncated.strip()


def infer_tech_stack(repo_path: str, max_items: int = 5) -> set:
    tech_stack = set()
    if not os.path.exists(repo_path):
        logger.warning("Repository path does not exist: %s", repo_path)
        return tech_stack

    tech_map = {
        ".java": "Java backend",
        ".py": "Python",
        ".ts": "TypeScript/React",
        ".tsx": "TypeScript/React",
        ".js": "JavaScript",
        ".sql": "SQL/database",
        ".yaml": "CI/CD or configuration",
        ".yml": "CI/CD or configuration"
    }

    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in tech_map:
                tech_stack.add(tech_map[ext])
            if len(tech_stack) >= max_items:
                return tech_stack
    return tech_stack


def dynamic_prompt_planner(state: dict) -> dict:
    """Refine the prompt using LLM, based on context and potential code."""
    try:
        if not state.get("prompt"):
            raise ValueError("Missing required user story in state['prompt']")

        system_message = (
            "You are a senior engineering assistant. Given a Jira user story and relevant code context, "
            "optimize the prompt for an LLM that will generate production-grade code. Identify the type of task "
            "(e.g., frontend, backend, database, infra) and include that in the prompt. Be concise, specific, and unambiguous."
        )

        user_prompt = f"User story:\n{state['prompt']}\n"

        if state.get("target_code"):
            truncated_code = truncate_code(state["target_code"], max_chars=1500)
            user_prompt += f"\nRelevant existing code:\n{truncated_code}\n"

        tech_stack = set()
        if state.get("repository_path"):
            tech_stack = infer_tech_stack(state["repository_path"])
            if tech_stack:
                user_prompt += f"\nTech stack indicators: {', '.join(sorted(tech_stack))}\n"

        logger.debug("Preparing to send the following prompt to the LLM:\n%s", user_prompt)
        logger.debug("Detected tech stack: %s", tech_stack)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        model = state.get("llm_model", "gpt-4")
        temperature = state.get("temperature", 0.3)
        max_tokens = state.get("max_tokens", 800)

        resp = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages
        )

        state["prompt"] = resp.choices[0].message.content.strip()
        state["current_step"] = "prompt_optimized"
        logger.info("Prompt successfully optimized.")

    except openai.OpenAIError as e:
        state["status"] = "Failed"
        state["error_message"] = f"OpenAIError: {type(e).__name__}: {str(e)}"
        logger.error("Prompt planning failed due to OpenAI API error: %s", state["error_message"])

    except FileNotFoundError as e:
        state["status"] = "Failed"
        state["error_message"] = f"FileNotFoundError: {str(e)}"
        logger.error("File system issue in prompt planning: %s", state["error_message"])

    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = f"PlanningError: {type(e).__name__}: {str(e)}"
        logger.error("Prompt planning failed: %s", state["error_message"])

    return state


def enhanced_code_generator(state: CodeGenerationState) -> CodeGenerationState:
    """Generate code using the refined prompt and optional context."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. Generate exactly the code requested based on the prompt. "
                    "Your output must:\n"
                    "- Contain only code (no explanations, comments, or markdown)\n"
                    "- Be production-ready and complete\n"
                    "- Follow clean architecture and coding standards\n"
                    "- Modify or extend existing code if provided"
                )
            },
            {"role": "user", "content": f"{state['prompt']}"}
        ]

        if state.get("target_code"):
            messages.append({"role": "user", "content": f"Relevant code:\n{state['target_code']}"})

        resp = openai.chat.completions.create(
            model="gpt-4",
            temperature=0,
            max_tokens=2048,
            messages=messages
        )

        state["generated_code"] = resp.choices[0].message.content
        state["status"] = "Completed"
        state["current_step"] = "code_generated"
        
        # Generate summary of the work done
        summary_messages = [
            {
                "role": "system",
                "content": "You are a technical writer. Create a concise one-line summary of the work performed."
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this coding task in one line:\n"
                    f"Ticket: {state['ticket_key']}\n"
                    f"Prompt: {state['prompt']}\n"
                    f"Generated code type: {'Test' if 'Test' in state.get('file_path', '') else 'Production'}"
                )
            }
        ]
        
        summary_resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=100,
            messages=summary_messages
        )
        
        state["summary"] = summary_resp.choices[0].message.content.strip()
        logger.info(f"Generated workflow summary: {state['summary']}")

    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = f"{type(e).__name__}: {str(e)}"
        logger.error("Code generation failed: %s", state["error_message"])
    
    return state


def commit_generated_code(state: CodeGenerationState) -> CodeGenerationState:
    try:
        repo = state["repository_path"]
        if not repo:
            raise ValueError("Repository path missing in state")

        # Wait/retry to ensure code is generated
        max_wait_sec = 10
        waited = 0
        while (not state.get("generated_code")) and waited < max_wait_sec:
            logger.debug("Waiting for generated_code to be ready before commit...")
            time.sleep(1)
            waited += 1

        code = state.get("generated_code")
        if not code:
            raise ValueError("Generated code is missing after wait")

        code_lower = code.lower()
        is_test_code = any(kw in code_lower for kw in [
            "test", "assert", "mock", "@test", "unittest", "asserttrue", "assertfalse"
        ])

        if is_test_code:
            file_path = "src/test/java/com/marketplace/onlinemarketplace/service/BidServiceTest.java"
        else:
            file_path = "src/main/java/com/marketplace/onlinemarketplace/service/BidService.java"   

        logger.info(f"Committing code: repo={repo}, file_path={file_path}, code_present={bool(code)}")

        abs_path = os.path.join(repo, file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        cleaned_code = sanitize_generated_code(code)

        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                existing_code = f.read()
            merged_code = smart_append_code(existing_code, cleaned_code)
            file_action = "modify"
        else:
            merged_code = cleaned_code
            file_action = "create"

        # Write the result
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(merged_code)

        branch_name = state.get("branch_name")
        if not branch_name:
            raise ValueError("Branch name missing in state")

        logger.debug(f"Checking out branch: {branch_name}")

        subprocess.run(["git", "checkout", branch_name], cwd=repo, check=True)
        subprocess.run(["git", "add", file_path], cwd=repo, check=True)

        commit_msg = f"{file_action.capitalize()} {file_path} for {state['ticket_key']}"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo, check=True)

        # Track changed files
        if 'changed_files' not in state:
            state['changed_files'] = []
        state['changed_files'].append(file_path)
        state["file_path"] = file_path
        state["file_action"] = file_action    
        state["current_step"] = "code_committed"
        logger.info("Committed changes to %s", file_path)
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = f"CommitError: {type(e).__name__}: {str(e)}"
        logger.error("Commit failed: %s", state["error_message"])
    return state


def push_changes(state: CodeGenerationState) -> CodeGenerationState:
    try:
        repo_path = state["repository_path"]
        branch_name = state.get("branch_name")

        if not repo_path or not branch_name:
            raise ValueError("Missing repository path or branch name.")

        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_path, check=True)
        state["current_step"] = "code_pushed"
        logger.info("Pushed changes to branch '%s'", branch_name)
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = f"PushError: {type(e).__name__}: {str(e)}"
        logger.error("Push failed: %s", state["error_message"])
    return state


def create_pull_request(state: CodeGenerationState) -> CodeGenerationState:
    try:
        headers = {
            "Authorization": f"token {os.getenv('GIT_PAT')}",
            "Accept": "application/vnd.github+json"
        }

        repo_url = state["git_url"].replace(".git", "").split("github.com/")[-1]
        api_url = f"https://api.github.com/repos/{repo_url}/pulls"

        # Fix the f-string formatting here
        changed_files_list = "\n".join(f"- {f}" for f in state.get('changed_files', []))
        pr_body = f"""This PR was auto-generated for ticket {state['ticket_key']}.

### Summary
{state.get('summary', 'Implementation of ticket requirements')}

### Changed Files
{changed_files_list}"""

        pr_data = {
            "title": f"[{state['ticket_key']}] AI-generated implementation",
            "head": state["branch_name"],
            "base": state["base_branch"],
            "body": pr_body
        }

        response = requests.post(api_url, headers=headers, json=pr_data)
        response.raise_for_status()

        pr_url = response.json().get("html_url")
        state["current_step"] = "pull_request_created"
        state["status"] = "Completed"
        state["pr_url"] = pr_url
        logger.info("Pull request created: %s", pr_url)

    except Exception as e:
        logger.error("Pull request creation failed: %s", str(e))
        state["status"] = "Failed"
        state["error_message"] = f"PullRequestError: {type(e).__name__}: {str(e)}"
    return state

def smart_append_code(existing_code: str, generated_code: str) -> str:
    existing_lines = existing_code.splitlines()
    generated_lines = generated_code.splitlines()

    new_imports = [line for line in generated_lines if line.strip().startswith('import')]
    new_body = [line for line in generated_lines if line not in new_imports]

    import_end_index = 0
    for i, line in enumerate(existing_lines):
        if line.strip().startswith('import'):
            import_end_index = i + 1
        elif line.strip() == '':
            continue
        else:
            break

    existing_imports = set(line.strip() for line in existing_lines[:import_end_index])
    unique_new_imports = [line for line in new_imports if line.strip() not in existing_imports]

    full_code = existing_lines[:]
    try:
        last_brace_index = max(i for i, line in enumerate(full_code) if line.strip() == "}")
    except ValueError:
        last_brace_index = len(full_code)

    new_body_indented = ['    ' + line if line.strip() else line for line in new_body]
    modified_code = (
        full_code[:import_end_index]
        + unique_new_imports
        + full_code[import_end_index:last_brace_index]
        + ['']  # spacing
        + new_body_indented
        + ['']  # spacing
        + full_code[last_brace_index:]
    )

    return '\n'.join(modified_code)


def sanitize_generated_code(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```[\w+-]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    lines = code.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove all // or # or <!-- single-line comments
        if re.match(r"^\s*(//|#|<!--).*", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def cleanup_repository(state: CodeGenerationState):
    repo_path = state.get("repository_path")
    if repo_path:
        try:
            shutil.rmtree(os.path.dirname(repo_path), ignore_errors=False)
            logger.info("Temporary repository directory cleaned up: %s", os.path.dirname(repo_path))
        except Exception as e:
            logger.error("Error cleaning up repository directory: %s", str(e))    


def create_enhanced_workflow():
    builder = StateGraph(CodeGenerationState)

    # Define states in order
    builder.add_node("detect_branch", lambda state: {**state, "base_branch": detect_default_branch(state["git_url"])})
    builder.add_node("analyze_repo", enhanced_repository_analyzer)
    builder.add_node("create_branch", create_feature_branch)
    builder.add_node("optimize_prompt", dynamic_prompt_planner)
    builder.add_node("generate_code", enhanced_code_generator)
    builder.add_node("commit_code", commit_generated_code)
    builder.add_node("push_changes", push_changes)
    builder.add_node("create_pr", create_pull_request)

    # Add transitions between steps
    builder.set_entry_point("detect_branch")
    builder.add_edge("detect_branch", "analyze_repo")
    builder.add_edge("analyze_repo", "create_branch")
    builder.add_edge("create_branch", "optimize_prompt")
    builder.add_edge("optimize_prompt", "generate_code")
    builder.add_edge("generate_code", "commit_code")
    builder.add_edge("commit_code", "push_changes")
    builder.add_edge("push_changes", "create_pr")
    builder.add_edge("create_pr", END)

    return builder.compile()


def run_enhanced_code_generation_workflow(session_data: dict) -> dict:
    # Set global git config for user
    try:
        subprocess.run(["git", "config", "--global", "user.email", "Ganesh@infinite.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "Ganesh"], check=True)
    except Exception as e:
        pass  # Ignore errors if already set or in restricted environments

    default_branch = session_data.get("base_branch") or detect_default_branch(session_data["git_url"])

    initial: CodeGenerationState = {
        "session_id": session_data["session_id"],
        "ticket_key": session_data["ticket_key"],
        "git_url": session_data["git_url"],
        "base_branch": default_branch,
        "prompt": session_data["prompt"],
        "repository_path": None,
        "target_code": None,
        "status": "Running",
        "current_step": "starting",
        "generated_code": None,
        "error_message": None,
        "branch_name": None,
        "file_path": None,
        "file_action": None,
        "summary": None,
        "changed_files": [],
        "pr_url": None,
    }

    workflow = create_enhanced_workflow()
    final = workflow.invoke(initial)
    cleanup_repository(final)

    return {
        "session_id": final["session_id"],
        "status": final["status"],
        "current_step": final["current_step"],
        "error_message": final.get("error_message"),
        "branch_name": final.get("branch_name"),
        "generated_code": final.get("generated_code"),
        "pr_url": final.get("pr_url"),
        "summary": final.get("summary"),
        "changed_files": final.get("changed_files", [])
    }