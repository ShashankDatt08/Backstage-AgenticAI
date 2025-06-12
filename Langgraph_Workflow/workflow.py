import os
import subprocess
import tempfile
import shutil
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import logging
import openai

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
    target_code: Optional[str]  # Renamed from target_service_code
    status: str
    current_step: str
    generated_code: Optional[str]
    error_message: Optional[str]
    branch_name: Optional[str]


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

        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name, state["base_branch"]],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            alt = "master" if state["base_branch"] == "main" else "main"
            subprocess.run(
                ["git", "checkout", "-b", branch_name, alt],
                cwd=repo_path,
                check=True, capture_output=True, text=True
            )
            state["base_branch"] = alt

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

    finally:
        if state.get("repository_path"):
            shutil.rmtree(os.path.dirname(state["repository_path"]), ignore_errors=True)

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

        # Try to find a file mentioned in the prompt
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
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = f"{type(e).__name__}: {str(e)}"
        logger.error("Code generation failed: %s", state["error_message"])

    if state.get("repository_path"):
        shutil.rmtree(state["repository_path"], ignore_errors=True)
    return state

def create_enhanced_workflow():
    graph = StateGraph(CodeGenerationState)
    graph.add_node("create_branch", create_feature_branch)
    graph.add_node("analyze", enhanced_repository_analyzer)
    graph.add_node("plan", dynamic_prompt_planner)
    graph.add_node("generate", enhanced_code_generator)

    graph.set_entry_point("create_branch")
    graph.add_edge("create_branch","analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

def run_enhanced_code_generation_workflow(session_data: dict) -> dict:
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
        "branch_name": None 
    }

    workflow = create_enhanced_workflow()
    final = workflow.invoke(initial)

    return {
        "session_id": final["session_id"],
        "status": final["status"],
        "current_step": final["current_step"],
        "error_message": final.get("error_message"),
        "branch_name": final.get("branch_name"),
        "generated_code": final.get("generated_code")
    }
