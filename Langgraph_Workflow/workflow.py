import os
import subprocess
import tempfile
import shutil
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import logging
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CodeGenerationState(TypedDict):
    session_id: int
    ticket_key: str
    git_url: str
    base_branch: str
    prompt: str
    repository_path: Optional[str]
    target_service_code: Optional[str]
    status: str
    current_step: str
    generated_code: Optional[str]
    error_message: Optional[str]

def enhanced_repository_analyzer(state: CodeGenerationState) -> CodeGenerationState:
    """Clone the repo and read the main service file if the prompt mentions a Service."""
    try:
        temp = tempfile.mkdtemp()
        repo = os.path.join(temp, "repo")
        subprocess.run(
            ["git", "clone", "--branch", state["base_branch"], "--depth", "1", state["git_url"], repo],
            check=True, capture_output=True, text=True, timeout=60
        )
        state["repository_path"] = repo

        svc = None
        for word in state["prompt"].split():
            if word.endswith("Service") or word.endswith("service"):
                svc = word if word.endswith("Service") else word.capitalize()
                break

        if svc:
            for root, _, files in os.walk(repo):
                if f"{svc}.java" in files:
                    path = os.path.join(root, f"{svc}.java")
                    state["target_service_code"] = open(path, "r", encoding="utf-8").read()
                    break

        state["current_step"] = "repository_analyzed"
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = str(e)
    return state

def enhanced_code_generator(state: CodeGenerationState) -> CodeGenerationState:
    """Send the prompt + optional service code to the LLM and return exactly the code requested."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. Generate exactly the code requested based on the user's prompt. "
                    "The response must meet the following criteria:\n"
                    "- Output only valid, production-grade code with all required dependencies.\n"
                    "- Do not include any explanations, comments, placeholder functions, or markdown formatting.\n"
                    "- Always provide complete, fully functional code — never stubs or empty methods.\n"
                    "- If existing code is provided, integrate or modify it exactly as instructed.\n"
                    "- Assume a professional environment where the code must run as-is."
                )
            }
        ]

        user_content = f"Prompt:\n{state['prompt']}\n"
        if state.get("target_service_code"):
            user_content += f"\nExisting code:\n{state['target_service_code']}\n"
        messages.append({"role": "user", "content": user_content})

        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=2048,
            messages=messages
        )

        state["generated_code"] = resp.choices[0].message.content
        state["status"] = "Completed"
        state["current_step"] = "code_generated"
    except Exception as e:
        state["status"] = "Failed"
        state["error_message"] = str(e)

    if state.get("repository_path"):
        shutil.rmtree(state["repository_path"], ignore_errors=True)
    return state

def create_enhanced_workflow():
    graph = StateGraph(CodeGenerationState)
    graph.add_node("analyze", enhanced_repository_analyzer)
    graph.add_node("generate", enhanced_code_generator)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

def run_enhanced_code_generation_workflow(session_data: dict) -> dict:
    initial: CodeGenerationState = {
        "session_id": session_data["session_id"],
        "ticket_key": session_data["ticket_key"],
        "git_url": session_data["git_url"],
        "base_branch": session_data["base_branch"],
        "prompt": session_data["prompt"],
        "repository_path": None,
        "target_service_code": None,
        "status": "Running",
        "current_step": "starting",
        "generated_code": None,
        "error_message": None
    }

    workflow = create_enhanced_workflow()
    final = workflow.invoke(initial)

    return {
        "session_id": final["session_id"],
        "status": final["status"],
        "current_step": final["current_step"],
        "error_message": final.get("error_message"),
        "generated_code": final.get("generated_code")
    }
