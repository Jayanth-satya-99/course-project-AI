# Policy-Adherent Agent — Web Application Backend
# ==================================================
# Flask server providing interactive web endpoints for:
#   - Real-time customer support chat with live module inspection telemetry
#   - Multi-agent simulation sandbox (Genuine vs Red-Team adversaries)
#   - Automated benchmark evaluation suite (comparative baseline analytics)
#   - Policy graph visualizer

import os
from flask import Flask, render_template, request, jsonify

from orchestrator import PolicyAdherentOrchestrator
from module5.multi_agent_sim import MultiAgentEnvironment
from evaluation import BenchmarkEvaluator
from module5.generator import OllamaGenerationError


app = Flask(__name__, template_folder="templates", static_folder="static")


@app.errorhandler(OllamaGenerationError)
def handle_ollama_generation_error(error):
    """Report local model failures without silently switching to templates."""
    return jsonify({"error": str(error), "code": "ollama_generation_unavailable"}), 503


# Shared orchestrator and simulation environment instances
orchestrator = PolicyAdherentOrchestrator()
sim_env = MultiAgentEnvironment(orchestrator)
evaluator = BenchmarkEvaluator()


@app.route("/")
def index():
    """Serves the main single-page interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Processes an incoming customer message through the 5-module pipeline.
    Returns: response message, decisions, and execution trace for inspector.
    """
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_web_session")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = orchestrator.process_message(user_message, session_id=session_id)
    # Strip raw internal python objects before JSON serialization
    safe_result = {k: v for k, v in result.items() if not k.endswith("_raw")}
    return jsonify(safe_result)


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Resets the active conversation session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default_web_session")
    orchestrator.reset_session(session_id)
    return jsonify({"status": "success", "message": f"Session '{session_id}' reset."})


@app.route("/api/verify", methods=["POST"])
def complete_verification():
    """Records a simulated reviewer verification and re-evaluates the active claim."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default_web_session")
    verification = data.get("verification", "")

    try:
        conv = orchestrator.complete_verification(session_id, verification)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    last_user_message = next(
        (entry["content"] for entry in reversed(conv.conversation_history)
         if entry["role"] == "user"),
        None,
    )
    if not last_user_message:
        return jsonify({"error": "No customer message is available to re-evaluate."}), 409

    result = orchestrator.process_message(last_user_message, session_id=session_id)
    safe_result = {key: value for key, value in result.items() if not key.endswith("_raw")}
    return jsonify({"verification": verification, "result": safe_result})


@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """
    Executes an autonomous multi-turn red-teaming or cooperative simulation.
    """
    data = request.get_json() or {}
    mode = data.get("mode", "redteam")          # 'cooperative' or 'redteam'
    scenario_idx = int(data.get("scenario_idx", 0))
    max_turns = int(data.get("max_turns", 3))

    sim_result = sim_env.run_simulation(
        mode=mode,
        scenario_idx=scenario_idx,
        max_turns=max_turns
    )
    return jsonify(sim_result)


@app.route("/api/benchmark", methods=["GET"])
def run_benchmark():
    """
    Runs the automated evaluation test suite comparing the proposed agent
    with Naive, Always Escalate, and Strict Reject baselines.
    """
    results = evaluator.run_full_evaluation()
    return jsonify(results)


@app.route("/api/policies", methods=["GET"])
def get_policies():
    """
    Returns all policies and clauses in the policy graph for inspection.
    """
    policies_data = []
    for policy in orchestrator.policy_graph.get_all_policies():
        clauses_data = []
        for clause in policy.clauses.values():
            clauses_data.append({
                "clause_id": clause.clause_id,
                "description": clause.description,
                "priority": clause.priority,
                "constraints": clause.constraints
            })
        policies_data.append({
            "policy_id": policy.policy_id,
            "name": policy.name,
            "description": policy.description,
            "type": policy.policy_type,
            "possible_actions": policy.possible_actions,
            "clauses": clauses_data
        })
    return jsonify({
        "metadata": orchestrator.policy_graph.metadata,
        "evaluation_statuses": orchestrator.policy_graph.evaluation_statuses,
        "policies": policies_data,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Policy-Adherent Agent server running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
