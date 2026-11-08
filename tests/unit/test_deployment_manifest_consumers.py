"""Offline manifest consumers check routing isolation and rendezvous wiring."""

import shlex

import pandas as pd
import pytest
import yaml

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.dl.k8s import render_torchrun_ddp_job


def _agent(name, arguments):
    session = Session.ingest(pd.DataFrame({"x": [1.0, 2.0]})).ai.configure(provider="mock")
    registry = build_default_registry()
    result = execute_tool(
        session,
        propose_tool_execution(name, arguments, registry),
        confirmed=True,
        registry=registry,
    )
    assert result.executed, result.error
    return list(yaml.safe_load_all(result.result.yaml_text))


@pytest.mark.parametrize("replicas,port", [(1, 8080), (3, 8097)])
def test_agent_serving_manifests_route_only_to_their_own_model(replicas, port, tmp_path):
    deployments = []
    for name in ("risk-model", "price-model"):
        docs = _agent(
            "emit_k8s_serve_deployment",
            {
                "path": str(tmp_path / (name + ".yaml")),
                "name": name,
                "replicas": replicas,
                "port": port,
            },
        )
        deployment, service = docs
        assert deployment["spec"]["replicas"] == replicas
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert container["ports"][0]["containerPort"] == port
        assert service["spec"]["ports"][0]["port"] == port
        deployments.append((deployment, service))
    for index, (deployment, service) in enumerate(deployments):
        labels = deployment["spec"]["template"]["metadata"]["labels"]
        other = deployments[1 - index][0]["spec"]["template"]["metadata"]["labels"]
        for selector in (
            service["spec"]["selector"],
            deployment["spec"]["selector"]["matchLabels"],
        ):
            assert all(labels.get(key) == value for key, value in selector.items())
            assert not all(other.get(key) == value for key, value in selector.items())


@pytest.mark.parametrize("agent", [False, True])
@pytest.mark.parametrize("script", ["/workspace/train.py", "/workspace/train model.py"])
def test_ddp_job_dns_and_script_argv_are_consistent(agent, script, tmp_path):
    if agent:
        docs = _agent(
            "emit_k8s_ddp_job",
            {
                "path": str(tmp_path / "job.yaml"),
                "nnodes": 3,
                "nproc_per_node": 2,
                "script_path": script,
            },
        )
    else:
        docs = list(
            yaml.safe_load_all(
                render_torchrun_ddp_job(nnodes=3, nproc_per_node=2, script_path=script)
            )
        )
    job, service = docs[:2]
    pod = job["spec"]["template"]["spec"]
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["parallelism"] == job["spec"]["completions"] == 3
    assert pod.get("subdomain") == service["metadata"]["name"]
    assert service["spec"]["clusterIP"] == "None"
    command = pod["containers"][0]["command"][-1]
    argv = shlex.split(command[command.index("torchrun") :].replace("\\\n", " "))
    assert argv[-1] == script
    assert "--nnodes=3" in argv and "--nproc_per_node=2" in argv
