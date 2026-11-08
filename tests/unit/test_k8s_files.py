"""Checks on the Kubernetes files, so that a wrong edit is found before the cluster test."""

from pathlib import Path

import yaml

K8S = Path(__file__).resolve().parents[2] / "k8s"


def load(name):
    return yaml.safe_load((K8S / name).read_text())


def container():
    return load("deployment.yaml")["spec"]["template"]["spec"]["containers"][0]


def test_two_copies_with_a_zero_downtime_rolling_update():
    spec = load("deployment.yaml")["spec"]
    assert spec["replicas"] == 2
    assert spec["strategy"]["type"] == "RollingUpdate"
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    assert spec["strategy"]["rollingUpdate"]["maxSurge"] == 1


def test_a_stopping_copy_waits_before_it_exits():
    c = container()
    assert c["lifecycle"]["preStop"]["exec"]["command"] == ["sleep", "5"]
    pod = load("deployment.yaml")["spec"]["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] >= 30


def test_probes_use_the_api_endpoints():
    c = container()
    assert c["startupProbe"]["httpGet"]["path"] == "/ready"
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/live"


def test_service_and_budget_select_the_deployment_pods():
    labels = load("deployment.yaml")["spec"]["template"]["metadata"]["labels"]
    assert load("deployment.yaml")["spec"]["selector"]["matchLabels"] == labels
    assert load("service.yaml")["spec"]["selector"] == labels
    assert load("pdb.yaml")["spec"]["selector"]["matchLabels"] == labels


def test_service_reaches_the_container_port():
    c = container()
    assert c["ports"][0]["name"] == "http" and c["ports"][0]["containerPort"] == 8000
    assert load("service.yaml")["spec"]["ports"][0]["targetPort"] == "http"


def test_runs_as_non_root_with_limits():
    pod = load("deployment.yaml")["spec"]["template"]["spec"]
    c = container()
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert c["securityContext"]["allowPrivilegeEscalation"] is False
    assert "memory" in c["resources"]["requests"] and "memory" in c["resources"]["limits"]
