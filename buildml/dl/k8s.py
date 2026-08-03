"""Generate the Kubernetes YAML for distributed training and serving.

Running multi-node training on Kubernetes needs an Indexed Job so each pod knows
its node rank, a headless Service so the pods can find each other for the
torchrun rendezvous, and the environment wiring that connects the two. Getting
that arrangement right the first time takes a while, and the failure mode — pods
that start and then hang at the rendezvous — is not self-explanatory.

These functions emit that YAML. Two templates: a training Job with its Service
and optional ConfigMap, and a serving Deployment with health probes.

They write files. They do not apply them, talk to a cluster, or manage anything.
BuildML is not a Helm chart, a managed training platform, or a control plane, and
you still need to supply GPU nodes, networking, shared storage, and RBAC. Every
result says so in its ``limitations``.

Treat the output as a working starting point rather than a finished
configuration. Cluster conventions around image registries, node selectors,
tolerations, and storage vary too much for a generic template to be
production-ready anywhere in particular.

See Also
--------
buildml.dl.ddp : The distributed training these manifests launch.
buildml.dl.packaging : Serving artifacts for the Deployment to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from buildml.core.errors import ValidationError


@dataclass(slots=True)
class K8sJobRenderResult:
    """The YAML that was rendered, and what it does not include.

    Attributes
    ----------
    yaml_text:
        The manifest. Multiple documents separated by ``---``.
    path:
        Where it was written, or ``None`` when only rendered.
    nnodes:
        Node count for a training Job, or replica count for a Deployment.
    nproc_per_node:
        Processes per node for training; 1 for serving.
    disclosures:
        What was written.
    limitations:
        What the cluster operator still has to provide.
    meta:
        Names, namespace, and the options that were used.

    Notes
    -----
    ``to_dict`` reports the YAML's length rather than its content, since a full
    manifest embedded in a history record is noise.

    See Also
    --------
    write_torchrun_ddp_job : Produces this for training.
    write_serve_deployment : Produces this for serving.
    """

    yaml_text: str
    path: Path | None
    nnodes: int
    nproc_per_node: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the render outcome as JSON-safe values.

        The manifest itself is reported as a character count rather than
        embedded — a history entry should record that YAML was written, not
        reproduce it.

        Returns
        -------
        dict
            Path, node and process counts, disclosures, limitations, metadata,
            and the manifest length.
        """
        return {
            "path": None if self.path is None else str(self.path),
            "nnodes": self.nnodes,
            "nproc_per_node": self.nproc_per_node,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "meta": dict(self.meta),
            "yaml_chars": len(self.yaml_text),
        }


def _validate_dns_label(name: str, *, field_name: str = "name") -> None:
    if not name or not str(name).replace("-", "").isalnum():
        raise ValidationError(f"{field_name} must be a simple DNS-1123 label")


def render_torchrun_ddp_job(
    *,
    job_name: str = "buildml-torchrun-ddp",
    namespace: str = "default",
    image: str = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    nnodes: int = 2,
    nproc_per_node: int = 2,
    script_path: str = "/workspace/train.py",
    master_port: int = 29500,
    cpu_request: str = "2",
    memory_request: str = "4Gi",
    gpu_limit: int = 1,
    gpu_request: int | None = None,
    service_account: str | None = None,
    include_configmap: bool = True,
) -> str:
    """Produce the manifest for a multi-node torchrun training job.

    Emits three documents: an Indexed Job whose completion index becomes each
    pod's node rank, a headless Service giving the pods stable DNS names so they
    can find each other, and optionally a ConfigMap holding the shared settings.

    Parameters
    ----------
    job_name:
        Name for the Job and Service. Must be a DNS-1123 label — letters,
        digits, and hyphens.
    namespace:
        Kubernetes namespace.
    image:
        Container image. Must contain PyTorch and your training code.
    nnodes:
        Pods to run. Becomes both ``completions`` and ``parallelism``, so all of
        them run at once — which torchrun requires, since the rendezvous waits
        for every node.
    nproc_per_node:
        Processes per pod, normally the GPU count per node.
    script_path:
        Path to the training script inside the container.
    master_port:
        Rendezvous port.
    cpu_request / memory_request:
        Per-pod resources.
    gpu_limit:
        GPUs per pod. 0 omits the GPU resource entirely.
    gpu_request:
        GPU request when it should differ from the limit. Defaults to the
        limit, which is what Kubernetes requires for GPUs anyway.
    service_account:
        Service account for the pods, when RBAC needs one.
    include_configmap:
        Also emit a ConfigMap and mount it via ``envFrom``.

    Returns
    -------
    str
        The manifest, documents separated by ``---``.

    Raises
    ------
    ValidationError
        If ``nnodes`` or ``nproc_per_node`` is below 1, if ``gpu_limit`` is
        negative, or if the job name is not a valid DNS label.

    Notes
    -----
    **The Indexed completion mode is what makes this work.** Kubernetes sets
    ``JOB_COMPLETION_INDEX`` per pod, and the generated command passes it to
    torchrun as ``--node_rank``. Without indexed mode every pod would claim rank
    0 and the rendezvous would never complete.

    **The Service is headless — ``clusterIP: None``.** Load balancing is exactly
    wrong here: the pods need to address each other individually, and the
    manifest points ``MASTER_ADDR`` at the rank-0 pod's stable DNS name.

    **The pods need shared access to your data.** Nothing here provisions
    storage. Add a volume mount, or bake the data into the image.

    Examples
    --------
    Two nodes, two GPUs each::

        yaml_text = render_torchrun_ddp_job(
            nnodes=2, nproc_per_node=2, gpu_limit=2,
            script_path="/workspace/train.py",
        )

    See Also
    --------
    write_torchrun_ddp_job : Render and write in one step.
    buildml.dl.ddp : The training code these pods run.
    """
    if nnodes < 1:
        raise ValidationError("nnodes must be >= 1")
    if nproc_per_node < 1:
        raise ValidationError("nproc_per_node must be >= 1")
    _validate_dns_label(job_name, field_name="job_name")
    if gpu_limit < 0:
        raise ValidationError("gpu_limit must be >= 0")
    resolved_gpu_request = gpu_limit if gpu_request is None else int(gpu_request)
    cm_name = f"{job_name}-config"
    sa_line = f"      serviceAccountName: {service_account}\n" if service_account else ""
    gpu_req = (
        f'\n              nvidia.com/gpu: "{resolved_gpu_request}"'
        if resolved_gpu_request > 0
        else ""
    )
    gpu_lim = f"\n              nvidia.com/gpu: {gpu_limit}" if gpu_limit > 0 else ""
    env_from = (
        (
            "          envFrom:\n"
            "            - configMapRef:\n"
            f"                name: {cm_name}\n"
        )
        if include_configmap
        else ""
    )
    body = (
        "# Generated by buildml.dl.k8s.render_torchrun_ddp_job\n"
        "# Honesty: template only — not live multi-cluster orchestration.\n"
        "apiVersion: batch/v1\n"
        "kind: Job\n"
        "metadata:\n"
        f"  name: {job_name}\n"
        f"  namespace: {namespace}\n"
        "  labels:\n"
        "    app.kubernetes.io/name: buildml-torchrun-ddp\n"
        "    app.kubernetes.io/part-of: buildml\n"
        "spec:\n"
        "  completionMode: Indexed\n"
        f"  completions: {nnodes}\n"
        f"  parallelism: {nnodes}\n"
        "  backoffLimit: 1\n"
        "  template:\n"
        "    metadata:\n"
        "      labels:\n"
        "        app.kubernetes.io/name: buildml-torchrun-ddp\n"
        "    spec:\n"
        f"{sa_line}"
        "      restartPolicy: Never\n"
        "      containers:\n"
        "        - name: trainer\n"
        f"          image: {image}\n"
        "          resources:\n"
        "            requests:\n"
        f'              cpu: "{cpu_request}"\n'
        f'              memory: "{memory_request}"{gpu_req}\n'
        "            limits:\n"
        f'              memory: "{memory_request}"{gpu_lim}\n'
        f"{env_from}"
        "          env:\n"
        "            - name: MASTER_ADDR\n"
        f"              value: {job_name}-0.{job_name}\n"
        "            - name: MASTER_PORT\n"
        f'              value: "{master_port}"\n'
        "            - name: PET_NNODES\n"
        f'              value: "{nnodes}"\n'
        "            - name: PET_NPROC_PER_NODE\n"
        f'              value: "{nproc_per_node}"\n'
        "          command:\n"
        "            - bash\n"
        "            - -lc\n"
        "            - |\n"
        "              set -euo pipefail\n"
        '              NODE_RANK="${JOB_COMPLETION_INDEX:-0}"\n'
        "              torchrun \\\n"
        f"                --nnodes={nnodes} \\\n"
        f"                --nproc_per_node={nproc_per_node} \\\n"
        '                --node_rank="$NODE_RANK" \\\n'
        '                --master_addr="$MASTER_ADDR" \\\n'
        '                --master_port="$MASTER_PORT" \\\n'
        f"                {script_path}\n"
        "---\n"
        "apiVersion: v1\n"
        "kind: Service\n"
        "metadata:\n"
        f"  name: {job_name}\n"
        f"  namespace: {namespace}\n"
        "spec:\n"
        "  clusterIP: None\n"
        "  selector:\n"
        "    app.kubernetes.io/name: buildml-torchrun-ddp\n"
        f"    job-name: {job_name}\n"
        "  ports:\n"
        "    - name: rdzv\n"
        f"      port: {master_port}\n"
        f"      targetPort: {master_port}\n"
    )
    if include_configmap:
        body += (
            "---\n"
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            f"  name: {cm_name}\n"
            f"  namespace: {namespace}\n"
            "  labels:\n"
            "    app.kubernetes.io/name: buildml-torchrun-ddp\n"
            "    app.kubernetes.io/part-of: buildml\n"
            "data:\n"
            f'  PET_NNODES: "{nnodes}"\n'
            f'  PET_NPROC_PER_NODE: "{nproc_per_node}"\n'
            f'  BUILDML_SCRIPT: "{script_path}"\n'
            '  BUILDML_NOTE: "Template ConfigMap — not a live control plane."\n'
        )
    return body


def write_torchrun_ddp_job(path: str | Path, **kwargs: Any) -> K8sJobRenderResult:
    """Render a training job manifest and write it to disk.

    A thin wrapper over :func:`render_torchrun_ddp_job` that creates parent
    directories, writes the file, and returns a result carrying the honest
    limitations alongside it.

    Parameters
    ----------
    path:
        Where to write. Parent directories are created.
    kwargs:
        Passed through to :func:`render_torchrun_ddp_job`.

    Returns
    -------
    K8sJobRenderResult
        The manifest text, the path, and what the operator still owns.

    Raises
    ------
    ValidationError
        Propagated from the renderer.

    Examples
    --------
    Write and apply::

        result = write_torchrun_ddp_job("k8s/train-job.yaml", nnodes=4)
        result.limitations  # read before assuming this is deployable

    See Also
    --------
    render_torchrun_ddp_job : The renderer, and the full parameter list.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = render_torchrun_ddp_job(**kwargs)
    destination.write_text(yaml_text, encoding="utf-8")
    return K8sJobRenderResult(
        yaml_text=yaml_text,
        path=destination,
        nnodes=int(kwargs.get("nnodes", 2)),
        nproc_per_node=int(kwargs.get("nproc_per_node", 2)),
        disclosures=(
            "Wrote Kubernetes Job+Service"
            + ("+ConfigMap" if kwargs.get("include_configmap", True) else "")
            + f" YAML to {destination}.",
        ),
        limitations=(
            "Example template only — not live multi-cluster orchestration.",
            "Operator must provide GPU nodes, networking, shared data, and RBAC.",
            "Not a Helm chart product or managed training cloud.",
        ),
        meta={
            "job_name": kwargs.get("job_name", "buildml-torchrun-ddp"),
            "namespace": kwargs.get("namespace", "default"),
            "include_configmap": kwargs.get("include_configmap", True),
            "gpu_limit": kwargs.get("gpu_limit", 1),
        },
    )


def render_serve_deployment(
    *,
    name: str = "buildml-serve",
    namespace: str = "default",
    image: str = "python:3.12-slim",
    replicas: int = 1,
    port: int = 8080,
    cpu_request: str = "1",
    memory_request: str = "2Gi",
    gpu_limit: int | None = None,
    service_account: str | None = None,
    bundle_path: str = "/models/bundle",
    kind: str = "pipeline",
) -> str:
    """Produce the manifest for serving a saved bundle.

    Emits a Deployment running the BuildML serve command against a bundle path,
    with readiness and liveness probes on ``/health``, plus a ClusterIP Service
    in front of it.

    Parameters
    ----------
    name:
        Name for the Deployment and Service. Must be a DNS-1123 label.
    namespace:
        Kubernetes namespace.
    image:
        Container image. Must have BuildML and its serving dependencies.
    replicas:
        How many pods. More than one gives redundancy and throughput; the
        Service load-balances across them.
    port:
        HTTP port the server listens on.
    cpu_request / memory_request:
        Per-pod resources.
    gpu_limit:
        GPUs per pod, when inference needs one. ``None`` omits it.
    service_account:
        Service account, when RBAC needs one.
    bundle_path:
        Where the model bundle lives inside the container. Mount it or bake it
        in — nothing here provisions it.
    kind:
        Bundle kind, ``'pipeline'`` or ``'torch'``.

    Returns
    -------
    str
        The manifest, documents separated by ``---``.

    Raises
    ------
    ValidationError
        If the name is not a valid DNS label, or if ``replicas`` is below 1.

    Notes
    -----
    **The two probes do different jobs.** Readiness controls whether the Service
    routes traffic to a pod, so a pod still loading its model is kept out of
    rotation rather than returning errors. Liveness restarts a pod that has
    stopped responding. The liveness delay is longer so that slow startup is not
    mistaken for failure.

    **The Service is ClusterIP, reachable only inside the cluster.** External
    access needs an Ingress, a LoadBalancer, or a port-forward, none of which
    are generated here — TLS, hostnames, and certificates are too
    cluster-specific to template.

    Examples
    --------
    Three replicas serving a pipeline bundle::

        yaml_text = render_serve_deployment(
            replicas=3, bundle_path="/models/churn", kind="pipeline",
        )

    See Also
    --------
    write_serve_deployment : Render and write in one step.
    """
    _validate_dns_label(name, field_name="name")
    if replicas < 1:
        raise ValidationError("replicas must be >= 1")
    sa_line = f"      serviceAccountName: {service_account}\n" if service_account else ""
    gpu_lim = f"\n              nvidia.com/gpu: {int(gpu_limit)}" if gpu_limit else ""
    return (
        "# Generated by buildml.dl.k8s.render_serve_deployment\n"
        "# Honesty: template only — not live multi-cluster / managed IAM.\n"
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        f"  name: {name}\n"
        f"  namespace: {namespace}\n"
        "  labels:\n"
        "    app.kubernetes.io/name: buildml-serve\n"
        "    app.kubernetes.io/part-of: buildml\n"
        "spec:\n"
        f"  replicas: {replicas}\n"
        "  selector:\n"
        "    matchLabels:\n"
        "      app.kubernetes.io/name: buildml-serve\n"
        "  template:\n"
        "    metadata:\n"
        "      labels:\n"
        "        app.kubernetes.io/name: buildml-serve\n"
        "    spec:\n"
        f"{sa_line}"
        "      containers:\n"
        "        - name: serve\n"
        f"          image: {image}\n"
        "          args:\n"
        '            - "--bundle"\n'
        f'            - "{bundle_path}"\n'
        '            - "--kind"\n'
        f'            - "{kind}"\n'
        '            - "--host"\n'
        '            - "0.0.0.0"\n'
        '            - "--port"\n'
        f'            - "{port}"\n'
        "          ports:\n"
        "            - name: http\n"
        f"              containerPort: {port}\n"
        "          resources:\n"
        "            requests:\n"
        f'              cpu: "{cpu_request}"\n'
        f'              memory: "{memory_request}"\n'
        "            limits:\n"
        f'              memory: "{memory_request}"{gpu_lim}\n'
        "          readinessProbe:\n"
        "            httpGet:\n"
        "              path: /health\n"
        "              port: http\n"
        "            initialDelaySeconds: 5\n"
        "            periodSeconds: 10\n"
        "          livenessProbe:\n"
        "            httpGet:\n"
        "              path: /health\n"
        "              port: http\n"
        "            initialDelaySeconds: 15\n"
        "            periodSeconds: 20\n"
        "---\n"
        "apiVersion: v1\n"
        "kind: Service\n"
        "metadata:\n"
        f"  name: {name}\n"
        f"  namespace: {namespace}\n"
        "spec:\n"
        "  type: ClusterIP\n"
        "  selector:\n"
        "    app.kubernetes.io/name: buildml-serve\n"
        "  ports:\n"
        "    - name: http\n"
        f"      port: {port}\n"
        "      targetPort: http\n"
    )


def write_serve_deployment(path: str | Path, **kwargs: Any) -> K8sJobRenderResult:
    """Render a serving manifest and write it to disk.

    A thin wrapper over :func:`render_serve_deployment` that creates parent
    directories, writes the file, and returns a result carrying the honest
    limitations alongside it.

    Parameters
    ----------
    path:
        Where to write. Parent directories are created.
    kwargs:
        Passed through to :func:`render_serve_deployment`.

    Returns
    -------
    K8sJobRenderResult
        The manifest text, the path, and what the operator still owns. The
        replica count is reported in ``nnodes``, since the result type is
        shared with the training path.

    Raises
    ------
    ValidationError
        Propagated from the renderer.

    See Also
    --------
    render_serve_deployment : The renderer, and the full parameter list.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = render_serve_deployment(**kwargs)
    destination.write_text(yaml_text, encoding="utf-8")
    return K8sJobRenderResult(
        yaml_text=yaml_text,
        path=destination,
        nnodes=int(kwargs.get("replicas", 1)),
        nproc_per_node=1,
        disclosures=(f"Wrote serve Deployment+Service YAML to {destination}.",),
        limitations=(
            "Example template only — not live multi-cluster orchestration.",
            "Not a managed IAM / Ingress / cert product.",
        ),
        meta={
            "name": kwargs.get("name", "buildml-serve"),
            "namespace": kwargs.get("namespace", "default"),
        },
    )
