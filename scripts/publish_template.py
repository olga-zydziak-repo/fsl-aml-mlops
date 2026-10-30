"""Helper: auto-increment pipeline template versions in Artifact Registry.

Instead of hardcoding tags="vN" (easy to get wrong, and can silently overwrite
an existing version if you reuse a number), this reads the existing tags, finds
the highest vN, and assigns the next one automatically.

Usage in a pipeline notebook, replacing the manual upload cell:

    from publish_template import publish_next_version

    template_name, version, tag = publish_next_version(
        project=PROJECT, region=REGION, repo=REPO,
        yaml_path=PIPELINE_YAML,
        description="Evaluation gate: conditional registration",
    )
"""
import re
from typing import Tuple

from kfp.registry import RegistryClient


def _next_version_tag(existing_tags) -> str:
    """From tags like ['v1','v2','latest','v3'] return 'v4'.

    Numeric sort (not lexical): v10 > v9. Non-vN tags (latest, custom) are
    ignored. If there are no vN tags yet, returns 'v1'.
    """
    numbers = []
    for tag in existing_tags:
        m = re.fullmatch(r"v(\d+)", tag)
        if m:
            numbers.append(int(m.group(1)))
    return "v1" if not numbers else f"v{max(numbers) + 1}"


def publish_next_version(
    project: str,
    region: str,
    repo: str,
    yaml_path: str,
    description: str = "",
    pipeline_package_name: str = "fsl-hello-pipeline",
) -> Tuple[str, str, str]:
    """Compile-independent: uploads yaml_path as the next vN version + 'latest'.

    Reads existing tags to determine the next version number, so numbering can
    never collide or go backwards. `pipeline_package_name` must match the
    @dsl.pipeline(name=...) used when compiling — that's the package key under
    which versions are grouped in the registry.

    Returns (template_name, version_id, assigned_tag).
    """
    client = RegistryClient(host=f"https://{region}-kfp.pkg.dev/{project}/{repo}")

    # Read existing tags for this pipeline package. If the package doesn't exist
    # yet (first upload ever), list_tags raises — treat that as "no tags".
    try:
        existing = [t["name"].split("/")[-1] for t in client.list_tags(pipeline_package_name)]
    except Exception:
        existing = []

    next_tag = _next_version_tag(existing)
    print(f"Existing version tags: {sorted(existing) or '(none)'} -> assigning {next_tag}")

    template_name, version_id = client.upload_pipeline(
        file_name=yaml_path,
        tags=[next_tag, "latest"],
        extra_headers={"description": description} if description else None,
    )
    print(f"Published {template_name} @ {next_tag} (version id: {version_id})")
    return template_name, version_id, next_tag
