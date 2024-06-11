import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from jsonargparse import ArgumentParser
from spython.main import Client

# Define the directory where the projects are located
BASE_DIR: Path = Path(__file__).resolve().parent.parent / "projects"

# List of all available project names
PROJECTS: List[str] = [x.name for x in BASE_DIR.iterdir() if x.is_dir()]


def build_container(project_name: str, container_root: Path) -> str:
    project_path: Path = BASE_DIR / project_name
    container_path: Path = container_root / f"{project_name}.sif"

    if not container_path.exists():
        out = (
            f"Apptainer definition file for {project_name} "
            "does not exist. Skipping build."
        )
        return out

    try:
        Client.build(
            image=str(container_path),
            recipe=str(project_path / "apptainer.def"),
            sudo=False,
            options=["--force"],
        )
        return f"Successfully built container for {project_name}"
    except Exception as e:
        return f"Failed to build container for {project_name}: {e}"


def build(projects: List[str], container_root: Path, max_workers: int) -> None:
    if not container_root:
        logging.info("Container root path is not set.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_container, project, container_root): project
            for project in projects
            if project in PROJECTS
        }
        for future in as_completed(futures):
            project = futures[future]
            try:
                result = future.result()
                logging.info(result)
            except Exception as exc:
                logging.info(
                    f"Project {project} generated an exception: {exc}"
                )


def main():
    parser = ArgumentParser(
        description="Automatically rebuild Aframe "
        "apptainer images for sub-projects"
    )

    parser.add_argument(
        "projects",
        nargs="*",
        default=PROJECTS,
        help="List of projects to build. "
        f"Default is all: {', '.join(PROJECTS)}",
    )

    parser.add_argument(
        "--container-root",
        type=Path,
        default=Path(os.getenv("AFRAME_CONTAINER_ROOT", "")),
        help="Path to the container root directory. "
        "Defaults to the $AFRAME_CONTAINER_ROOT environment variable.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,  # Set a default number of workers
        help="Maximum number of concurrent builds. Can be useful to set if "
        "your local TMPDIR is being overfilled when building containers. "
        "Default is `None`.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(args.projects, args.container_root)


if __name__ == "__main__":
    main()