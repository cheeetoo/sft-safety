import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Python equivalent of run.bat to orchestrate docker compose run and artifact collection.",
		add_help=True,
	)
	parser.add_argument(
		"-f",
		"--folder",
		dest="job_label",
		help="Set run label/folder name (default: experiment)",
		default="experiment",
	)
	parser.add_argument(
		"-r",
		"--remove-images",
		dest="clean_after",
		help="Prune Docker images after the run completes",
		action="store_true",
	)
	parser.add_argument(
		"-t",
		"--no-timestamp",
		dest="no_timestamp",
		help="Do not include timestamp in result folder naming",
		action="store_true",
	)
	parser.add_argument(
		"--openai-base-url",
		dest="openai_base_url",
		help="OpenAI API base URL",
		default="http://host.docker.internal:1234/v1"
	)
	parser.add_argument(
		"--openai-api-key",
		dest="openai_api_key",
		help="OpenAI API key",
		default="das"
	)
	parser.add_argument(
		"--openai-model",
		dest="openai_model",
		help="OpenAI model to use",
		default="openai/gpt-oss-120b"
	)
	parser.add_argument(
		"--openai-temperature",
		dest="openai_temperature",
		help="OpenAI model temperature (0.0 for deterministic, 0.7 for sampled decoding)",
		default="0.0"
	)
	return parser.parse_args()


def ensure_working_directory_is_script_dir() -> Path:
	# Mirror `pushd "%~dp0"` behavior from the batch script
	script_dir = Path(__file__).resolve().parent
	os.chdir(script_dir)
	return script_dir


def build_artifact_dir(base_label: str, no_timestamp: bool = False) -> Path:
	base = Path("experiments") / base_label
	if not no_timestamp:
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		base = Path(f"{base}_{stamp}")
	# Recreate directory fresh
	if base.exists():
		shutil.rmtree(base, ignore_errors=True)
	base.mkdir(parents=True, exist_ok=True)
	return base


def append_log(log_file: Path, message: str) -> None:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	with log_file.open("a", encoding="utf-8") as f:
		f.write(message.rstrip("\n") + "\n")


def run_docker_compose(log_file: Path) -> int:
	append_log(log_file, "Starting docker-compose with 20 minute timeout...")
	# The batch script does not actually enforce a timeout; it only logs the intent.
	proc = subprocess.run(
		["docker", "compose", "up", "--build", "--abort-on-container-exit"],
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	# Persist docker compose stdout to a side log like the commented line in .bat
	compose_side_log = log_file.parent / "docker_output.log"
	compose_side_log.write_text(proc.stdout or "", encoding="utf-8")
	return proc.returncode


def resolve_executor_container() -> str | None:
	# container_name was removed from docker-compose.yml so concurrent clones
	# don't collide; resolve this project's executor container (running or
	# exited) by service name instead.
	result = subprocess.run(
		["docker", "compose", "ps", "-a", "-q", "executor"],
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	if result.returncode != 0:
		return None
	ids = result.stdout.split()
	return ids[0] if ids else None


def docker_cp(container_path: str, dest: Path) -> bool:
	result = subprocess.run(
		["docker", "cp", container_path, str(dest)],
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	return result.returncode == 0


def write_fallback_result(dest: Path) -> None:
	fallback = {
		"status": "BUILD_OR_RUN_FAILURE",
		"evaluation": 0,
		"reasoning": "Validation failed due to build or run failure.",
		"error_type": "container_failure",
	}
	dest.write_text(json.dumps(fallback, indent=2), encoding="utf-8")


def main() -> int:
	args = parse_args()
	ensure_working_directory_is_script_dir()

	job_label = args.job_label
	clean_after = bool(args.clean_after)
	no_timestamp = bool(args.no_timestamp)

	artifact_dir = build_artifact_dir(job_label, no_timestamp=no_timestamp)
	run_log = artifact_dir / "run.log"

	# Initialize log
	initial_lines = [
		"Initiating run...",
		f"Run label: {job_label}",
		f"Output directory: {artifact_dir}",
		f"Log file: {run_log}",
		f"Remove images after execution: {str(clean_after).lower()}",
		"==================================",
	]
	for line in initial_lines:
		append_log(run_log, line)

	# Echo to console similar to the batch script
	print("Output directory:")
	print(str(artifact_dir))

	# Set OpenAI environment variables from command line arguments
	os.environ["OPENAI_BASE_URL"] = args.openai_base_url
	os.environ["OPENAI_API_KEY"] = args.openai_api_key
	os.environ["OPENAI_MODEL"] = args.openai_model
	os.environ["OPENAI_TEMPERATURE"] = args.openai_temperature

	# Report chosen OpenAI configuration (convenience)
	print("OpenAI settings:")
	print(f"  BASE_URL = {os.environ.get('OPENAI_BASE_URL')}")
	print(f"  MODEL = {os.environ.get('OPENAI_MODEL')}")
	print(f"  TEMPERATURE = {os.environ.get('OPENAI_TEMPERATURE')}")

	compose_status = run_docker_compose(run_log)

	print("Again Output directory:")
	print(str(artifact_dir))

	if compose_status == 0:
		append_log(run_log, "Docker-compose execution completed successfully.")
	else:
		append_log(run_log, f"Docker-compose execution failed with exit code: {compose_status}")

	# Copy artifacts from the executor container (as in run.bat)
	executor_container = resolve_executor_container()
	if executor_container is None:
		append_log(run_log, "Could not resolve executor container via docker compose ps")

	append_log(run_log, "Attempting to copy result.json from agent container...")
	result_json_path = artifact_dir / "result.json"
	if executor_container and docker_cp(f"{executor_container}:/app/result.json", result_json_path):
		append_log(run_log, "Successfully copied result.json")
	else:
		append_log(run_log, "Failed to copy result.json, creating fallback result")
		write_fallback_result(result_json_path)

	append_log(run_log, "Attempting to copy messages_record.txt from agent container...")
	if executor_container and docker_cp(f"{executor_container}:/app/messages_record.txt", artifact_dir / "messages_record.txt"):
		append_log(run_log, "Successfully copied messages_record.txt")
	else:
		append_log(run_log, "Failed to copy messages_record.txt")

	# Cleanup
	if clean_after:
		append_log(run_log, "Cleaning up containers, volumes, and images...")
		cleanup_cmd = ["docker", "compose", "down", "-v", "--rmi", "all"]
	else:
		append_log(run_log, "Cleaning up containers and volumes...")
		cleanup_cmd = ["docker", "compose", "down", "-v"]

	# Run cleanup (best-effort)
	_ = subprocess.run(cleanup_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

	append_log(run_log, "==================================")
	append_log(run_log, "Submission execution completed.")
	append_log(run_log, f"Results saved to: {artifact_dir}")

	return 0


if __name__ == "__main__":
	sys.exit(main())


