import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class DockerManager:
    """Simplified Docker container manager for trace collection."""
    
    def __init__(self, project_root: Path, image_name: str = "bb-tracer"):
        self.project_root = Path(project_root)
        self.collection_dir = self.project_root / "collection"
        self.image_name = image_name
        self.container_id: Optional[str] = None
    
    def _copy_required_binaries(self) -> bool:
        """Copy DynamoRIO bin64 directory and logger binary to collection/bin/."""
        bin_dir = self.collection_dir / "bin"
        bin_dir.mkdir(exist_ok=True)

        try:
            dirs_to_copy = ["bin64", "lib32", "lib64"]
            for dir_name in dirs_to_copy:
                # Create destination directory
                dst_dir = bin_dir / dir_name
                dst_dir.mkdir(exist_ok=True)

                # Copy source directory
                src_dir = self.project_root / "vendor" / "drio-11.90" / dir_name
                if src_dir.exists():
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
                else:
                    print(f"Warning: DynamoRIO {dir_name} directory not found at {src_dir}")
                    return False

            for file in (bin_dir / "bin64").iterdir():
                if file.is_file() and file.name in ['drrun', 'drconfig', 'drcontrol']:
                    file.chmod(0o755)

            # copy logger
            logger_src = self.project_root / "target" / "release" / "liblogger.so"
            logger_dst = bin_dir / "liblogger.so"
            if logger_src.exists():
                shutil.copy2(logger_src, logger_dst)
            else:
                print(f"Warning: Logger library not found at {logger_src}")
                return False
            
            return True
        except Exception as e:
            print(f"Failed to copy binaries: {e}")
            return False
    
    def _ensure_image(self) -> bool:
        """Build image if it doesn't exist."""
        dockerfile_path = self.collection_dir / "Dockerfile"
        
        try:
            # check if image exists
            result = subprocess.run(
                ["docker", "images", "-q", self.image_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                return True
            
            # build image (using collection dir as context)
            subprocess.run(
                [
                    "docker", "build", "-f", str(dockerfile_path),
                    "-t", self.image_name, str(self.collection_dir)
                ],
                check=True
            )
            
            return True
        except:
            return False
    
    def start_container(self) -> bool:
        """Start container with collection directory mounted."""
        if not self._copy_required_binaries():
            return False
            
        if not self._ensure_image():
            return False
            
        try:
            result = subprocess.run(
                [
                    "docker", "run", "-d", "--rm",
                    "-v", f"{self.collection_dir}:/workspace",
                    "-w", "/workspace",
                    self.image_name,
                    "sleep", "3600"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.container_id = result.stdout.strip()
            return True
        except:
            return False
    
    def execute_command(self, command: str, working_dir: str = "/workspace", timeout: int = 30) -> tuple[bool, str, str]:
        """Execute command in container."""
        if not self.container_id:
            return False, "", "No container running"
        
        try:
            result = subprocess.run(
                [
                    "docker", "exec", "-w", working_dir,
                    self.container_id, "bash", "-c", command
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {timeout}s"
        except:
            return False, "", "Command failed"
    
    def create_work_dir(self, program_id: str) -> bool:
        """Create work directory for program."""
        success, _, _ = self.execute_command(f"mkdir -p {self.get_work_path(program_id)}")
        return success

    def remove_work_dir(self, program_id: str) -> bool:
        """Remove work directory for program."""
        success, _, _ = self.execute_command(f"rm -rf {self.get_work_path(program_id)}")
        return success
    
    def get_work_path(self, program_id: str) -> str:
        """Get work directory path for program."""
        return f"/workspace/work/{program_id}"
    
    def stop_container(self):
        """Stop container."""
        if self.container_id:
            try:
                subprocess.run(["docker", "stop", self.container_id], capture_output=True, timeout=10)
            except:
                pass
            self.container_id = None
    
    def __enter__(self):
        if not self.start_container():
            raise RuntimeError("Failed to start container")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_container()