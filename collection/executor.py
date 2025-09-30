from typing import Dict, Any
from collection.config import ProgramConfig
from collection.docker_manager import DockerManager


class CommandExecutor:
    """Executes setup/trace/cleanup commands for programs."""
    
    def __init__(self, docker: DockerManager, timeout: int = 30):
        self.docker = docker
        self.timeout = timeout
    
    def execute_program(self, config: ProgramConfig) -> Dict[str, Any]:
        """Execute all commands for a program."""
        program_id = config.program_id
        
        if not self.docker.create_work_dir(program_id):
            return {"success": False, "error": "Failed to create work directory"}
        
        work_path = self.docker.get_work_path(program_id)

        result = {
            "program_id": program_id,
            "success": True,
            "setup_results": [],
            "trace_results": [],
            "cleanup_results": [],
            "successful_traces": 0
        }
        
        # run setup commands
        for i, cmd in enumerate(config.setup_commands):
            success, stdout, stderr = self.docker.execute_command(cmd, work_path, self.timeout)

            result["setup_results"].append({
                "command": cmd,
                "success": success,
                "stdout": stdout,
                "stderr": stderr
            })

            if not success:
                print(f"Setup command {i} failed for {program_id}: {stderr}")
        
        # run trace commands
        for i, cmd in enumerate(config.trace_commands):
            trace_file = f"/workspace/traces/{program_id}_{i}.trace"
            drrun_cmd = f"/workspace/bin/bin64/drrun -c /workspace/bin/liblogger.so -file={trace_file} -- {cmd}"
            
            success, stdout, stderr = self.docker.execute_command(drrun_cmd, work_path, self.timeout)

            result["trace_results"].append({
                "command": cmd,
                "trace_file": trace_file,
                "success": success,
                "stdout": stdout,
                "stderr": stderr
            })
            
            if success:
                result["successful_traces"] += 1
                print(f"✓ Traced: {program_id}_{i}.trace")
            else:
                print(f"✗ Trace failed for {program_id} command {i}: {stderr}")
                result["success"] = False
        
        # run cleanup commands
        for i, cmd in enumerate(config.cleanup_commands):
            success, stdout, stderr = self.docker.execute_command(cmd, work_path, self.timeout)

            result["cleanup_results"].append({
                "command": cmd,
                "success": success,
                "stdout": stdout,
                "stderr": stderr
            })

            if not success:
                print(f"Cleanup command {i} failed for {program_id}: {stderr}")

        self.docker.remove_work_dir(program_id)
        
        return result