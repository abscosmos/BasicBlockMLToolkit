#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from tqdm import tqdm

from collection.config import ConfigLoader
from collection.docker_manager import DockerManager
from collection.executor import CommandExecutor


def main():
    parser = argparse.ArgumentParser(description="Collect basic block traces using Docker")
    parser.add_argument("--programs-dir", default="programs", help="Directory containing JSON program definitions")
    parser.add_argument("--timeout", type=int, default=30, help="Command timeout in seconds")
    parser.add_argument("--program", help="Run specific program by ID")
    args = parser.parse_args()
    
    # setup paths
    collection_dir = Path(__file__).parent
    programs_dir = collection_dir / args.programs_dir
    project_root = collection_dir.parent
    
    print("Basic Block Trace Collection:")
    print(f"programs: {programs_dir}")
    print(f"root: {project_root}")
    
    try:
        # load configurations
        loader = ConfigLoader(programs_dir)
        configs = loader.load_all_configs()
        
        # filter to specific program if requested
        if args.program:
            if args.program not in configs:
                print(f"Error: Program '{args.program}' not found")
                return 1
            configs = {args.program: configs[args.program]}
        
        print(f"Loaded {len(configs)} program configurations")
        
        # start docker container
        print("\nStarting Docker container...")
        with DockerManager(project_root) as docker:
            executor = CommandExecutor(docker, timeout=args.timeout)
            
            # execute each program
            start_time = time.time()
            results = []
            successful_programs = 0
            total_traces = 0
            
            # calculate total trace commands across all programs
            total_trace_commands = sum(len(config.trace_commands) for config in configs.values())
            
            with tqdm(total=total_trace_commands, desc="Collecting traces", position=0, leave=True) as pbar:
                for program_id, config in configs.items():
                    result = executor.execute_program(config)
                    results.append(result)
                    
                    # show trace command outputs
                    if result["trace_results"]:
                        # clear previous output and show current program
                        tqdm.write(f"\n{program_id} trace outputs:", file=None)
                        for i, trace_result in enumerate(result["trace_results"]):
                            status = "✓" if trace_result["success"] else "✗"
                            tqdm.write(f"  {status} Command {i}: {trace_result['command']}")
                            if trace_result["stdout"]:
                                tqdm.write(f"    stdout: {trace_result['stdout'][:200]}...")
                            if trace_result["stderr"] and trace_result["success"]:
                                tqdm.write(f"    stderr: {trace_result['stderr'][:200]}...")
                            
                            # update progress bar for each trace command
                            pbar.update(1)
                    
                    if result["success"]:
                        successful_programs += 1
                    
                    total_traces += result["successful_traces"]
                    
                    # update progress bar description with current stats
                    pbar.set_postfix({
                        'successful': total_traces,
                        'programs': f"{successful_programs}/{len(results)}"
                    })
            
            # final summary
            elapsed = time.time() - start_time
            print(f"\n=== Collection Complete ===")
            print(f"Total programs: {len(configs)}")
            print(f"Successful programs: {successful_programs}")
            print(f"Failed programs: {len(configs) - successful_programs}")
            print(f"Total traces generated: {total_traces}")
            print(f"Elapsed time: {elapsed:.1f}s")
            print(f"Traces saved to: {collection_dir / 'traces'}")
            
            # show failed programs
            failed = [r for r in results if not r["success"]]
            if failed:
                print(f"\nFailed programs:")
                for result in failed:
                    print(f"  - {result['program_id']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())