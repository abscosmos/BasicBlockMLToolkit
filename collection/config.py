import json
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProgramConfig:
    """Represents a single program's trace collection configuration."""
    program_id: str
    setup_commands: list[str]
    trace_commands: list[str]
    cleanup_commands: list[str]
    
    @classmethod
    def from_dict(cls, data: dict, filename: str) -> 'ProgramConfig':
        """Create ProgramConfig from dictionary, validating required fields."""
        # validate required fields
        required_fields = ['program_id', 'setup_commands', 'trace_commands']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in {filename}")
        
        # validate program_id format
        program_id = data['program_id']
        if not isinstance(program_id, str) or not program_id.strip():
            raise ValueError(f"program_id must be non-empty string in {filename}")
        
        if not re.match(r'^[a-zA-Z0-9_]+$', program_id):
            raise ValueError(f"program_id '{program_id}' must contain only alphanumeric characters and underscores in {filename}")
        
        # validate command arrays
        setup_commands = data['setup_commands']
        if not isinstance(setup_commands, list):
            raise ValueError(f"setup_commands must be array in {filename}")
        
        trace_commands = data['trace_commands']
        if not isinstance(trace_commands, list) or len(trace_commands) == 0:
            raise ValueError(f"trace_commands must be non-empty array in {filename}")
        
        # validate all commands are non-empty strings
        for i, cmd in enumerate(setup_commands):
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"setup_commands[{i}] must be non-empty string in {filename}")
        
        for i, cmd in enumerate(trace_commands):
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"trace_commands[{i}] must be non-empty string in {filename}")
        
        cleanup_commands = data.get('cleanup_commands', [])
        if not isinstance(cleanup_commands, list):
            raise ValueError(f"cleanup_commands must be array in {filename}")
        
        for i, cmd in enumerate(cleanup_commands):
            if not isinstance(cmd, str) or not cmd.strip():
                raise ValueError(f"cleanup_commands[{i}] must be non-empty string in {filename}")
        
        return cls(
            program_id=program_id,
            setup_commands=setup_commands,
            trace_commands=trace_commands,
            cleanup_commands=cleanup_commands
        )


def _load_single_config(json_file: Path) -> ProgramConfig:
    """Load and validate a single JSON configuration file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    if not isinstance(data, dict):
        raise ValueError("JSON must contain an object at root level")

    return ProgramConfig.from_dict(data, json_file.name)


class ConfigLoader:
    """Loads and validates JSON program configurations."""
    
    def __init__(self, programs_dir: Path):
        self.programs_dir = Path(programs_dir)
    
    def load_all_configs(self) -> dict[str, ProgramConfig]:
        """
        Load all JSON program configurations from programs directory.
            
        Returns:
            Dictionary mapping program_id to ProgramConfig
            
        Raises:
            FileNotFoundError: If programs directory doesn't exist
            ValueError: If configs are invalid or have duplicate program_ids
        """
        if not self.programs_dir.exists():
            raise FileNotFoundError(f"Programs directory not found: {self.programs_dir}")
        
        configs = {}
        errors = []
        
        # find all JSON files
        json_files = list(self.programs_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {self.programs_dir}")
        
        for json_file in json_files:
            try:
                config = _load_single_config(json_file)
                
                # check for duplicate program_ids
                if config.program_id in configs:
                    existing_file = None
                    for other_file in json_files:
                        if other_file != json_file:
                            try:
                                other_config = _load_single_config(other_file)
                                if other_config.program_id == config.program_id:
                                    existing_file = other_file
                                    break
                            except:
                                continue
                    
                    raise ValueError(
                        f"Duplicate program_id '{config.program_id}' found in {json_file.name} "
                        f"and {existing_file.name if existing_file else 'another file'}"
                    )
                
                configs[config.program_id] = config
                
            except Exception as e:
                errors.append(f"Error loading {json_file.name}: {e}")
        
        # report all errors at once
        if errors:
            error_msg = "Configuration errors found:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)
        
        if not configs:
            raise ValueError("No valid program configurations found")
        
        return configs

    def validate_single_file(self, json_file: Path) -> ProgramConfig:
        """Validate a single JSON file without checking for duplicates."""
        return _load_single_config(json_file)