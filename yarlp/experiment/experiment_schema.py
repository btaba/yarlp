schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Experiment JSON Schema",
    "type": "object",
    "description": "The JSON Schema for an experiment specification",
    "properties": {
        "runs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "$ref": "#/definitions/experiment"
            }
        }
    },
    "required": ["runs"],
    "definitions": {
        "experiment": {
            "agent": {
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "the class name of the agent you want to train"
                    },
                    "params": {"type": "object"},
                    "training_params": {"type": "object"},
                    "seeds": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["type", "seeds"]
            },
            "env": {
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "the environment name"
                    },
                    "timestep_limit": {
                        "type": "integer"
                    },
                    "normalize_obs": {
                        "type": "boolean",
                        "description": "whether to normalize the observation space"
                    },
                    "is_parallel": {
                        "type": "boolean"
                    },
                    "is_atari": {
                        "type": "boolean"
                    }
                },
                "required": ["name"]
            }
        }
    },
}
