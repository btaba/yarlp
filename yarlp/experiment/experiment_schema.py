schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Experiment JSON Schema",
    "type": "object",
    "description": "The JSON Schema for an experiment specification",
    "properties": {
        "agents": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "$ref": "#/definitions/agent"
            }
        },
        "envs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "$ref": "#/definitions/env"
            }
        }
    },
    "required": ["agents", "envs"],
    "definitions": {
        "agent": {
            "properties": {
                "type": {
                    "type": "string",
                    "description": "the class name of the agent you want to train"
                },
                "training_epochs": {
                    "type": "integer",
                    "description": "the number of training epochs",
                    "minimum": 1
                },
                "testing_epochs": {
                    "type": "integer",
                    "description": "the number of testing epochs",
                    "minimum": 0
                },
                "params": {"type": "object"},
                "training_params": {"type": "object"}
            },
            "required": ["type", "training_epochs", "testing_epochs",
                         "params", "training_params"]
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
                "repeats": {
                    "type": "integer",
                    "description": "the number of times to run an agent on this environment",
                    "minimum": 1
                },
                "normalize_obs": {
                    "type": "boolean",
                    "description": "whether to normalize the observation space",
                }
            },
            "required": ["name", "repeats"]
        }
    },
}