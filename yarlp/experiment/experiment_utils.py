import os
import json
import inspect
from datetime import datetime


class ExperimentUtils:
    @staticmethod
    def get_agent_cls_dict():
        import yarlp.agent
        clsmembers = inspect.getmembers(yarlp.agent, inspect.isclass)
        return dict(clsmembers)

    @staticmethod
    def get_datetime_str():
        return datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

    @staticmethod
    def create_log_directory(name, prepend_dir_name):
        name += ExperimentUtils.get_datetime_str()
        full_dir = os.path.join(prepend_dir_name, name)
        os.makedirs(full_dir)
        return full_dir

    @staticmethod
    def save_spec_to_dir(spec, dir):
        file_path = os.path.join(dir, 'spec.json')
        json.dump(spec, open(file_path, 'w'), indent=4)

    @staticmethod
    def create_video_callable(video):
        assert isinstance(video, bool)

        if not video:
            return False

        return None
