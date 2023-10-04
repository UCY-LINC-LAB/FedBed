import functools
import os
from datetime import datetime
import sys
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import json

from flwr.common import GetPropertiesIns

profile_prefix = os.getenv("FL_PROFILE_PREFIX", "")

@dataclass
class Profiler:

    identifier: str
    log_size: bool = False
    store_per_run: bool = False
    profiles: ClassVar = {}
    path: ClassVar[str] = "/profile"
    has_executed: ClassVar[bool] = False

    @staticmethod
    def __actualsize(input_obj):
        memory_size = 0
        ids = set()
        objects = [input_obj]
        while objects:
            new = []
            for obj in objects:
                if id(obj) not in ids:
                    ids.add(id(obj))
                    memory_size += sys.getsizeof(obj)
                    new.append(obj)
            objects = gc.get_referents(*new)
        return memory_size

    @classmethod
    def add_to_dict(cls, name, elapsed_time, output_size):
        cls.profiles[name] = elapsed_time
        if output_size is not None:
            cls.profiles[f"{name}_size"] = output_size


    def __call__(self, func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            tic = datetime.now()
            value = func(*args, **kwargs)
            toc = datetime.now()
            elapsed_time = toc - tic
            output_size = None
            if self.log_size:
                output_size = Profiler.__actualsize(value)
            total_seconds = elapsed_time.total_seconds()
            Profiler.add_to_dict(self.identifier, total_seconds if total_seconds >= 0.0001 else 0.0, output_size)
            if self.store_per_run:
                Profiler.log_metrics()
            return value
        return wrapper_timer

    @classmethod
    def log_metrics(cls, filename="profile.jl"):
        cur_prefix = f"{profile_prefix}/" if profile_prefix != "" else ""
        file = f"{Profiler.path}/{cur_prefix}{filename}"
        cls.profiles["timestamp"] = f"{datetime.now()}"
        if not os.path.exists(Profiler.path):
            os.makedirs(Profiler.path)
        if not os.path.exists(f"{Profiler.path}/{cur_prefix}"):
            os.makedirs(f"{Profiler.path}/{cur_prefix}")
        if not cls.has_executed:
            [f.unlink() for f in Path(f"{Profiler.path}/{cur_prefix}").glob("*") if f.is_file()]
            cls.has_executed = True
        if not os.path.exists(Profiler.path):
            os.makedirs(Profiler.path)
        with open(file, "a") as f:
            json.dumps(cls.profiles)
            f.write(json.dumps(cls.profiles) + "\n")
        cls.profiles = {}

    @classmethod
    def log_metric(cls, metric: str, value: float):
        cls.profiles[metric] = value

@dataclass
class ClientsLogger:

    ips_to_hostnames = {}
    path: ClassVar[str] = "/profile"

    def get_hostname_from_client(self, client):
        res = self.ips_to_hostnames.get(client.cid)
        if res is not None:
            return res
        res = client.get_properties(ins=GetPropertiesIns(config={}), timeout=None).properties.get("hostname", client.cid)
        self.ips_to_hostnames[client.cid] = None
        return res

    def log_selected_clients(self, func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            value = func(*args, **kwargs)

            res = {
                "timestamp": f"{datetime.now()}",
                "clients": [self.get_hostname_from_client(i[0]) for i in value]
                }
            self.write("log_selected_clients.jl", res)
            return value
        return _func

    def write(self, filename, obj):
        cur_prefix = f"{profile_prefix}/" if profile_prefix != "" else ""
        filename = f"{Profiler.path}/{cur_prefix}/{filename}"
        with open(filename, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def log_all_clients(self, func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            _self = args[0]
            res = {
                "timestamp": f"{datetime.now()}",
                "clients": [self.get_hostname_from_client(i) for _, i in _self._client_manager.all().items()]
                }
            self.write("log_all_clients.jl", res)
            value = func(*args, **kwargs)
            return value

        return _func