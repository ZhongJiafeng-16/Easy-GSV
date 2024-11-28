import time
import torch
from loguru import logger

class RuntimeTracker:
    def __init__(self) -> None:
        self.timings = {}
        self.memory_usages = {}
        self.start_time = None
        self.end_time = None
        self.init_time = time.time()

    def start(self, task_name="default"):
        self.start_time = time.time()  # 记录当前时间
        self.task_name = task_name  # 记录任务名称
        torch.cuda.empty_cache()  # 清空未使用显存
        logger.debug(f"Tracker start to log <{task_name}> task runing time...")

    def end(self):
        if self.start_time is None or self.task_name is None:
            raise RuntimeError("Please call start() first.")

        elapsed_time = time.time() - self.start_time  # 计算所用时间
        self.timings[self.task_name] = elapsed_time

        logger.debug(
            f"Task <{self.task_name}> finish, Time: {elapsed_time:.2f} s.")

        # reset start_time
        self.start_time = None
        self.task_name = None

    def print_all_records(self):
        logger.debug("all tasks runing time:")
        for task, timing in self.timings.items():
            logger.debug(
                f"Task <{task}> finish, time: {timing:.2f} s."
            )
        total_elapsed_time = time.time() - self.init_time
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        logger.debug(f"Total Time: {total_elapsed_time / 3600:.2f} h, Max Memory: {max_reserved:.2f} MB.")