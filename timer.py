import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """
    Manages individual time measurements through program run
    """

    def __init__(self):
        self.timers = dict()

    def start_measure(self, name):
        if name not in self.timers:
            self.timers[name] = Measure()

        self.timers[name].start()

    def end_measure(self, name):
        if name not in self.timers:
            raise TimerError(f"Timer was not yet created")

        self.timers[name].stop()

    def get_measurement(self, name):
        if name not in self.timers:
            #raise TimerError(f"Timer was not yet created")
            return 0

        return f"{self.timers[name].elapsed_time:0.4f} s"


class Measure:
    """
    Represents one timer measurement
    """

    def __init__(self):
        self._start_time = None
        self.elapsed_time = 0

    def start(self):
        """
        Start a new timer
        """
        self._start_time = time.perf_counter()

    def stop(self):
        """
        Stop the timer
        """
        actual_elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.elapsed_time += actual_elapsed_time
