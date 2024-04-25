"""Helper functions for performance timing."""

import numpy as np
import sys
from typing import List


class time_accumulator:
	"""An accumulator object that collects performance timings."""
	def __init__(self, n_breaks: int, labels: List[str] = None):
		"""Initializes an accumulator.

		Args:
			n_breaks: number of breaks that constitute a "loop"
			labels: labels of each breakpoint
		"""
		self.__labels = labels
		self.__n_breaks = n_breaks
		self.__time_arrs = [[] for x in range(n_breaks)]
		self.__count_samples = 0

	def add_batch_times(self, timings: List[float]):
		"""Adds timings of a batch.

		Args:
			timings: List of times

		Raises:
			ValueError if timings are not the correct length.
		"""
		if len(timings) != self.__n_breaks + 1:
			raise ValueError(f'Timer expects {self.__n_breaks + 1} times, received {len(timings)}.')

		deltas = np.asarray(timings)[1:] - np.asarray(timings)[:-1]
		self.add_batch_deltas(deltas)

	def add_batch_deltas(self, deltas: List[float]):
		"""Adds timing deltas for a batch.

		Args:
			deltas: List of time deltas

		Raises:
			ValueError if deltas are not the correct length.
		"""
		if len(deltas) != self.__n_breaks:
			raise ValueError(f'Timer has {self.__n_breaks} breakpoints, received {len(deltas)}.')

		_ = [arr.append(new_val) for arr, new_val in zip(self.__time_arrs, deltas)]
		self.__count_samples += 1

	def print_performance(self, skip_warmup: bool = False, out_stream=sys.stdout):
		"""Prints performance.

		Args:
			skip_warmup: boolean to skip the first batch (typically longer)
			out_stream: output stream to write performance
		"""
		if self.__count_samples >= 1:
			if skip_warmup and self.__count_samples >= 2:
				avg_times = [np.mean(cur_timer[1:]) for cur_timer in self.__time_arrs]
			else:
				avg_times = [np.mean(cur_timer) for cur_timer in self.__time_arrs]
			total_time = np.sum(avg_times)
			for timer_idx in np.arange(self.__n_breaks):
				print(f'{self.__labels[timer_idx]}: {np.round(avg_times[timer_idx], 4)}s ({np.round(avg_times[timer_idx] / total_time, 4)*100}%)', file=out_stream)
			print(f'Overall: {np.round(total_time, 4)}s/batch ({np.round(1/total_time, 4)} FPS)', file=out_stream)
