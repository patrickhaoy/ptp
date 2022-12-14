import abc
from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger, eval_util
from rlkit.core.logging import append_log
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    return times


class BaseOfflineRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            evaluation_env,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            num_epochs,
            evaluation_get_diagnostic_functions=None,
            eval_epoch_freq=1,
            eval_only=False,
    ):
        self.trainer = trainer
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.post_train_funcs = []
        self.post_epoch_funcs = []
        self.epoch = self._start_epoch
        self.num_epochs = num_epochs
        if evaluation_get_diagnostic_functions is None:
            evaluation_get_diagnostic_functions = [
                eval_util.get_generic_path_information,
            ]
            if hasattr(self.eval_env, 'get_diagnostics'):
                evaluation_get_diagnostic_functions.append(
                    self.eval_env.get_diagnostics)
        self._eval_get_diag_fns = evaluation_get_diagnostic_functions

        self._eval_epoch_freq = eval_epoch_freq
        self._eval_only = eval_only

    def train(self):
        timer.return_global_times = True
        for _ in range(self.num_epochs):
            self._begin_epoch()
            timer.start_timer('saving')
            logger.save_itr_params(self.epoch, self._get_snapshot())
            timer.stop_timer('saving')
            log_dict, _ = self._train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self._end_epoch()
        logger.save_itr_params(self.epoch, self._get_snapshot())

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, self.epoch)

        self.eval_data_collector.end_epoch(self.epoch)
        self.replay_buffer.end_epoch(self.epoch)
        self.trainer.end_epoch(self.epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, self.epoch)
        self.epoch += 1

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _get_diagnostics(self):
        timer.start_timer('logging', unique=False)
        algo_log = OrderedDict()
        append_log(algo_log, self.replay_buffer.get_diagnostics(),
                   prefix='replay_buffer/')
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        # Eval
        if self.epoch % self._eval_epoch_freq == 0:
            self._prev_eval_log = OrderedDict()
            eval_diag = self.eval_data_collector.get_diagnostics()
            self._prev_eval_log.update(eval_diag)
            append_log(algo_log, eval_diag, prefix='eval/')
            eval_paths = self.eval_data_collector.get_epoch_paths()
            for fn in self._eval_get_diag_fns:
                addl_diag = fn(eval_paths)
                self._prev_eval_log.update(addl_diag)
                append_log(algo_log, addl_diag, prefix='eval/')
        else:
            append_log(algo_log, self._prev_eval_log, prefix='eval/')

        append_log(algo_log, _get_epoch_timings())
        algo_log['epoch'] = self.epoch
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            algo_log['RAM Usage (Mb)'] = int(process.memory_info().rss / 1000000)
        except ImportError:
            pass
        timer.stop_timer('logging')
        return algo_log

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass


class BatchOfflineRLAlgorithm(BaseOfflineRLAlgorithm):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            *args,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                timer.start_timer('training', unique=False)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False
