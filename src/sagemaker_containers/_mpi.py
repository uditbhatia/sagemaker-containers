import argparse
import os
import shlex
import socket
import stat
import subprocess
import sys
import textwrap
import time

from retrying import retry

import sagemaker_containers
from sagemaker_containers import _logging
from sagemaker_containers._mapping import to_cmd_args
from sagemaker_containers._timeout import timeout

logger = _logging.get_logger()

_MPI_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
touch %s
%s
EXIT_CODE=$?
touch %s
exit ${EXIT_CODE}
"""

# MPI files.
_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"

# MPI Configurations
SAGEMAKER_MPI_NUM_PROCESSES_PER_HOST = "sagemaker_mpi_num_of_processes_per_host"
SAGEMAKER_MPI_CUSTOM_MPI_OPTIONS = "sagemaker_mpi_custom_mpi_options"


def _change_hostname(current_host):  # type: (str) -> None
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.
    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("/change-hostname.sh {}".format(current_host))


def _start_ssh_daemon():  # type: () -> None
    """Starts the ssh deamon
    """
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _setup_mpi_environment(env):  # type: (TrainingEnv) -> None
    """Setup MPI environment, i.e. executing change hostname script and starting ssh deamon.
    """
    _change_hostname(current_host=env.current_host)
    _start_ssh_daemon()


def _can_connect(host, port, s):  # type: (str,int,socket.socket) -> bool
    """Checks if the connection to provided ``host`` and ``port`` is possible or not.
    """
    try:
        print("Testing connection to host {}".format(host))
        s.connect((host, port))
        s.close()
        print("Can connect to host {}".format(host))
        return True
    except socket.error:
        print("Can't connect to host {}".format(host))
        return False


def _create_mpi_script(hyperparameters, channel_input_dirs, train_script,
                       code_dir, mpi_script_path, mpi_is_running_flag_file,
                       mpi_is_finished_flag_file):  # type: (dict, dict, str, str, str, str, str) -> None
    """Creates a MPI script with user provided information.
        For distributed training: the 'master node' runs mpirun with this script, '/mpi_script.sh'.
        This script creates a file '/mpi_is_running' that worker nodes use to determine whether training # (started by
        MPI from the master node) is still running. Processes on worker nodes use # /mpi_is_finished file to determine
        when to exit.
    Args:
        hyperparameters (dict): Hyperparameters for trainig job.
        channel_input_dirs (dict): Channel Input directories for training job.
        train_script (str): Training script to be executed via MPI.
        code_dir (str): Path to directory containing ``train_script``
        mpi_script_path (str): Path where the MPI script is created.
        mpi_is_running_flag_file (str): Path to the file used to flag the MPI is running status.
        mpi_is_finished_flag_file (str): Path to the file used to flag the MPI is finished status.
    """

    python_cmd = [sys.executable, "{}/{}".format(code_dir, train_script)]
    python_cmd.extend(to_cmd_args(hyperparameters))
    python_cmd.extend(to_cmd_args(channel_input_dirs))

    content = textwrap.dedent(_MPI_SCRIPT_TEMPLATE % (mpi_is_running_flag_file,
                                                      ' '.join(python_cmd),
                                                      mpi_is_finished_flag_file))

    with open(mpi_script_path, 'w') as w:
        w.write(content)

    st = os.stat(mpi_script_path)
    os.chmod(mpi_script_path, st.st_mode | stat.S_IEXEC)

    logger.info("MPI script created at: {}".format(mpi_script_path))


class MPIMaster(object):
    """MPI Master
        Args:
            env (TrainingEnv): an instance of the training environment.
            process_per_host (int): Number of processes per host to be executed by MPI
            mpi_script_path (str): Path where the MPI script is created.
            custom_mpi_options (str): Custom MPI options provided by user, this string will be parsed into arguments and
             all known argument's value will override the defaults whereas unknown arguments will be appended directly
             to MPI command.
    """

    def __init__(self, env, process_per_host, mpi_script_path, custom_mpi_options=None):
        self.env = env
        self.process_per_host = process_per_host
        self.mpi_script_path = mpi_script_path
        self.custom_mpi_options = custom_mpi_options

    def _wait_for_worker_nodes_to_start_sshd(self, hosts, interval=1,
                                             timeout_in_seconds=180):  # type: (list, int, int) -> None
        """Wait for worker nodes to start their ssh deamon to allow MPI communication.
        """
        with timeout(seconds=timeout_in_seconds):
            while hosts:
                for host in hosts:
                    ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    if _can_connect(host, 22, ssh_socket):
                        hosts.remove(host)
                time.sleep(interval)
            logger.info("Worker node available for communication: {}".format(len(hosts) == 0))

    def _run_mpi_on_all_nodes(self):  # type: () -> None
        """Run MPI command to execute MPI_SCRIPT on all hosts.
        """
        mpi_command = self._build_mpi_command()
        cmd = shlex.split(mpi_command)

        _logging.log_script_invocation(cmd, self.env.to_env_vars(), logger)

        logger.info("MPI Command: {}".format(mpi_command))
        with open(self.mpi_script_path) as f:
            logger.info('Running user script:\n\n%s', f.read())

        subprocess.check_call(cmd)

    def _parse_custom_mpi_options(self):  # type: () -> (ArgumentParser,list)
        """Parse custom MPI options provided by user. Known options default value will be overriden and unknown options
        would be identified separately."""

        parser = argparse.ArgumentParser()
        parser.add_argument('--NCCL_DEBUG', default="INFO", type=str)

        if self.custom_mpi_options:
            return parser.parse_known_args(self.custom_mpi_options.split())
        else:
            return parser.parse_known_args([])

    def _build_mpi_command(self):  # type: ()-> None
        """Build MPI command with all required MPI flags for sagemaker infrastructure, environment variables, provided
        hyeprparameters and custom mpi options.
        """
        num_hosts = len(self.env.hosts)
        num_processes = self.process_per_host * num_hosts

        # By default, use one process per GPU, or one process per node (if training with CPU).
        host_list = self.env.hosts if self.process_per_host == 1 else \
            [host + ':{}'.format(self.process_per_host) for host in self.env.hosts]

        logger.info("Env Hosts: {} Hosts: {} process_per_hosts: {} num_processes: {}".format(self.env.hosts, host_list,
                                                                                             self.process_per_host,
                                                                                             num_processes))

        overriden_known_options, additional_options = self._parse_custom_mpi_options()

        interface_name = self.env.network_interface_name
        logger.info("Network interface name: {}".format(interface_name))

        mpi_command = 'mpirun --host {}'.format(",".join(host_list)) \
                      + " -np {} ".format(num_processes) \
                      + " --allow-run-as-root" \
                      + " --display-map" \
                      + " --tag-output" \
                      + " -mca btl_tcp_if_include {}".format(interface_name) \
                      + " -mca oob_tcp_if_include {}".format(interface_name) \
                      + " -x NCCL_SOCKET_IFNAME={}".format(interface_name) \
                      + " --mca plm_rsh_no_tree_spawn 1" \
                      + " -mca orte_abort_on_non_zero_status 1" \
                      + " -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG={}".format(overriden_known_options.NCCL_DEBUG) \
                      + " -x LD_LIBRARY_PATH -x PATH" \
                      + " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) \
                      + " {}".format(" ".join(additional_options))

        credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
        for v in credential_vars:
            if v in os.environ:
                mpi_command += " -x {}".format(v)

        for name, value in self.env.to_env_vars().items():
            mpi_command += ' -x {}="{}"'.format(name, value)

        mpi_command += " {}".format(self.mpi_script_path)

        return mpi_command

    def is_master(self, hosts, current_host):  # type: (list, str) -> bool
        """Checks if the current host is master or worker.
        """
        _is_master = current_host == sorted(list(hosts))[0]
        logger.info("Is current host: {} among hosts: {} master: {}".format(current_host, hosts, _is_master))
        return _is_master

    def __call__(self):  # type: () -> None
        self._wait_for_worker_nodes_to_start_sshd(self.env.hosts.copy())
        self._run_mpi_on_all_nodes()


class MPIWorker(object):
    """ MPI Worker
        Args:
            current_host (str): Current host id.
    """

    def __init__(self, current_host):
        self._current_host = current_host

    @retry(stop_max_delay=30000 * 1000, wait_fixed=1000, retry_on_result=lambda result: result is False)
    def _wait_for_mpi_to_start_running(self):  # type: () -> None
        """Wait and retry loop until the MPI training starts on this worker.
        """
        return os.path.isfile(_MPI_IS_RUNNING)

    @retry(wait_fixed=5000, retry_on_result=lambda result: result is False)
    def _wait_until_mpi_stops_running(self):  # type: () -> None
        """Wait and retry loop until the MPI training is finished on this worker.
        """
        return os.path.isfile(_MPI_IS_FINISHED)

    def __call__(self):  # type: () -> None
        logger.info("Worker node {} is waiting for MPI to start training process".format(self._current_host))
        self._wait_for_mpi_to_start_running()

        logger.info("MPI started training process on worker node {}".format(self._current_host))

        self._wait_until_mpi_stops_running()
        logger.info("Training process started by MPI on worker node {} stopped".format(self._current_host))


def mpi_run(train_script, code_dir):  # type: (str, str, int, str) -> None
    """It runs the mpi command to launch user provided training script.
        Args:
            train_script (str): Train script to executed by the ``MPILauncher``
            code_dir (str): Path to directory containing ``train_script``
        """
    env = sagemaker_containers.training_env()

    num_of_processes_per_host = env.additional_framework_parameters.get(SAGEMAKER_MPI_NUM_PROCESSES_PER_HOST, 1)
    custom_mpi_options = env.additional_framework_parameters.get(SAGEMAKER_MPI_CUSTOM_MPI_OPTIONS, None)

    logger.info("MPI requested for train_script: {} code_dir: {} process per hosts: {} and custom_mpi_options: {}"
                .format(train_script, code_dir, num_of_processes_per_host, custom_mpi_options))

    _setup_mpi_environment(env)
    _create_mpi_script(env.hyperparameters, env.channel_input_dirs, train_script, code_dir, _MPI_SCRIPT,
                       _MPI_IS_RUNNING, _MPI_IS_FINISHED)

    mpi_master = MPIMaster(env, num_of_processes_per_host, _MPI_SCRIPT, custom_mpi_options)
    if mpi_master.is_master(env.hosts, env.current_host):
        logger.info("Inside Master")
        mpi_master()
    else:
        logger.info("Inside Worker")
        MPIWorker(env)()
