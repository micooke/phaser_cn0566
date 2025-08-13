#!/bin/bash python3
from uart_utils import *

import argparse
try:
    import daemon
    from daemon import pidfile
except ImportError:
    daemon = None

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--daemonise", type=int, default=0, help="Daemonise the uart listener")

# --- Daemonization (Optional) ---
# This part makes the script run as a proper background service.
# Requires 'python-daemon' library: pip install python-daemon

if __name__ == "__main__":
    LOG_FILE='/data/uart_listener.log'
    PID_FILE = "/data/uart_listener.pid" # PID file for daemon management
        
    args = vars(parser.parse_args())
    if (args["daemonise"] == 1) and daemon:
        log_message("UART listener daemonised", log_file=LOG_FILE)

        # Redirect stdout/stderr to log file when daemonized
        # This is crucial for debugging when running as a daemon
        
        stdout_logger = open(LOG_FILE, 'a')
        stderr_logger = open(LOG_FILE, 'a')

        # Ensure the log file directory exists (e.g., /var/log/)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)

        with daemon.DaemonContext(
            pidfile=pidfile.TimeoutPidfile(PID_FILE),
            stdout=stdout_logger,
            stderr=stderr_logger,
            umask=0o002,
            working_directory='/', # Set a safe working directory
            files_preserve=[stdout_logger.fileno(), stderr_logger.fileno()]
        ):
            uart_listener()
    else:
        if (args["daemonise"] == 1) and not daemon:
            log_message("python-daemon library not found. Running in foreground. Install with 'pip install python-daemon' for proper daemonization.", log_file=LOG_FILE)

        log_message("Uart listener started", log_file=LOG_FILE)
        uart_listener()
