#!/bin/bash python3
import serial
import subprocess
import time
import os
import sys

# --- Configuration ---
UART_PORT = '/dev/ttyS0'  # Adjust this to your Raspberry Pi's UART port
                          # Common alternatives: '/dev/ttyAMA0', '/dev/serial0'
BAUD_RATE = 9600          # Adjust to your device's baud rate
ENCODING = 'utf-8'        # Or 'ascii' depending on your communication
DELIMITER = '\n'          # Or '\r\n' or whatever your command termination is

# Define commands and the bash scripts they trigger
COMMAND_MAP = {
    "TRIGGER_ACTION_1": "/home/pi/my_command_script.sh",
    "TRIGGER_ACTION_2": "/home/pi/another_script.sh", # Example for another command
    # Add more commands and their corresponding script paths here
}

# Log file for the Python script's activity
PYTHON_LOG_FILE = "/var/log/uart_listener.log"

# --- Logging Setup ---
def log_message(message):
    """Simple logging function."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(PYTHON_LOG_FILE, 'a') as f:
        f.write(log_entry)
    print(log_entry.strip()) # Also print to console if not daemonized

# --- Main Listener Function ---
def uart_listener_main():
    log_message("UART Listener started.")
    ser = None
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1) # Timeout for non-blocking read
        log_message(f"Successfully opened serial port: {UART_PORT} at {BAUD_RATE} baud.")

        read_buffer = ""
        while True:
            if ser.in_waiting > 0:
                # Read bytes and decode
                read_byte = ser.read(ser.in_waiting).decode(ENCODING, errors='ignore')
                read_buffer += read_byte

                # Process buffer line by line (or by delimiter)
                while DELIMITER in read_buffer:
                    command_line, read_buffer = read_buffer.split(DELIMITER, 1)
                    command_line = command_line.strip() # Remove leading/trailing whitespace

                    if command_line:
                        log_message(f"Received command: '{command_line}'")
                        if command_line in COMMAND_MAP:
                            script_path = COMMAND_MAP[command_line]
                            log_message(f"Executing bash script: {script_path}")
                            try:
                                # Using subprocess.run for simple command execution
                                # capture_output=True, text=True to get stdout/stderr
                                result = subprocess.run([script_path], capture_output=True, text=True, check=True)
                                log_message(f"Bash script stdout:\n{result.stdout.strip()}")
                                if result.stderr:
                                    log_message(f"Bash script stderr:\n{result.stderr.strip()}")
                                log_message(f"Bash script exited with code: {result.returncode}")
                            except subprocess.CalledProcessError as e:
                                log_message(f"Error executing bash script '{script_path}': {e}")
                                log_message(f"Script stdout (on error):\n{e.stdout.strip()}")
                                log_message(f"Script stderr (on error):\n{e.stderr.strip()}")
                            except FileNotFoundError:
                                log_message(f"Error: Bash script not found at {script_path}")
                            except Exception as e:
                                log_message(f"An unexpected error occurred while running script: {e}")
                        else:
                            log_message(f"Unknown command received: '{command_line}'")
            time.sleep(0.1) # Small delay to prevent busy-waiting
    except serial.SerialException as e:
        log_message(f"Serial port error: {e}")
        # Consider adding a retry mechanism here
    except KeyboardInterrupt:
        log_message("UART Listener stopped by user (KeyboardInterrupt).")
    except Exception as e:
        log_message(f"An unhandled error occurred: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            log_message("Serial port closed.")

# --- Daemonization (Optional) ---
# This part makes the script run as a proper background service.
# Requires 'python-daemon' library: pip install python-daemon
try:
    import daemon
    from daemon import pidfile
except ImportError:
    daemon = None
    log_message("python-daemon library not found. Running in foreground. Install with 'pip install python-daemon' for proper daemonization.")

if __name__ == "__main__":
    if daemon:
        PID_FILE = "/var/run/uart_listener.pid" # PID file for daemon management
        # Redirect stdout/stderr to log file when daemonized
        # This is crucial for debugging when running as a daemon
        stdout_logger = open(PYTHON_LOG_FILE, 'a')
        stderr_logger = open(PYTHON_LOG_FILE, 'a')

        # Ensure the log file directory exists (e.g., /var/log/)
        os.makedirs(os.path.dirname(PYTHON_LOG_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)

        with daemon.DaemonContext(
            pidfile=pidfile.TimeoutPidfile(PID_FILE),
            stdout=stdout_logger,
            stderr=stderr_logger,
            umask=0o002,
            working_directory='/', # Set a safe working directory
            files_preserve=[stdout_logger.fileno(), stderr_logger.fileno()]
        ):
            uart_listener_main()
    else:
        # If daemon library not available, run in foreground
        uart_listener_main()