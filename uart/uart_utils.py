#!/bin/bash python3
import serial
import subprocess
import time
import os
 
# Configuration
UART_RX_PORT = '/dev/ttyUSB0'
UART_TX_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
ENCODING = 'utf-8'
DELIMITER = '\n' # '\n' or '\r\n'

LOG_TYPES = {'i':'[INFO]','d':'[DEBUG]','e':'[ERROR]','s':'[STATUS]','t':'[TEMP]'}

# Define commands and the bash scripts they trigger
COMMAND_MAP = {
    "XCLOCK": "/usr/bin/xclock",
    "XEYES": "/usr/bin/xeyes",
}

def timestamp(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] {message}\n"
    
# Logging Setup
def log_message(message, log_type='i', log_file=None):
    if log_type not in LOG_TYPES:
        log_type = 'i'
    
    message = LOG_TYPES[log_type]+' '+message
    
    if log_file is None:
        log_file = f'/data/{time.strftime("%Y-%m-%d")}-uart.log'
    log_entry = timestamp(message)
    with open(log_file, 'a') as f:
        f.write(log_entry)
    print(log_entry.strip()) # Also print to console if not daemonized

def uart_listener(UART_PORT:str=UART_RX_PORT, UART_BAUD:int=BAUD_RATE):
    log_message("UART listener started", 'd')
    ser = None
    KEEP_ALIVE:bool=True
    try:
        while True:
            try:
                ser = serial.Serial(UART_PORT, UART_BAUD, timeout=1) # Timeout for non-blocking read
                log_message(f"Opened serial port: {UART_PORT} @ {BAUD_RATE} baud", 'd')

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
                                    script_path = COMMAND_MAP[command_line] # named command received
                                else:
                                    script_path = command_line # basic command received?
                                
                                # Try executing the script
                                log_message(f"Executing bash script: {script_path}")
                                try:
                                    # Using subprocess.run for simple command execution
                                    # capture_output=True, text=True to get stdout/stderr
                                    result = subprocess.run([script_path], capture_output=True, text=True, check=True)
                                    log_message(f"Bash script stdout:\n{result.stdout.strip()}")
                                    if result.stderr:
                                        log_message(f"Bash script stderr:\n{result.stderr.strip()}", 'e')
                                    log_message(f"Bash script exited with code: {result.returncode}", 'd')
                                except subprocess.CalledProcessError as e:
                                    log_message(f"Error executing bash script '{script_path}': {e}", 'e')
                                    log_message(f"Script stdout (on error):\n{e.stdout.strip()}", 'e')
                                    log_message(f"Script stderr (on error):\n{e.stderr.strip()}", 'e')
                                except FileNotFoundError:
                                    log_message(f"Error: Bash script not found at {script_path}", 'e')
                                except Exception as e:
                                    log_message(f"An unexpected error occurred: {e}", 'e')
                    time.sleep(0.1) # Small delay to prevent busy-waiting
            except serial.SerialException as e:
                log_message(f"Serial port error: {e}", 'e')
                # Consider adding a retry mechanism here
            except KeyboardInterrupt:
                #[TODO] raise an Exception here
                log_message("UART listener stopped by user (KeyboardInterrupt)", 'd')
            except Exception as e:
                log_message(f"An unhandled error occurred: {e}", 'e')
            finally:
               if ser and ser.is_open:
                   ser.close()
                   log_message("Serial port closed", 'd')
            
            # delay 5seconds until the next attemp
            time.sleep(5.0)
    except KeyboardInterrupt:
        log_message("UART listener stopped by user (KeyboardInterrupt)", 'd')
    except Exception as e:
        log_message(f"An unhandled error occurred: {e}", 'e')

def uart_writer(UART_PORT:str=UART_RX_PORT, UART_BAUD:int=BAUD_RATE, DELAY_S:float=3.0):
    messages = ['Hello, World!']
    
    log_message("UART (dummy) writer started", 'd')
    ser = None
    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=1) # Timeout for non-blocking read
        log_message(f"Opened serial port: {UART_PORT} @ {BAUD_RATE} baud", 'd')

        while True:
            for message in messages:
                # Write bytes
                ser.write(timestamp(message).encode(ENCODING, errors='ignore'))
                time.sleep(DELAY_S)
    except serial.SerialException as e:
        log_message(f"Serial port error: {e}", 'e')
        # Consider adding a retry mechanism here
    except KeyboardInterrupt:
        log_message("UART writer stopped by user (KeyboardInterrupt)", 'd')
    except Exception as e:
        log_message(f"An unhandled error occurred: {e}", 'e')
    finally:
        if ser and ser.is_open:
            ser.close()
            log_message("Serial port closed", 'd')

