#!/bin/bash python3
from uart_utils import *

if __name__ == "__main__":
    LOG_FILE='/data/uart_writer.log'
    log_message("UART (dummy) writer started", log_file=LOG_FILE)

    uart_writer()
