#!/bin/bash

# This script will be called by the Python UART listener.
LOG_FILE="/var/log/my_uart_commands.log"

echo "$(date): Bash script executed! Arguments: $*" >> "$LOG_FILE"
echo "Hello from the bash script!"

# I might also trigger an LED e.g.
# sudo echo "1" > /sys/class/gpio/gpioXX/value
