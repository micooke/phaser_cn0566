### Daemonise the UART listener
Running the Script as a Background Service (Daemon)

To run this reliably on boot and in the background, you'll typically use systemd.

Create a systemd Service File:

Create a file named uart_listener.service in /etc/systemd/system/:

```Bash
sudo nano /etc/systemd/system/uart_listener.service
```
Paste the following content (adjust User and ExecStart path as needed):

```ini
[Unit]
Description=Raspberry Pi UART Listener
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/uart_listener.py
WorkingDirectory=/home/pi/
StandardOutput=inherit  # Inherit for daemon.DaemonContext to redirect
StandardError=inherit   # Inherit for daemon.DaemonContext to redirect
Restart=always
User=pi  # Or the user you want the script to run as (e.g., root)

[Install]
WantedBy=multi-user.target
Systemd Commands:
```

Reload systemd daemon:
```Bash
sudo systemctl daemon-reload
```

Enable the service (starts on boot):
```Bash
sudo systemctl enable uart_listener.service
```

Start the service now:
```Bash
sudo systemctl start uart_listener.service
```

Check the service status:
```Bash
sudo systemctl status uart_listener.service
```

View logs (important for debugging):
```Bash
journalctl -u uart_listener.service -f
``` 
Use Ctrl+C to exit journalctl -f)
You can also check the PYTHON_LOG_FILE (```/var/log/uart_listener.log```) directly.

Stop the service:
```Bash
sudo systemctl stop uart_listener.service
```

Disable the service (stops on boot):
```Bash
sudo systemctl disable uart_listener.service
```

Important Considerations for Raspberry Pi UART:

Disable Serial Console: By default, the Raspberry Pi might use the UART for a serial console. You'll need to disable this to use the UART for your own purposes.

Run
```Bash
sudo raspi-config
```

Select "3 Interface Options" -> "P6 Serial Port"

Answer "No" to "Would you like a login shell to be accessible over serial?"

Answer "Yes" to "Would you like the serial port hardware to be enabled?"

Reboot your Pi.

UART Device Names:

On Raspberry Pi 3/4, ```/dev/ttyS0``` is often the mini UART (which can be affected by CPU frequency).

```/dev/ttyAMA0``` is the primary UART.

```/dev/serial0``` is a symlink that generally points to the "correct" UART you should use if you've disabled the console as above. It's often safer to use ```/dev/serial0```.

Permissions: Ensure the user running the script (e.g., pi) is part of the dialout group to access serial ports:
```Bash
sudo usermod -a -G dialout $USER
```
You'll need to log out and log back in for this to take effect

### 1 Mbps
Setting up a Raspberry Pi 4's PL011 UART and a Jetson Orin's UART for 1 Mbps (1,000,000 baud) involves specific configurations to ensure stability. While 1 Mbps is often achievable on both, it's crucial to understand the nuances.

Raspberry Pi 4B: Enabling PL011 and Setting 1 Mbps
The Raspberry Pi 4B has multiple PL011 UARTs, but the one usually referred to as the "full UART" or "primary UART" (which is ttyAMA0) is shared with Bluetooth. To use it reliably at higher speeds, you need to disable Bluetooth.

1. Disable Serial Console and Enable Hardware UART:

First, ensure the serial console is disabled and the hardware UART is enabled.

Run sudo raspi-config

Select "3 Interface Options"

Select "P6 Serial Port"

Answer "No" to "Would you like a login shell to be accessible over serial?"

Answer "Yes" to "Would you like the serial port hardware to be enabled?"

Exit raspi-config and reboot.

2. Disable Bluetooth and Map PL011 to ttyAMA0 (or ttyS0):

By default, on the Pi 3B+ and 4B, the primary PL011 UART (UART0) is used by Bluetooth, and the mini UART (ttyS0) is assigned to the GPIO pins. To use the PL011, you need to swap them.

Edit ```/boot/config.txt```:
```Bash
sudo nano /boot/config.txt
```

Add the following lines to the end of the file:
```ini
# Disable Bluetooth (frees up PL011 UART0)
dtoverlay=disable-bt

# Remap PL011 UART0 to /dev/ttyS0 (GPIO 14, 15).
# This makes ttyS0 the PL011 UART and ttyAMA0 the Mini UART.
# If you prefer ttyAMA0 to be PL011, you'd use a different overlay.
# However, for simplicity and common usage, remapping ttyS0 to PL011 is fine.
# The default /dev/serial0 symlink will then point to /dev/ttyS0.
dtoverlay=miniuart-bt
```
Explanation:

```dtoverlay=disable-bt```: This overlay disables the Bluetooth module, releasing the ```PL011``` UART.

```dtoverlay=miniuart-bt```: This overlay swaps the UARTs, so the ```PL011 ```(which was originally UART0 and used by Bluetooth) is now assigned to ```/dev/ttyS0``` on ```GPIOs 14 and 15```, and the mini UART (```UART1```) is effectively moved to the Bluetooth side.

Alternatively, if you want ```ttyAMA0``` to be the ```PL011``` and ```ttyS0``` to be the mini UART (the original naming scheme before the swap), you might need enable_uart=1 and ensure no ```dtoverlay=pi3-miniuart-overlay``` or similar is active, but ```miniuart-bt``` is the most common way to get the full UART on the standard GPIO pins for external use.

Save and exit (Ctrl+X, Y, Enter).

Reboot: sudo reboot

3. Verify UART Assignment:

After rebooting, check which UART is assigned to ```/dev/ttyS0``` (or ```/dev/ttyAMA0``` if you chose that path):

```Bash
ls -l /dev/serial0
```
# This symlink should point to the correct active UART

You might see it points to ```/dev/ttyS0``` (meaning ```PL011``` is now on ```ttyS0```).

4. Set Baud Rate in Python:

Now, in your Python script, you can use pyserial to open the port at 1 Mbps.

```Python
import serial
import time

UART_PORT = '/dev/serial0'  # Or '/dev/ttyS0' if you confirmed that's your PL011
BAUD_RATE = 1000000         # 1 Mbps

try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"Opened serial port {UART_PORT} at {BAUD_RATE} baud.")

    # Example: Send and receive data
    while True:
        # Send data
        message = "Hello from RPi at 1Mbps!\n"
        ser.write(message.encode('utf-8'))
        print(f"Sent: {message.strip()}")

        # Read data
        if ser.in_waiting > 0:
            received_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"Received: {received_data.strip()}")

        time.sleep(1)

except serial.SerialException as e:
    print(f"Error opening or communicating with serial port: {e}")
except KeyboardInterrupt:
    print("Script terminated by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
```
Important Notes for RPi 1 Mbps:

Signal Integrity: At 1 Mbps, signal integrity becomes crucial. Use short, shielded cables.

Flow Control: For sustained high-speed communication, consider enabling hardware flow control (RTS/CTS) if your devices support it. This helps prevent buffer overflows. You can often add rtscts=True to the serial.Serial constructor, but it also requires proper wiring and potential dtoverlay parameters if not enabled by default.

CPU Load: While the PL011 is hardware-accelerated, heavy CPU load on the Pi could still indirectly affect serial performance if interrupt latency increases significantly.

Jetson Orin: Setting 1 Mbps UART
The Jetson Orin modules typically have multiple high-speed UARTs. For 1 Mbps, the standard Linux serial drivers usually handle it without deep device tree modifications, but there are a few things to check.

1. Identify the UART Port:

Jetson Orin modules have several UARTs. Common ones you might use from the 40-pin header are ttyTHS1 or ttyTHS0 (or sometimes /dev/ttyS0 which is a symlink to one of these).

On the Orin Nano Dev Kit, UART2_TX (Pin 8) and UART2_RX (Pin 10) on the 40-pin header typically map to ```/dev/ttyTHS1```.

Always consult the official NVIDIA Jetson Orin Developer Kit documentation or pinout diagrams for your specific Orin module to confirm the exact UART mapping to GPIO pins and /dev/tty... names.

2. Disable Serial Console (if applicable):

If the UART you intend to use is currently being used as a debug console, you'll need to disable it. For example, to disable the console on ttyTHS0 (which is often the default debug UART):

```Bash
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
sudo udevadm trigger # Reload udev rules
```

3. Set Baud Rate in Python:

The Linux kernel and pyserial on Jetson Orin should be able to set 1 Mbps directly.

```Python
import serial
import time

UART_PORT = '/dev/ttyTHS1'  # Common UART on Jetson Orin 40-pin header
                          # Adjust based on your specific module/pinout (e.g., /dev/ttyTHS0)
BAUD_RATE = 1000000         # 1 Mbps

try:
    ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
    print(f"Opened serial port {UART_PORT} at {BAUD_RATE} baud.")

    # Example: Send and receive data
    while True:
        # Send data
        message = "Hello from Jetson Orin at 1Mbps!\n"
        ser.write(message.encode('utf-8'))
        print(f"Sent: {message.strip()}")

        # Read data
        if ser.in_waiting > 0:
            received_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"Received: {received_data.strip()}")

        time.sleep(1)

except serial.SerialException as e:
    print(f"Error opening or communicating with serial port: {e}")
except KeyboardInterrupt:
    print("Script terminated by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
```

Important Notes for Jetson Orin 1 Mbps:

Permissions: Ensure your user is in the dialout group: sudo usermod -a -G dialout $USER. Log out and back in.

Hardware Flow Control (RTS/CTS): For reliable 1 Mbps communication, especially bidirectional, it is highly recommended to use hardware flow control. Connect RTS/CTS lines and enable them in your pyserial configuration: ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1, rtscts=True).

Device Tree Overlays (for higher speeds or specific UARTs): While 1 Mbps usually works out of the box, if you push to much higher speeds (e.g., 4 Mbps or 12.5 Mbps) or need to enable/configure a specific UART that isn't readily available, you might need to modify the device tree. This is more advanced and involves recompiling and flashing the device tree, so start with the standard setup.

Logic Levels: Jetson Orin's GPIOs (and thus its UART pins) are typically 1.8V logic. Ensure your external device is compatible or use a logic level shifter if it operates at 3.3V or 5V. This is crucial for preventing damage.

General Best Practices for High-Speed UART:

Wiring: Use good quality, short, shielded wires. Avoid long jumper wires.

Grounding: Ensure a common ground connection between your Raspberry Pi/Jetson and the external device.

Error Checking: Implement checksums or CRC in your communication protocol to detect data corruption, especially at high speeds.

Buffering: Be aware of internal UART FIFOs and how quickly your software can read/write data to avoid overflows. Hardware flow control is the best defense against this.

Testing: Thoroughly test the communication at your desired baud rate under various conditions (e.g., continuous data, bursts, different message lengths) to confirm stability.

By following these steps, you should be able to set up 1 Mbps UART communication on both your Raspberry Pi 4B and Jetson Orin.

## Two UARTS on Jetson Orin

On many Jetson boards, you'll find:

UART1 (often /dev/ttyTHS1): Typically on pins 8 (TX) and 10 (RX) of the 40-pin header. It often also has RTS/CTS on pins 11/36. This is a very common user-accessible UART.

UART0 (often /dev/ttyTHS0): This is frequently the debug UART. If you want to use it for your application, you'll need to disable the serial console on it.

Other UARTs might be available on different headers or internal to the module, requiring a custom carrier board.

Disable Serial Console (if necessary):
If one of the UARTs you want to use (e.g., ttyTHS0) is configured as the system's debug console, you'll need to disable it to prevent conflicts and allow your application to use it.


```Bash
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
sudo udevadm trigger # Reload udev rules
```
This stops the getty service (which provides the login prompt) on the debug UART.

Verify User Permissions:
Ensure the user running your Python script (or any other application accessing the UARTs) is part of the dialout group. This group has permissions to access serial devices.

```Bash
sudo usermod -a -G dialout $USER
```

(Replace $USER with your actual username, e.g., nvidia or ubuntu).
You will need to log out and log back in for group changes to take effect.

Python Code for Two UARTs:
Once you've identified the /dev/ttyTHSx paths for your two UARTs, using them in Python with pyserial is straightforward.

```Python
import serial
import time

# --- Configuration for UART 1 ---
UART_PORT_1 = '/dev/ttyTHS1' # Example: Common user UART on 40-pin header
BAUD_RATE_1 = 115200         # Example baud rate
# Enable hardware flow control (RTS/CTS) for reliability at higher speeds
# Ensure these pins are wired if you enable flow control.
USE_FLOW_CONTROL_1 = True

# --- Configuration for UART 2 ---
UART_PORT_2 = '/dev/ttyTHS0' # Example: Often the debug UART, ensure disabled as console
BAUD_RATE_2 = 9600           # Example baud rate (can be different)
USE_FLOW_CONTROL_2 = False

# --- Open Serial Ports ---
ser1 = None
ser2 = None

try:
    # Open UART 1
    ser1 = serial.Serial(
        UART_PORT_1,
        BAUD_RATE_1,
        timeout=1,
        rtscts=USE_FLOW_CONTROL_1 # Enable RTS/CTS if needed
    )
    print(f"Opened UART 1: {UART_PORT_1} at {BAUD_RATE_1} baud (RTS/CTS: {USE_FLOW_CONTROL_1})")

    # Open UART 2
    ser2 = serial.Serial(
        UART_PORT_2,
        BAUD_RATE_2,
        timeout=1,
        rtscts=USE_FLOW_CONTROL_2 # Enable RTS/CTS if needed
    )
    print(f"Opened UART 2: {UART_PORT_2} at {BAUD_RATE_2} baud (RTS/CTS: {USE_FLOW_CONTROL_2})")

    # --- Main Loop to Read and Write ---
    counter = 0
    while True:
        # --- UART 1 Operations ---
        message1 = f"Hello from UART 1: {counter}\n"
        ser1.write(message1.encode('utf-8'))
        print(f"UART 1 Sent: {message1.strip()}")

        if ser1.in_waiting > 0:
            received_data1 = ser1.read(ser1.in_waiting).decode('utf-8', errors='ignore')
            print(f"UART 1 Received: {received_data1.strip()}")

        # --- UART 2 Operations ---
        message2 = f"Hello from UART 2: {counter * 10}\n"
        ser2.write(message2.encode('utf-8'))
        print(f"UART 2 Sent: {message2.strip()}")

        if ser2.in_waiting > 0:
            received_data2 = ser2.read(ser2.in_waiting).decode('utf-8', errors='ignore')
            print(f"UART 2 Received: {received_data2.strip()}")

        counter += 1
        time.sleep(1) # Adjust sleep as needed for your application

except serial.SerialException as e:
    print(f"Serial port error: {e}")
except KeyboardInterrupt:
    print("Script terminated by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Close ports cleanly
    if ser1 and ser1.is_open:
        ser1.close()
        print("UART 1 closed.")
    if ser2 and ser2.is_open:
        ser2.close()
        print("UART 2 closed.")
```

UART1 (often /dev/ttyTHS1)
TX (Transmit): Pin 8 (J30, UART1_TX)
RX (Receive): Pin 10 (J30, UART1_RX)
RTS (Request To Send): Pin 11 (J30, UART1_RTS) - Hardware Flow Control
CTS (Clear To Send): Pin 36 (J30, UART1_CTS) - Hardware Flow Control
GND (Ground): Pin 9, 20, 25, 30, 34, 39 (various GND pins on J30)

This is usually the go-to UART for users as it's typically configured for general use and has hardware flow control lines available.

UART0 (often /dev/ttyTHS0 or accessed via the Micro-USB debug port)
TX (Transmit): Pin 13 on the Automation Header (J42), or via the Micro-USB port (J26) on the Developer Kit.
RX (Receive): Pin 14 on the Automation Header (J42), or via the Micro-USB port (J26) on the Developer Kit.

Associated /dev/tty device: This is usually /dev/ttyTHS0 in Jetson Linux, but it's often configured as the system debug console (where you see boot messages and can log in).

Important Note for UART0: If you want to use UART0 for your application, you must disable the serial console that typically runs on it. If you don't, your application will conflict with the system's getty service trying to provide a login prompt.

To verify the pinout for your specific developer kit version, always refer to the official NVIDIA Jetson AGX Orin Developer Kit documentation and carrier board specifications. You can often find detailed pinout diagrams on the NVIDIA developer website or resources like JetsonHacks.

Steps to Enable and Use Two UARTs
Assuming you want to use UART1 (on the 40-pin header) and UART0 (after disabling the console):

Step 1: Disable the Serial Console on UART0 (if using it)
If you plan to use /dev/ttyTHS0 for your application, you need to free it from the system's serial console:

Stop the nvgetty service:

```Bash
sudo systemctl stop nvgetty
```
Disable the nvgetty service from starting on boot:

```Bash
sudo systemctl disable nvgetty
```
Reload udev rules:

```Bash
sudo udevadm trigger
```
This ensures that the device node is correctly set up for user access.

Step 2: Verify User Permissions
Your user account needs to be part of the dialout group to access serial ports.

Add your user to the dialout group:

```Bash
sudo usermod -a -G dialout $USER
```
(Replace $USER with your actual username, e.g., nvidia or ubuntu).

Log out and log back in for the group change to take effect.

Step 3: Python Code Example with pyserial
Now you can write a Python script to communicate over both UARTs.

```Python
import serial
import time
import os

# --- Configuration for UART 1 (on 40-pin header) ---
# Check your specific AGX Orin Developer Kit documentation for exact mapping
UART_PORT_1 = '/dev/ttyTHS1'  # Common for UART1 on 40-pin header
BAUD_RATE_1 = 1000000         # Example: 1 Mbps
USE_FLOW_CONTROL_1 = True     # Recommended for high speeds (RTS/CTS pins 11, 36)

# --- Configuration for UART 0 (Debug UART, after disabling console) ---
# This is typically the debug UART, accessible via Micro-USB or Automation Header
UART_PORT_0 = '/dev/ttyTHS0'  # Common for UART0
BAUD_RATE_0 = 115200          # Example: A more typical baud rate for debug
USE_FLOW_CONTROL_0 = False    # Often not used for the debug UART, but can be enabled if wired

# --- Log file for script activity ---
LOG_FILE = "/var/log/orin_uart_log.log"

def log_message(message):
    """Simple logging function."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)
    print(log_entry.strip()) # Also print to console for immediate feedback

# --- Main UART communication function ---
def main():
    ser0 = None
    ser1 = None
    try:
        log_message("Attempting to open UART ports...")

        # Open UART 0
        try:
            ser0 = serial.Serial(
                UART_PORT_0,
                BAUD_RATE_0,
                timeout=0.1,  # Short timeout for non-blocking read
                rtscts=USE_FLOW_CONTROL_0
            )
            log_message(f"Successfully opened UART 0: {UART_PORT_0} at {BAUD_RATE_0} baud (RTS/CTS: {USE_FLOW_CONTROL_0})")
        except serial.SerialException as e:
            log_message(f"Could not open UART 0 ({UART_PORT_0}): {e}. Make sure console is disabled.")
            # Exit or handle gracefully if UART0 is critical

        # Open UART 1
        try:
            ser1 = serial.Serial(
                UART_PORT_1,
                BAUD_RATE_1,
                timeout=0.1,  # Short timeout for non-blocking read
                rtscts=USE_FLOW_CONTROL_1
            )
            log_message(f"Successfully opened UART 1: {UART_PORT_1} at {BAUD_RATE_1} baud (RTS/CTS: {USE_FLOW_CONTROL_1})")
        except serial.SerialException as e:
            log_message(f"Could not open UART 1 ({UART_PORT_1}): {e}. Check wiring and permissions.")
            # Exit or handle gracefully if UART1 is critical

        if not ser0 and not ser1:
            log_message("No UART ports could be opened. Exiting.")
            return

        counter = 0
        while True:
            # --- Communicate on UART 0 (if open) ---
            if ser0 and ser0.is_open:
                msg0 = f"Hello from UART0 (msg {counter})\n"
                ser0.write(msg0.encode('utf-8'))
                # log_message(f"UART0 Sent: {msg0.strip()}") # Can be verbose

                if ser0.in_waiting > 0:
                    received0 = ser0.read(ser0.in_waiting).decode('utf-8', errors='ignore').strip()
                    if received0:
                        log_message(f"UART0 Received: {received0}")

            # --- Communicate on UART 1 (if open) ---
            if ser1 and ser1.is_open:
                msg1 = f"Hello from UART1 (cmd {counter * 10})\n"
                ser1.write(msg1.encode('utf-8'))
                # log_message(f"UART1 Sent: {msg1.strip()}") # Can be verbose

                if ser1.in_waiting > 0:
                    received1 = ser1.read(ser1.in_waiting).decode('utf-8', errors='ignore').strip()
                    if received1:
                        log_message(f"UART1 Received: {received1}")

            counter += 1
            time.sleep(0.5) # Adjust based on your communication frequency

    except KeyboardInterrupt:
        log_message("Script terminated by user.")
    except Exception as e:
        log_message(f"An unexpected error occurred: {e}")
    finally:
        if ser0 and ser0.is_open:
            ser0.close()
            log_message("UART 0 closed.")
        if ser1 and ser1.is_open:
            ser1.close()
            log_message("UART 1 closed.")

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    main()
```
Step 4: Physical Wiring
For UART1 (Pins 8, 10, 11, 36 on 40-pin header):

Connect your external device's TX to Orin's Pin 10 (UART1_RX).

Connect your external device's RX to Orin's Pin 8 (UART1_TX).

Connect your external device's CTS to Orin's Pin 11 (UART1_RTS) if using flow control.

Connect your external device's RTS to Orin's Pin 36 (UART1_CTS) if using flow control.

Connect common ground (e.g., Orin Pin 9) to your external device's ground.

For UART0 (Micro-USB Debug Port or Automation Header Pins 13, 14):

If using the Micro-USB debug port, you'll need a USB to serial adapter cable (like an FTDI cable) connected to another computer running a serial terminal (e.g., PuTTY, minicom).

If using the Automation Header, you'd wire similarly to UART1 (external TX to Orin RX, external RX to Orin TX), ensuring you know which physical pins correspond to ttyTHS0.

Crucial Logic Level Note: The Jetson AGX Orin's GPIOs (including UART pins) operate at 1.8V logic levels. Most external devices (like Arduino, Raspberry Pi, or other microcontrollers) might use 3.3V or 5V logic. You MUST use a logic level shifter if your external device is not 1.8V compatible, otherwise, you risk damaging your Jetson.

Step 5: Run the Python Script
You can run the script directly:

Bash

python3 your_uart_script.py
For background operation, consider using systemd as outlined in the previous response, redirecting output to a log file for debugging.

System-level UART Configuration (Advanced/Troubleshooting)
Generally, for ttyTHS0 and ttyTHS1 on the AGX Orin Dev Kit, the necessary drivers are already enabled in Jetson Linux, and baud rates are handled by pyserial through standard termios calls.

You typically do not need to manually modify device tree overlays (.dtbo files) for these two UARTs at common baud rates (including 1 Mbps). Device tree modifications are usually only needed if:

You are using a custom carrier board where UARTs are routed differently or are disabled by default.

You need to enable extremely high baud rates (e.g., >4 Mbps) that require specific clock source configurations not exposed via standard Linux APIs.

You want to re-purpose pins that are not, by default, configured as UART.

If you ever suspect a pinmuxing issue or need to confirm configurations, the jetson-io.py tool (located at /opt/nvidia/jetson-io/jetson-io.py) is a helpful graphical utility that allows you to reconfigure the 40-pin header interfaces and generates appropriate device tree overlays for you. However, for just using ttyTHS0 and ttyTHS1 (with ttyTHS0 freed from the console), it's often not necessary.