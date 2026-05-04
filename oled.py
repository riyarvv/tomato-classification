import smbus2
import time

# I2C address and bus
I2C_ADDR = 0x3C
bus = smbus2.SMBus(1)  # Pi 5 uses I2C bus 1

def send_command(cmd):
    # Control byte 0x00 indicates a command is coming
    bus.write_byte_data(I2C_ADDR, 0x00, cmd)

def send_data(data):
    # Control byte 0x40 indicates data (pixels) is coming
    bus.write_byte_data(I2C_ADDR, 0x40, data)

def init_display():
    print("Initializing SSD1306...")
    # Standard initialization sequence
    commands = [
        0xAE, # Display OFF
        0xD5, 0x80, # Set Display Clock Divide Ratio
        0xA8, 0x3F, # Set Multiplex Ratio (64 lines)
        0xD3, 0x00, # Set Display Offset
        0x40, # Set Display Start Line
        0x8D, 0x14, # Enable Charge Pump (CRITICAL)
        0x20, 0x00, # Set Memory Addressing Mode (Horizontal)
        0xA1, # Set Segment Re-map (flipped)
        0xC8, # Set COM Output Scan Direction (flipped)
        0xDA, 0x12, # Set COM Pins Hardware Configuration
        0x81, 0xCF, # Set Contrast Control
        0xD9, 0xF1, # Set Pre-charge Period
        0xDB, 0x40, # Set VCOMH Deselect Level
        0xA4, # Entire Display ON (Resume from RAM)
        0xA6, # Set Normal Display
        0xAF  # Display ON
    ]
    for cmd in commands:
        send_command(cmd)

def fill_screen():
    print("Filling screen with pattern...")
    # 128x64 pixels = 1024 bytes (8 pixels per byte)
    for _ in range(1024):
        send_data(0xAA) # 0xAA creates a striped pattern (10101010)

try:
    init_display()
    fill_screen()
    print("Test finished. You should see stripes on the OLED.")
    time.sleep(5)
    send_command(0xAE) # Turn display back off
except Exception as e:
    print(f"Failed to talk to OLED: {e}")
finally:
    bus.close()
