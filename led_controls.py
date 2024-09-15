#!/usr/bin/env python3
import time
from rpi_ws281x import *

# LED strip configuration:
LED_COUNT      = 300      # Number of LED pixels.
LED_PIN        = 18       # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ    = 800000   # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10       # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255      # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0        # Set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create an instance of the LED strip
strip = Adafruit_NeoPixel(
    LED_COUNT, LED_PIN, LED_FREQ_HZ,
    LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
)
strip.begin()
time.sleep(1)  # Allow time for initialization

def turn_off_leds():
    """Turn off all LEDs."""
    for x in range(LED_COUNT):
        strip.setPixelColor(x, Color(0, 0, 0))  # Turn off all LEDs
    strip.show()

def set_led_color(r, g, b):
    """Set all LEDs to a specific color."""
    for x in range(LED_COUNT):
        strip.setPixelColor(x, Color(r, g, b))
    strip.show()


