#!/bin/bash

while true;
do
  cat /sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input
  sleep 1
done

