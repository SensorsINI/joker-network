#!/bin/bash

source /home/tobi/miniconda3/etc/profile.d/conda.sh
conda activate joker-network
cd /home/tobi/Dropbox/GitHub/SensorsINI/joker-network
terminator -T "producer" -e "python -m producer" &
terminator -T "consumer" -e "python -m consumer" &