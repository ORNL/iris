#!/bin/bash
WATCH_ID=$1

echo Watcher is watching: $WATCH_ID
tail --pid $WATCH_ID -f /dev/null
sleep 60
echo Stopping slurm job: $(cat slurm.job)
[ -s "slurm.job" ] && cat slurm.job | xargs scancel