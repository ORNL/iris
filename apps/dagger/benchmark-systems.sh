#!/bin/bash
WD=`pwd`
#reset the experiment (including regenerate the dagger payloads)
rm -fr dagger-figures dagger-payloads dagger-results dagger-graphs

#echo "Running on explorer..."
ssh -l 9bj explorer "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 0 ] && echo "FAILED on explorer!" &&  exit 1

echo "Running on radeon..."
ssh -l 9bj radeon "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 0 ] && echo "FAILED on radeon!" && exit 1

echo "Running on equinox..."
ssh -l 9bj equinox "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 1 ] && echo "FAILED on equinox!" && exit 1

echo "Running on leconte..."
ssh -l 9bj leconte "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 0 ] && echo "FAILED on leconte!" && exit 1

echo "Running on oswald00..."
ssh -l 9bj oswald00 "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 0 ] && echo "FAILED on oswald00!" && exit 1

echo "Running on zenith..."
ssh -l 9bj zenith "cd $WD && ./run-policy-evaluation.sh"
[ $? -ne 0 ] && echo "FAILED on zenith!" && exit 1

