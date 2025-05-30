#!/bin/bash

errors=0
total=0

# Validate against record schema.
echo "Tests directory: tests-${IRIS_ARCHS}-${IRIS_ASYNC}-${IRIS_TAG}-${IRIS_MACHINE}${IRIS_TESTNAME}"
for f in tests-${IRIS_ARCHS}-${IRIS_ASYNC}-${IRIS_TAG}-${IRIS_MACHINE}${IRIS_TESTNAME}/**/output.json
do
   echo python utils/validate_schema.py -i $f -s schema/record.schema.json
   python utils/validate_schema.py -i $f -s schema/record.schema.json
   es=$?
   if (( es > 0 ))
   then
      errors=$((errors+1))
   fi
   total=$((total+1))
done

# Validate against dagger schema.
for f in tests-${IRIS_ARCHS}-${IRIS_ASYNC}-${IRIS_TAG}-${IRIS_MACHINE}${IRIS_TESTNAME}/**/*.json
do
   echo python utils/validate_schema.py -i $f -s schema/dagger.schema.json
   python utils/validate_schema.py -i $f -s schema/dagger.schema.json
   es=$?
   if (( es > 0 ))
   then
      errors=$((errors+1))
   fi
   total=$((total+1))
done

echo Schema check: $((total-errors)) of $total tests passed.
exit $errors
