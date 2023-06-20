#!/bin/bash

errors=0
total=0

# Validate against record schema.
for f in tests/18_record/*.json tests/28_json2/*.json tests/32_json3/*.json tests/35_json_mixed_args_record_replay/*.json tests/36_double_json_mixed_args_record_replay/*.json
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
for f in tests/22_json_mixed_args/*.json
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
