#!/usr/bin/env python

import iris

iris.init(True)

nplatforms = iris.platform_count()
for i in range(nplatforms):
  name = iris.platform_info(i, iris.iris_name)
  print "platform[", i, "] name[", name, "]"

ndevs = iris.device_count()
for i in range(ndevs):
  vendor = iris.device_info(i, iris.iris_vendor)
  name = iris.device_info(i, iris.iris_name)
  print "device[", i, "] vendor[", vendor, "] name[", name, "]"

iris.finalize()

