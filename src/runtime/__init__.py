from .iris import *

import sys
import types

module = sys.modules[__name__]

for attr in dir(module):
    if isinstance(getattr(module, attr), types.FunctionType):
        globals()[attr] = getattr(module, attr)

del sys, types, module
