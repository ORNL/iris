#ifndef IRIS_INCLUDE_IRIS_HEXAGON_IMP_H
#define IRIS_INCLUDE_IRIS_HEXAGON_IMP_H

#define IRIS_HEXAGON_KERNEL_ARGS     int32 _off, int32 _ndr
#define IRIS_HEXAGON_KERNEL_BEGIN(i) for (i = _off; i < _off + _ndr; i++) {
#define IRIS_HEXAGON_KERNEL_END      }

#endif /* IRIS_INCLUDE_IRIS_HEXAGON_IMP_H */

