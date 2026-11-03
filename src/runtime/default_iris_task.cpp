/*
 * Author : Narasinga Rao Miniskar
 * Date   : Jun 16 2022
 * Contact for bug : miniskarnr@ornl.gov
 *
 */ 
#include "iris/iris.h"
#include "iris/iris_macros.h"

#define IRIS_API_DEFINITION
#ifndef UNDEF_IRIS_MACROS
#include <stdint.h>
#include "iris/Tiling1D.h"
#include "iris/Tiling2D.h"
#include "iris/zip.hpp"
using namespace iris;
#include "iris_default_math_task_template.h"
#include "iris_default_math_tiled.h"
#endif
