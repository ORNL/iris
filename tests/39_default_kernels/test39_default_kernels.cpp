#include <math.h>
#include <stdio.h>
#include "iris/iris.h"
#include "iris/iris_default_math.h"

int main(int argc, char **argv)
{
    iris_enable_default_kernels(true);
    iris_init(&argc, &argv, true);
	float d[16], out[16];
	for(int i=0; i<16; i++) {
		d[i] = 2.0f;
		out[i] = 0.0f;
	}
	iris_cos<float>(iris_default, out, d, 16);
	printf(" Out:");
    int errors = 0;
	for(int i=0; i<16; i++) {
        if ((fabs(out[i] - -0.416147f)) > 1e-6)
            errors++;
		printf(" %f", out[i]);
	}
	printf("\n");
    if (errors > 0)
        fprintf(stderr, "Errors: %d\n", errors);
	iris_finalize();
	return errors+iris_error_count();
}
