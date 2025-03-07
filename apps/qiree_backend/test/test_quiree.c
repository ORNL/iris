#include <stdio.h>
#include <dlfcn.h> // For dlopen, dlsym, dlclose

int main() {
    // Path to the shared library
    const char *lib_path = "libqir.xacc.lib.so";

    // Open the shared library
    void *handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to open library: %s\n", dlerror());
        return 1;
    }

    // Clear any existing errors
    dlerror();

    // Get a pointer to the function
    void (*parse_input_c)(int, char **);
    *(void **)(&parse_input_c) = dlsym(handle, "parse_input_c");

    // Check for errors
    const char *error = dlerror();
    if (error != NULL) {
        fprintf(stderr, "Failed to find symbol: %s\n", error);
        dlclose(handle);
        return 1;
    }
    char* argv[4]; 
    argv[0] = "null";
    argv[1] = "bell.ll";
    argv[2] = "-a";
    argv[3] = "qpp";
    // Call the function with an argument
    parse_input_c(4, (char**)argv);

    // Close the library
    dlclose(handle);

    return 0;
}
