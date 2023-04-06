#ifndef IRIS_SRC_RT_UTILS_H
#define IRIS_SRC_RT_UTILS_H

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;
namespace iris {
    namespace rt {

        class Utils {
            public:
                static void Logo(bool color);
                static int ReadFile(char* path, char** string, size_t* len);
                static int Mkdir(char* path);
                static bool Exist(char* path);
                static long Mtime(char* path);
                static void Datetime(char* str);
                static void MemCpy3D(uint8_t *dev, uint8_t *host, size_t *off, size_t *dev_sizes, size_t *host_sizes, size_t elem_size, bool host_2_dev=true);

                template<typename DType>
                static DType *AllocateArray(size_t SIZE, DType init=0)
                {
                    DType *A = (DType *)malloc ( SIZE *  sizeof(DType));
                    for (size_t i = 0; i < SIZE; i++ )
                        A[i] = init;
                    return A;
                }
                template<typename DType>
                static DType *AllocateRandomArray(size_t SIZE)
                {
                    DType *A = (DType *)malloc ( SIZE *  sizeof(DType));
                    for (size_t i = 0; i < SIZE; i++ )
                    {
                        int a = rand() % 5+1;
                        A[i] = (DType)a;
                    }
                    return A;
                }
                template<class DType>
                static void PrintVectorLimited(vector<DType> & A, const char *description, int limit=8)
                    {
                        printf("%s (%d)\n", description, A.size());
                        int print_cols = A.size()>limit?limit:A.size();
                        for (int j = 0; j < print_cols; j++) {
                            cout << setw(6) << A[j] << " ";
                        }
                        cout << endl;
                    }
                template<class DType>
                static void PrintVectorFull(vector<DType> &A, const char *description)
                    {
                        printf("%s (%d)\n", description, A.size());
                        for (int j = 0; j < A.size(); j++) {
                            cout << setw(6) << A[j] << " ";
                        }
                        cout << endl;
                    }

                template<class DType>
                static void PrintArrayLimited(DType *A, int N, const char *description, int limit=8)
                    {
                        printf("%s data:%p (%d)\n", description, A, N);
                        int print_cols = N>limit?limit:N;
                        for (int j = 0; j < print_cols; j++) {
                            cout << setw(6) << A[j] << " ";
                        }
                        cout << endl;
                    }
                template<class DType>
                static void PrintAarrayFull(DType *A, int N, const char *description)
                    {
                        printf("%s data:%p (%d)\n", description, A, N);
                        for (int j = 0; j < N; j++) {
                            cout << setw(6) << A[j] << " ";
                        }
                        cout << endl;
                    }

                template<class DType>
                static void PrintMatrixLimited(DType *A, int M, int N, const char *description, int limit=8)
                    {
                        printf("%s data:%p (%d, %d)\n", description, A, M, N);
                        int print_rows = M>limit?limit:M;
                        int print_cols = N>limit?limit:N;
                        for(int i=0; i<print_rows; i++) {
                            for (int j = 0; j < print_cols; j++) {
                                cout << setw(6) << A[i*N+j] << " ";
                            }
                            cout << endl;
                        }
                    }
                template<class DType>
                static void PrintMatrixFull(DType *A, int M, int N, const char *description)
                    {
                        printf("%s data:%p (%d, %d)\n", description, A, M, N);
                        for(int i=0; i<M; i++) {
                            for (int j = 0; j < N; j++) {
                                cout << setw(6) << A[i*N+j] << " ";
                            }
                            cout << endl;
                        }
                    }

        };

    } /* namespace rt */
} /* namespace iris */

#endif /* IRIS_SRC_RT_UTILS_H */
