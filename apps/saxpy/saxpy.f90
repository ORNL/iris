PROGRAM SAXPY
    USE IRIS
    IMPLICIT NONE

    INTEGER :: I, IERROR
    INTEGER(8) :: SIZE

    REAL(4),DIMENSION(:),ALLOCATABLE :: Z
    REAL(4),DIMENSION(:),ALLOCATABLE :: X
    REAL(4),DIMENSION(:),ALLOCATABLE :: Y
    REAL(4) :: A

    INTEGER(8) :: MEM_Z
    INTEGER(8) :: MEM_X
    INTEGER(8) :: MEM_Y

    INTEGER(8),DIMENSION(3) :: OFF
    INTEGER(8),DIMENSION(3) :: GWS
    INTEGER(8),DIMENSION(3) :: LWS
    INTEGER :: NPARAMS
    INTEGER(8),DIMENSION(4) :: PARAMS
    INTEGER,DIMENSION(4) :: PARAMS_INFO

    INTEGER(8) :: TASK

    CALL IRIS_INIT(.TRUE., IERROR)
    IF (IERROR /= IRIS_SUCCESS) THEN
        PRINT*, 'FAILED AT INIT'
    ENDIF

    SIZE = 8

    ALLOCATE(Z(SIZE))
    ALLOCATE(X(SIZE))
    ALLOCATE(Y(SIZE))

    A = 10.0

    DO I = 1, SIZE
        X(I) = I
        Y(I) = I
    ENDDO

    DO I = 1, SIZE
        PRINT*, 'X[', I, '] ', X(I)
    ENDDO

    PRINT*, '==='

    DO I = 1, SIZE
        PRINT*, 'Y[', I, '] ', Y(I)
    ENDDO

    CALL IRIS_MEM_CREATE(4 * SIZE, MEM_X, IERROR)
    CALL IRIS_MEM_CREATE(4 * SIZE, MEM_Y, IERROR)
    CALL IRIS_MEM_CREATE(4 * SIZE, MEM_Z, IERROR)

    CALL IRIS_TASK_CREATE(TASK, IERROR)

    OFF(1) = 0
    GWS(1) = SIZE
    LWS(1) = SIZE
    NPARAMS = 4
    PARAMS = (/ MEM_Z, TRANSFER(A, TASK), MEM_X, MEM_Y /)
    PARAMS_INFO = (/ IRIS_RW, 4, IRIS_R, IRIS_R /)

    CALL IRIS_TASK_H2D_FULL(TASK, MEM_X, X, IERROR)
    CALL IRIS_TASK_H2D_FULL(TASK, MEM_Y, Y, IERROR)
    CALL IRIS_TASK_KERNEL(TASK, "saxpy", 1, OFF, GWS, LWS, &
      NPARAMS, PARAMS, PARAMS_INFO, IERROR)
    CALL IRIS_TASK_D2H_FULL(TASK, MEM_Z, Z, IERROR)
    CALL IRIS_TASK_SUBMIT(TASK, IRIS_GPU, .TRUE., IERROR)

    DO I = 1, SIZE
        PRINT*, 'Z[', I, '] ', Z(I)
    ENDDO

    DEALLOCATE(X)
    DEALLOCATE(Y)
    DEALLOCATE(Z)

    CALL IRIS_FINALIZE(IERROR)

END PROGRAM SAXPY
