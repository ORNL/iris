; ModuleID = 'HChain16'
source_filename = "HChain16"

%Qubit  = type opaque
%Result = type opaque

define void @main() #0 {
entry:
  ; ---------------------------------------------------------
  ; Set constants: t, omega, J
  ; ---------------------------------------------------------
  %t_const     = fadd double 0.000000e+00, 1.000000e+00   ; t = 1.0
  %omega_const = fadd double 0.000000e+00, 7.000000e-01   ; omega = 0.7
  %J_const     = fadd double 0.000000e+00, 4.000000e-01   ; J = 0.4

  ; angles
  %theta = fmul double %omega_const, %t_const
  %phi   = fmul double %J_const,     %t_const

  ; ---------------------------------------------------------
  ; Single-qubit Rz(theta) on qubits 0..15
  ; ---------------------------------------------------------
  call void @__quantum__qis__rz__body(double %theta, %Qubit* null)                        ; q0
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 1  to %Qubit*)) ; q1
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 2  to %Qubit*)) ; q2
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 3  to %Qubit*)) ; q3
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 4  to %Qubit*)) ; q4
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 5  to %Qubit*)) ; q5
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 6  to %Qubit*)) ; q6
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 7  to %Qubit*)) ; q7
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 8  to %Qubit*)) ; q8
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 9  to %Qubit*)) ; q9
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 10 to %Qubit*)) ; q10
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 11 to %Qubit*)) ; q11
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 12 to %Qubit*)) ; q12
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 13 to %Qubit*)) ; q13
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 14 to %Qubit*)) ; q14
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 15 to %Qubit*)) ; q15

  ; ---------------------------------------------------------
  ; ZZ coupling blocks via CNOT - Rz(phi) - CNOT
  ; Neighbor pairs: (0,1), (1,2), ..., (14,15)
  ; ---------------------------------------------------------

  ; Pair (0,1)
  call void @__quantum__qis__cnot__body(%Qubit* null,
                                        %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* null,
                                        %Qubit* inttoptr (i64 1 to %Qubit*))

  ; Pair (1,2)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 1 to %Qubit*),
                                        %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 1 to %Qubit*),
                                        %Qubit* inttoptr (i64 2 to %Qubit*))

  ; Pair (2,3)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 2 to %Qubit*),
                                        %Qubit* inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 2 to %Qubit*),
                                        %Qubit* inttoptr (i64 3 to %Qubit*))

  ; Pair (3,4)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 3 to %Qubit*),
                                        %Qubit* inttoptr (i64 4 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 4 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 3 to %Qubit*),
                                        %Qubit* inttoptr (i64 4 to %Qubit*))

  ; Pair (4,5)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 4 to %Qubit*),
                                        %Qubit* inttoptr (i64 5 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 5 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 4 to %Qubit*),
                                        %Qubit* inttoptr (i64 5 to %Qubit*))

  ; Pair (5,6)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 5 to %Qubit*),
                                        %Qubit* inttoptr (i64 6 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 6 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 5 to %Qubit*),
                                        %Qubit* inttoptr (i64 6 to %Qubit*))

  ; Pair (6,7)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 6 to %Qubit*),
                                        %Qubit* inttoptr (i64 7 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 7 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 6 to %Qubit*),
                                        %Qubit* inttoptr (i64 7 to %Qubit*))

  ; Pair (7,8)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 7 to %Qubit*),
                                        %Qubit* inttoptr (i64 8 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 8 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 7 to %Qubit*),
                                        %Qubit* inttoptr (i64 8 to %Qubit*))

  ; Pair (8,9)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 8 to %Qubit*),
                                        %Qubit* inttoptr (i64 9 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 9 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 8 to %Qubit*),
                                        %Qubit* inttoptr (i64 9 to %Qubit*))

  ; Pair (9,10)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 9 to %Qubit*),
                                        %Qubit* inttoptr (i64 10 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 10 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 9 to %Qubit*),
                                        %Qubit* inttoptr (i64 10 to %Qubit*))

  ; Pair (10,11)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 10 to %Qubit*),
                                        %Qubit* inttoptr (i64 11 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 11 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 10 to %Qubit*),
                                        %Qubit* inttoptr (i64 11 to %Qubit*))

  ; Pair (11,12)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 11 to %Qubit*),
                                        %Qubit* inttoptr (i64 12 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 12 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 11 to %Qubit*),
                                        %Qubit* inttoptr (i64 12 to %Qubit*))

  ; Pair (12,13)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 12 to %Qubit*),
                                        %Qubit* inttoptr (i64 13 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 13 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 12 to %Qubit*),
                                        %Qubit* inttoptr (i64 13 to %Qubit*))

  ; Pair (13,14)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 13 to %Qubit*),
                                        %Qubit* inttoptr (i64 14 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 14 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 13 to %Qubit*),
                                        %Qubit* inttoptr (i64 14 to %Qubit*))

  ; Pair (14,15)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 14 to %Qubit*),
                                        %Qubit* inttoptr (i64 15 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 15 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 14 to %Qubit*),
                                        %Qubit* inttoptr (i64 15 to %Qubit*))

  ; ---------------------------------------------------------
  ; Measure all 16 qubits in Z basis
  ; Result IDs 0..15
  ; ---------------------------------------------------------
  call void @__quantum__qis__mz__body(%Qubit* null,
                                      %Result* null)
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1  to %Qubit*),
                                      %Result* inttoptr (i64 1  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 2  to %Qubit*),
                                      %Result* inttoptr (i64 2  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 3  to %Qubit*),
                                      %Result* inttoptr (i64 3  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 4  to %Qubit*),
                                      %Result* inttoptr (i64 4  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 5  to %Qubit*),
                                      %Result* inttoptr (i64 5  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 6  to %Qubit*),
                                      %Result* inttoptr (i64 6  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 7  to %Qubit*),
                                      %Result* inttoptr (i64 7  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 8  to %Qubit*),
                                      %Result* inttoptr (i64 8  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 9  to %Qubit*),
                                      %Result* inttoptr (i64 9  to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 10 to %Qubit*),
                                      %Result* inttoptr (i64 10 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 11 to %Qubit*),
                                      %Result* inttoptr (i64 11 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 12 to %Qubit*),
                                      %Result* inttoptr (i64 12 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 13 to %Qubit*),
                                      %Result* inttoptr (i64 13 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 14 to %Qubit*),
                                      %Result* inttoptr (i64 14 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 15 to %Qubit*),
                                      %Result* inttoptr (i64 15 to %Result*))

  ; ---------------------------------------------------------
  ; Record outputs (16 results)
  ; ---------------------------------------------------------
  call void @__quantum__rt__array_record_output(i64 16, i8* null)
  call void @__quantum__rt__result_record_output(%Result* null,                        i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 2  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 3  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 4  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 5  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 6  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 7  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 8  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 9  to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 10 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 11 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 12 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 13 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 14 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 15 to %Result*), i8* null)

  ret void
}

; --- QIS / runtime declarations (Bell style) ---

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

declare void @__quantum__qis__rz__body(double, %Qubit*)

declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #1

declare void @__quantum__rt__array_record_output(i64, i8*)

declare void @__quantum__rt__result_record_output(%Result*, i8*)

attributes #0 = { "entry_point" "num_required_qubits"="16" "num_required_results"="16" "output_labeling_schema" "qir_profiles"="custom" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}

