; ModuleID = 'HChain4_3'
source_filename = "HChain4_3"

%Qubit  = type opaque
%Result = type opaque

define void @main() #0 {
entry:
  %t_const     = fadd double 0.000000e+00, 1.000000e+00
  %omega_const = fadd double 0.000000e+00, 7.000000e-01
  %J_const     = fadd double 0.000000e+00, 4.000000e-01

  %theta = fmul double %omega_const, %t_const
  %phi   = fmul double %J_const,     %t_const

  call void @__quantum__qis__rz__body(double %theta, %Qubit* null)
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__rz__body(double %theta, %Qubit* inttoptr (i64 3 to %Qubit*))

  ; (0,1)
  call void @__quantum__qis__cnot__body(%Qubit* null,
                                        %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 1 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* null,
                                        %Qubit* inttoptr (i64 1 to %Qubit*))

  ; (1,2)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 1 to %Qubit*),
                                        %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 2 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 1 to %Qubit*),
                                        %Qubit* inttoptr (i64 2 to %Qubit*))

  ; (2,3)
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 2 to %Qubit*),
                                        %Qubit* inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__rz__body(double %phi,
                                      %Qubit* inttoptr (i64 3 to %Qubit*))
  call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 2 to %Qubit*),
                                        %Qubit* inttoptr (i64 3 to %Qubit*))

  call void @__quantum__qis__mz__body(%Qubit* null,
                                      %Result* null)
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*),
                                      %Result* inttoptr (i64 1 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 2 to %Qubit*),
                                      %Result* inttoptr (i64 2 to %Result*))
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 3 to %Qubit*),
                                      %Result* inttoptr (i64 3 to %Result*))

  call void @__quantum__rt__array_record_output(i64 4, i8* null)
  call void @__quantum__rt__result_record_output(%Result* null,                       i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 2 to %Result*), i8* null)
  call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 3 to %Result*), i8* null)

  ret void
}

declare void @__quantum__qis__h__body(%Qubit*)
declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)
declare void @__quantum__qis__rz__body(double, %Qubit*)
declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #1
declare void @__quantum__rt__array_record_output(i64, i8*)
declare void @__quantum__rt__result_record_output(%Result*, i8*)

attributes #0 = { "entry_point" "num_required_qubits"="4" "num_required_results"="4" "output_labeling_schema" "qir_profiles"="custom" }
attributes #1 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}
!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}

