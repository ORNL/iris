unsigned long iris_create_new_uid() {
  static unsigned long uid = 1UL;
  unsigned long new_uid;
  do {
    new_uid = uid + 1;
  } while (!__sync_bool_compare_and_swap(&uid, uid, new_uid));
  return new_uid;
}

