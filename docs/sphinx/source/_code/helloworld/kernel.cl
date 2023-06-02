__kernel void uppercase(__global char* b, __global char* a) {
  int i = get_global_id(0);
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
}
