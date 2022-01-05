__kernel void setid(__global int* mem) {
  int id = get_global_id(0);
  mem[id] = id;
}

