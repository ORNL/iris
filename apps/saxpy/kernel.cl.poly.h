static int saxpy(BRISBANE_POLY_KERNEL_ARGS) {
  BRISBANE_POLY_ARRAY_1D(saxpy, X, sizeof(float), 0);
  BRISBANE_POLY_ARRAY_1D(saxpy, Y, sizeof(float), 0);
  BRISBANE_POLY_ARRAY_1D(saxpy, Z, sizeof(float), 0);
  {
  BRISBANE_POLY_DOMAIN(i0, 0, _ndr2 - 1);
  BRISBANE_POLY_DOMAIN(i1, 0, _ndr1 - 1);
  BRISBANE_POLY_DOMAIN(i2, 0, _ndr0 - 1);
  BRISBANE_POLY_READ_1D(saxpy, X, _off0+i2[0]);
  BRISBANE_POLY_READ_1D(saxpy, X, _off0+i2[1]);
  }
  {
  BRISBANE_POLY_DOMAIN(i0, 0, _ndr2 - 1);
  BRISBANE_POLY_DOMAIN(i1, 0, _ndr1 - 1);
  BRISBANE_POLY_DOMAIN(i2, 0, _ndr0 - 1);
  BRISBANE_POLY_READ_1D(saxpy, Y, _off0+i2[0]);
  BRISBANE_POLY_READ_1D(saxpy, Y, _off0+i2[1]);
  }
  {
  BRISBANE_POLY_DOMAIN(i0, 0, _ndr2 - 1);
  BRISBANE_POLY_DOMAIN(i1, 0, _ndr1 - 1);
  BRISBANE_POLY_DOMAIN(i2, 0, _ndr0 - 1);
  BRISBANE_POLY_MUWR_1D(saxpy, Z, _off0+i2[0]);
  BRISBANE_POLY_MUWR_1D(saxpy, Z, _off0+i2[1]);
  }
  return 1;
}

static int saxpy_poly_available() { return 1; }

