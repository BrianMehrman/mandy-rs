__kernel void mandy(__global double const * real0,
                    __global double const * img0,
                    int max,
                    int width,
                    write_only image2d_t dst_image) {
  unsigned int i = get_global_id(0);
  double real = real0[i];
  double img = img0[i];
  int count = 0;
  double old_r = 0;
  while ((count < max) && (real * real + img * img <= 4.0))
  {
    count++;
    old_r = real;
    real = real * real - img * img + real0[i];
    img = 2.0 * old_r * img + img0[i];
  }

  const int2 coord = (int2)(i % width, i / width);
  const float gray = (float) count / (float) max;
  write_imagef(dst_image, coord, (float4)(gray, gray, gray, 1.0));
}
// vim:ft=c
