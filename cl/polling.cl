__kernel void simple_polling( __global float* inImage, __global float* outImage, const int w, const int h, const int pw, const int ph, const int stride) {
    int x = get_global_id(0);
    float max_value = 0;
    int start = (x % pw) * stride + (x / pw) * w * ph;
    for (int i = 0; i < ph; i++) {
        for (int j = 0; j < pw; j++) {
            int index = start + i * w + j;
            if (index < w * h && inImage[index] > max_value) {
                max_value = inImage[index];
            }
        }
    }
    outImage[x] = max_value;

}

#define TILE_SIZE 32
__kernel void square_polling(__global int* inImage, __global int* outImage, const int w, const int h) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    __local int tile[TILE_SIZE * 2][TILE_SIZE + 1];
    int pointer = group_y * TILE_SIZE * w * 2 + 2 * TILE_SIZE * group_x + w * ly * 2 + 2 * lx;
    for (int i = 0; i < 2; i++) {
        if (pointer + i < w * h) {
           tile[2 * ly + i][lx] = inImage[pointer] > inImage[pointer + 1] ? inImage[pointer] : inImage[pointer + 1];
           pointer += w;
        }
      
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int lj = lx;
    int li = 2 * ly;
    pointer = gy * (w / 2) + gx;
    if (pointer < h * w) {
        outImage[pointer] = tile[li][lj] > tile[li + 1][lj] ? tile[li][lj] : tile[li + 1][lj];
    }
    /*if (pointer < h* w) {
        outImage[pointer] = group_y * TILE_SIZE * w * 2 + 2 * TILE_SIZE * group_x + w * ly * 2;
    }*/
    

}
//#define TILE_SIZE 2
//__kernel void square_polling(__global int2* inImage, __global int* outImage, const int w, const int h) {
//    int gx = get_global_id(0);
//    int gy = get_global_id(1);
//    int lx = get_local_id(0);
//    int ly = get_local_id(1);
//    __local int tile[TILE_SIZE * 2][TILE_SIZE + 1];
//    int p = gy * (w / 2) + gx;
//    int start_index = p * w / 2;
//    for (int i = 0; i < TILE_SIZE; i++) {
//        if (start_index + i < (w * h) / 2) {
//            tile[ly * TILE_SIZE + lx][i] = inImage[start_index + i].x > inImage[start_index + i].y ? inImage[start_index + i].x : inImage[start_index + i].y;
//        }
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    int li = (h / 2) * ly;
//    int lj = lx;
//    if (p < w* h / 4) {
//        outImage[p] = tile[li][lj] > tile[li + 1][lj] ? tile[li][lj] : tile[li + 1][lj];
//    }
//    /* if (p < w * h / 4) {
//         outImage[p] = start_index;
//     }*/
//
//}