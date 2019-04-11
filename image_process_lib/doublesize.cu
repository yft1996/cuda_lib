#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
/**
 * @property    图像缩放
 * @func        将图像放大为原图两倍  均匀插值，最后一行后最后一列与倒数第二行与倒数第二列相同
 * @param_out   out_image          放大后的图像首地址
 * @param_in    in_image           待放大图像首地址
 * @param_in    weight             输入图像的宽度
 * @param_in    height             输入图像的高度
 * @param_in    channels           输入图像的颜色通道数
 */
void double_size_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels);
/*****************************************************************************************************************/
//放大后的像素点由4个放大前的像素点决定，视放大后的像素点位置来决定四个输入(yoff存储输入的y坐标，xoff存储输入的x坐标)
/*****************************************************************************************************************/
__global__ void kernel_doublesize_dim3(float *out,float *in,int const image_x,int const image_y,int const iw)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int out_z=threadIdx.z;

    if(out_x<image_x&&out_y<image_y)
    {
        int idx=out_y*image_x*blockDim.z+out_x*blockDim.z+out_z;

        const bool nexty=(out_y+1)<image_y;
        const bool nextx=(out_x+1)<image_x;
        int yoff[2]={blockDim.z*iw*(out_y>>1),blockDim.z*iw*((out_y+nexty)>>1)};
        int xoff[2]={blockDim.z*(out_x>>1),blockDim.z*((out_x+nextx)>>1)};
        int index[4]={yoff[0]+xoff[0]+out_z,
                      yoff[0]+xoff[1]+out_z,
                      yoff[1]+xoff[0]+out_z,
                      yoff[1]+xoff[1]+out_z};
        out[idx]=0.25f*(in[index[0]]+in[index[1]]+in[index[2]]+in[index[3]]);

        int idx_2=out_y*image_x*blockDim.z+(out_x+blockDim.x)*blockDim.z+out_z;
        const bool nextx_2=(out_x+blockDim.x+1)<image_x;
        int xoff_2[2]={blockDim.z*((out_x+blockDim.x)>>1),blockDim.z*((out_x+blockDim.x+nextx_2)>>1)};
        int index_2[4]={yoff[0]+xoff_2[0]+out_z,
                      yoff[0]+xoff_2[1]+out_z,
                      yoff[1]+xoff_2[0]+out_z,
                      yoff[1]+xoff_2[1]+out_z};
        out[idx_2]=0.25f*(in[index_2[0]]+in[index_2[1]]+in[index_2[2]]+in[index_2[3]]);


    }

}
__global__ void kernel_doublesize(float *out,float *in,int const image_x,int const image_y,int const iw,int const ic)
{
    int out_x = threadIdx.x + blockIdx.x * blockDim.x * ic*3;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int c = 0; c <ic*3 ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<image_y&&fact_x<image_x*ic)
        {
            int idx=fact_x+out_y*image_x*ic;
            bool nexty=(out_y+1)<image_y;
            bool nextx=(fact_x+ic)<(image_x*ic);
            int yoff[2]={ic*iw*(out_y>>1),
                         ic*iw*((out_y+nexty)>>1)};
            int xoff[2]={((fact_x/ic)>>1)*ic+fact_x%ic,
                         (((fact_x/ic)+nextx)>>1)*ic+fact_x%ic};
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            out[idx]=0.25f*(in[index[0]]+in[index[1]]+in[index[2]]+in[index[3]]);
        }
    }
}
__global__ void kernel_doublesizebyshare(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_fact_x=share_x*3*ic;
    int share_idx_x;
    int share_idx_y= threadIdx.y;//共享内存块内y维索引
    int in_x0 = ((blockIdx.x * blockDim.x*3) >> 1) * ic;//对应共享块的输入首地址的x坐标
    int in_y0 = (blockIdx.y * blockDim.y) >> 1;//对应共享块的输入首地址的y坐标
    int x,y,c,fact_x;

    for ( c = 0; c <ic*3 ; ++c)
    {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        if (share_idx_x < share_fact_x && share_idx_y < share_y)
        {
            x = min(in_x0 + share_idx_x, iw * ic - ic + share_idx_x % ic);//实际读取输入的x坐标，防止越界
            y = min(in_y0 + share_idx_y, ih - 1);//实际读取输入的y坐标，防止越界
            data[share_idx_y * share_fact_x + share_idx_x] = in[y * iw * ic + x];
        }

    }
    __syncthreads();
    for ( c = 0; c <ic*3 ; ++c)
    {
        fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            int yoff[2]={(share_idx_y>>1)*share_fact_x,((share_idx_y+1)>>1)*share_fact_x};
            int xoff[2]={(share_idx_x/ic>>1)*ic+share_idx_x%ic,
                         ((share_idx_x/ic+1)>>1)*ic+share_idx_x%ic};
            int out_idx=out_y*ow*ic+fact_x;
            int index[4]={yoff[0]+xoff[0],
                          yoff[0]+xoff[1],
                          yoff[1]+xoff[0],
                          yoff[1]+xoff[1]};
            out[out_idx]=0.25f*(data[index[0]]+data[index[1]]+data[index[2]]+data[index[3]]);
        }
    }
    /*
    extern __shared__ float  data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;

    int share_x=(blockDim.x>>1)+1;//共享内存块x维（需乘ic）
    int share_y=(blockDim.y>>1)+1;//共享内存块y维
    int share_idx_x;
    int share_idx_y;

    for (int c = 0; c <ic ; ++c) {
        share_idx_x = threadIdx.x + blockDim.x * c;//共享内存块内x索引
        share_idx_y = threadIdx.y;//共享内存块内y维索引

        int in_x0 = ((blockIdx.x * blockDim.x) >> 1) * ic;
        int in_y0 = (blockIdx.y * blockDim.y) >> 1;

        if (share_idx_x < (share_x * ic) && share_idx_y < share_y)
        {
            int x = min(in_x0 + share_idx_x, iw * ic - ic + share_idx_x % ic);
            int y = min(in_y0 + share_idx_y, ih - 1);
            data[share_idx_y * share_x * ic + share_idx_x] = in[y * iw * ic + x];
        }

    }
    __syncthreads();
    for (int c = 0; c <ic ; ++c) {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic)
        {
            share_idx_x = threadIdx.x + blockDim.x * c;
            share_idx_y = threadIdx.y;

            int yoff[2]={(share_idx_y>>1)*share_x*ic,((share_idx_y+1)>>1)*share_x*ic};
            int xoff[2]={(share_idx_x/ic>>1)*ic+share_idx_x%ic,
                         ((share_idx_x/ic+1)>>1)*ic+share_idx_x%ic};
            float val[4]={data[yoff[0]+xoff[0]],
                          data[yoff[0]+xoff[1]],
                          data[yoff[1]+xoff[0]],
                          data[yoff[1]+xoff[1]]};
            int out_idx=out_y*ow*ic+fact_x;
            out[out_idx]=0.25f*(val[0]+val[1]+val[2]+val[3]);
        }
    }*/

}

void double_size_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels)
{
    int const ow=weight<<1;
    int const oh=height<<1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    int const bytes_in=size_in*channels* sizeof(float);
    int const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);

    int const  x=32;
    int const  y=16;
    int const  share_x=((x>>1)+1);
    int const  share_y=(y>>1)+1;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*3)/(x*3),(oh-1+y)/y,1);

    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    //kernel_doublesizebyshare<<<grid,block,share_x*share_y*3*channels*sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    kernel_doublesize<<<grid,block>>>(d_out,d_in,ow,oh,weight,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

}
