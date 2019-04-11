/******************************************************************************************/
///功能：图片缩小两倍
/*  函数名                         线程块大小       耗费时间
 *  kernel_halfsizebyshare1       [32,4,1]      582.797us
 *  kernel_halfsize               [32,8,1]      640.097us
 *  kernel_halfsize1              [32,4,1]      638.37us
 *  kernel_halfsizebyshare        [32,4,1]      607.5us
 */
/******************************************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
/**
 * @property    图像缩放
 * @func        将图像缩小为原图两倍　 像素点为(2*y,2*x)(2*y,2*x+1)(2*y+1,2*x)(2*y+1,2*x+1)的平均值
 *                                 若最后一行或最后一列为奇数列．则越界部分再取最后一行或最后一列
 * @param_out   out_image          放大后的图像首地址
 * @param_in    in_image           待放大图像首地址
 * @param_in    weight             输入图像的宽度
 * @param_in    height             输入图像的高度
 * @param_in    channels           输入图像的颜色通道数
 * 调用示例：
 * halfsize_by_cuda(&out_image->at(0),&img->at(0),img->width(),img->height(),img->channels());
 */
void halfsize_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels);
/**********************************************************************************************************************************/
__global__ void kernel_halfsizebyshare1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*2;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*blockDim.x*2*ic*2;//输入图像x维的第一起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的第一起始点
    int in_x1=in_x0+blockDim.x*ic*2;//输入图像x维的第二起始点
    int in_y1=in_y0+blockDim.y;//输入图像y维的第二起始点

    int share_x=blockDim.x*4;//共享块内x维最大像素点个数
    for (int c = 0; c < ic*2; ++c)
    {
        int fact_x_s=threadIdx.x+blockDim.x*c;//共享内存内第一个x的索引
        int x_s=fact_x_s+blockDim.x*ic*2;//共享内存内第二个x的索引
        int y_s0=threadIdx.y*share_x*ic;//共享内存内第一个y的索引
        int y_s1=y_s0+blockDim.y*share_x*ic;//共享内存内第二个y的索引
        int fact_iw=fact_x_s%ic+stride-ic;

        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;
        data[y_s0+fact_x_s]=in[y0+x0];
        data[y_s0+x_s]=in[y0+x1];
        data[y_s1+fact_x_s]=in[y1+x0];
        data[y_s1+x_s]=in[y1+x1];
    }
    __syncthreads();
    for (int c = 0; c <ic*2 ; ++c) {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int fact_x_s=threadIdx.x+blockDim.x*c;
            int srow1=threadIdx.y*2*share_x*ic;
            int srow2=srow1+share_x*ic;
            int scol1=(fact_x_s / ic) * 2 * ic + fact_x_s % ic;
            int scol2=scol1 + ic;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}
__global__ void kernel_halfsize(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=iw*ic;

    for(int c=0;c<ic;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < ih);
            int icol1 = (fact_x / ic) * 2 * ic + fact_x % ic;
            int icol2 = min((icol1 + ic), (iw * ic - ic + fact_x % ic));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (in[index[0]] + in[index[1]] + in[index[2]] + in[index[3]]);
        }

    }
}
__global__ void kernel_halfsize1(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    //若需要展开ic*3重循环只需修改out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*3;以及for(int c=0;c<ic*3;c++)即可，同时应修改网格大小
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic*2;
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;
    int stride=iw*ic;

    for(int c=0;c<ic*2;c++)
    {
        int fact_x=out_x+blockDim.x*c;
        if(out_y<oh&&fact_x<ow*ic) {
            int irow1 = out_y * 2 * stride;
            int irow2 = irow1 + stride * (out_y * 2 + 1 < ih);
            int icol1 = (fact_x / ic) * 2 * ic + fact_x % ic;
            int icol2 = min((icol1 + ic), (iw * ic - ic + fact_x % ic));
            int index[4] = {irow1 + icol1,
                            irow1 + icol2,
                            irow2 + icol1,
                            irow2 + icol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (in[index[0]] + in[index[1]] + in[index[2]] + in[index[3]]);
        }

    }
}
__global__ void kernel_halfsizebyshare(float *out,float *in,int const ow,int const oh,int const iw,int const ih,int const ic)
{
    extern __shared__ float data[];
    int out_x=threadIdx.x+blockIdx.x*blockDim.x*ic;//输出的x维起始索引
    int out_y=threadIdx.y+blockIdx.y*blockDim.y;//输出的y位索引
    int stride=iw*ic;//输入图像的行索引的最大值

    int in_x0=blockIdx.x*blockDim.x*2*ic;//输入图像x维的起始点
    int in_y0=blockIdx.y*blockDim.y*2;//输入图像y维的起始点
    int in_x1=in_x0+blockDim.x*ic;
    int in_y1=in_y0+blockDim.y;

    int share_x=blockDim.x*2;//共享块内x维最大像素点个数
    for (int c = 0; c < ic; ++c)
    {
        int fact_x_s=threadIdx.x+blockDim.x*c;
        int x_s=fact_x_s+blockDim.x*ic;
        int y_s0=threadIdx.y*share_x*ic;
        int y_s1=y_s0+blockDim.y*share_x*ic;
        int fact_iw=fact_x_s%ic+stride-ic;
        int x0=min(in_x0+fact_x_s,fact_iw);
        int x1=min(in_x1+fact_x_s,fact_iw);
        int y0=min(in_y0+threadIdx.y,ih-1)*stride;
        int y1=min(in_y1+threadIdx.y,ih-1)*stride;
        data[y_s0+fact_x_s]=in[y0+x0];
        data[y_s0+x_s]=in[y0+x1];
        data[y_s1+fact_x_s]=in[y1+x0];
        data[y_s1+x_s]=in[y1+x1];
    }
    __syncthreads();
    for (int c = 0; c <ic ; ++c)
    {
        int fact_x=out_x+blockDim.x*c;

        if(out_y<oh&&fact_x<ow*ic)
        {
            int fact_x_s=threadIdx.x+blockDim.x*c;
            int srow1=threadIdx.y*2*share_x*ic;
            int srow2=srow1+share_x*ic;
            int scol1=(fact_x_s / ic) * 2 * ic + fact_x_s % ic;
            int scol2=scol1 + ic;
            int index[4] = {srow1 + scol1,
                            srow1 + scol2,
                            srow2 + scol1,
                            srow2 + scol2};
            int out_idx = out_y * ow*ic + fact_x;
            out[out_idx] = 0.25f * (data[index[0]] + data[index[1]] + data[index[2]] + data[index[3]]);
        }
    }
}

void halfsize_by_cuda(float * const out_image,float const  * const in_image, int const weight,int const height,int const channels)
{
    int ow=(weight+1)>>1;
    int oh=(height+1)>>1;
    int const size_in=weight*height;
    int const size_out=ow*oh;
    int const bytes_in=size_in*channels* sizeof(float);
    int const bytes_out=size_out*channels* sizeof(float);

    float *d_in=NULL;
    float *d_out=NULL;
    cudaMalloc((void**)&d_in,bytes_in);
    cudaMalloc((void**)&d_out,bytes_out);

    int const  x=32;
    int const  y=4;
    int const   share_x=x*4;
    int const   share_y=y*2;
    dim3 block (x,y,1);
    dim3 grid ((ow-1+x*2)/(x*2),(oh-1+y)/y,1);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    kernel_halfsizebyshare1<<<grid,block,share_x*share_y*channels* sizeof(float)>>>(d_out,d_in,ow,oh,weight,height,channels);
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

