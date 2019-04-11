#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
/**
 * @property    图像饱和度降低
 * @func        将图像转换为几种HSL图像               
 * @param_out   out_image          转换后的图像          
 * @param_in    in_image           待转换图像 
 * @param_in    pixel_amount       像素点个数 
 * @param_in    type               亮度类型
 * @param_in    alpha              是否有alpha通道
 */
void desaturate_by_cuda(float * const out_image,float const  *in_image,const int pixel_amount, const int type,const bool alpha);
/****************************************************************************************************************************/

__global__ void kernel_desaturate_alpha(float *out,float const *in, const int size,const int type)
{
    extern __shared__   float s[];
    int in_idx = threadIdx.x  + blockIdx.x * blockDim.x * 8 ;
    int out_idx = threadIdx.x+ blockIdx.x * blockDim.x * 4 ;
    int tid=threadIdx.x;
    int stride=tid*4;
    int stride1=stride+blockDim.x*4;
    if (in_idx< size * 4)
    {
        s[tid]=in[in_idx];
        s[tid+blockDim.x]=in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=in[in_idx+blockDim.x*5];
        s[tid+blockDim.x*6]=in[in_idx+blockDim.x*6];
        s[tid+blockDim.x*7]=in[in_idx+blockDim.x*7];
    }
    __syncthreads();

    if(type==0)
    {
        out[out_idx]=max(s[stride+0],max(s[stride+1],s[stride+2]));
        out[out_idx+blockDim.x*2]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(type==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        out[out_idx+blockDim.x*2]=0.5f*(max_s+min_s);
    }
    if(type==2)
    {
        out[out_idx]=0.21f * s[stride+0] + 0.72f * s[stride+1] + 0.07f * s[stride+2];
        out[out_idx+blockDim.x*2]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(type==3)
    {
        out[out_idx]=0.30f * s[stride+0] + 0.59f * s[stride+1] + 0.11f * s[stride+2];
        out[out_idx+blockDim.x*2]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(type==4)
    {
        out[out_idx]=((float)(s[stride+0] + s[stride+1] + s[stride+2])) / 3.0f;
        out[out_idx+blockDim.x*2]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }
    out[out_idx+tid+1]=s[stride+3];
    out[out_idx+blockDim.x*2+tid+1]=s[stride1+3];
}
__global__ void kernel_desaturate(float *out,float const *in, const int size,const int type)
{
    extern __shared__   float s[];
    int in_idx = threadIdx.x  + blockIdx.x * blockDim.x * 6 ;
    int out_idx = threadIdx.x+ blockIdx.x * blockDim.x * 2 ;
    int tid=threadIdx.x;
    int stride=tid*3;
    int stride1=stride+blockDim.x*3;

    if (in_idx< size * 3)
    {
        s[tid]=in[in_idx];
        s[tid+blockDim.x]=in[in_idx+blockDim.x];
        s[tid+blockDim.x*2]=in[in_idx+blockDim.x*2];
        s[tid+blockDim.x*3]=in[in_idx+blockDim.x*3];
        s[tid+blockDim.x*4]=in[in_idx+blockDim.x*4];
        s[tid+blockDim.x*5]=in[in_idx+blockDim.x*5];
    }
    __syncthreads();
    if(type==0)
    {
        out[out_idx]=max(s[stride+0],max(s[stride+1],s[stride+2]));
        out[out_idx+blockDim.x]=max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
    }
    if(type==1)
    {
        float const max_v = max(s[stride+0],max(s[stride+1],s[stride+2]));
        float const min_v = min(s[stride+0],min(s[stride+1],s[stride+2]));
        out[out_idx]=0.5f*(max_v+min_v);
        float const max_s = max(s[stride1+0],max(s[stride1+1],s[stride1+2]));
        float const min_s = min(s[stride1+0],min(s[stride1+1],s[stride1+2]));
        out[out_idx+blockDim.x]=0.5f*(max_s+min_s);
    }
    if(type==2)
    {
        out[out_idx]=0.21f * s[stride+0] + 0.72f * s[stride+1] + 0.07f * s[stride+2];
        out[out_idx+blockDim.x]=0.21f * s[stride1+0] + 0.72f * s[stride1+1] + 0.07f * s[stride1+2];
    }
    if(type==3)
    {
        out[out_idx]=0.30f * s[stride+0] + 0.59f * s[stride+1] + 0.11f * s[stride+2];
        out[out_idx+blockDim.x]=0.30f * s[stride1+0] + 0.59f * s[stride1+1] + 0.11f * s[stride1+2];
    }
    if(type==4)
    {
        out[out_idx]=((float)(s[stride+0] + s[stride+1] + s[stride+2])) / 3.0f;
        out[out_idx+blockDim.x]=((float)(s[stride1+0] + s[stride1+1] + s[stride1+2])) / 3.0f;
    }

}


void desaturate_by_cuda(float  * const out_image,float const *in_image,const int pixel_amount, const int type,const bool alpha)
{
    float *d_in=NULL;
    float *d_out=NULL;

    int bytes_in=pixel_amount*(3+alpha)*sizeof(float);
    int bytes_out=pixel_amount*(1+alpha)* sizeof(float);
    const int  blocksize=256;
    dim3 block(blocksize,1,1);
    dim3 grid((pixel_amount-1+blocksize*2)/(blocksize*2),1,1);
    cudaMalloc(&d_in,bytes_in);
    cudaMalloc(&d_out,bytes_out);
    cudaMemcpy(d_in,in_image,bytes_in,cudaMemcpyHostToDevice);
    if(alpha)
    {
        kernel_desaturate_alpha<<<grid,block,blocksize*4* sizeof(float)>>>(d_out,d_in,pixel_amount,type);
    }
    else
    {
        kernel_desaturate<<<grid,block,blocksize*6* sizeof(float)>>>(d_out,d_in,pixel_amount,type);
    }
    cudaMemcpy(out_image,d_out,bytes_out,cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
