//1次元配列における複数台GPUを用いたステンシル計算


#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <utility>	//C++11
#include <cuda.h>
#include <omp.h>

#define GPUNUM 2
#define BLOCK 32
#define ELEM 1024
#define STEP 512
#define SLV 1
using namespace std;

#define CHECK(call)										\
{															\
	const cudaError_t error = call;						\
	if(error != cudaSuccess){								\
		cerr << "Error:" << __FILE__ << endl; \
		cerr << "code : "<< error << " reason : "<<cudaGetErrorString(error) << endl;	\
	}														\
}

__global__ void printDeviceMemory(float* Src,const int elem){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < elem){
		printf("%d | %f \n",index,Src[index]);
	}
}
__global__ void StencilOneStep(float* Src,float* Dst,const int MainElem,const int SleeveElem,const int Dev){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	switch (Dev){
		case 0:
			if(index>SleeveElem && index<MainElem+2*SleeveElem-1){
				Dst[index] = 0.6*Src[index] + 0.2*(Src[index-1]+Src[index+1]);
			}
			break;

		case GPUNUM-1:
			if(index>0 && index<MainElem+SleeveElem-1){
				Dst[index] = 0.6*Src[index] + 0.2*(Src[index-1]+Src[index+1]);
			}
			break;
		default:
			if(index>0 && index<MainElem+2*SleeveElem-1){
				Dst[index] = 0.6*Src[index] + 0.2*(Src[index-1]+Src[index+1]);
			}
			break;
	}	
}

void checkResult(float* hostRef,float* devRef,const int N){
  float epsilon = 1e-3;
  bool match = 1;
  int i;
  for(i=0;i<N;i++){
    //printf("host:%d,device:%d\n",hostRef[i],devRef[i]);
    if(fabsf(hostRef[i]-devRef[i])>epsilon){
      match = 0;
      cout << "Arrays don't match.on count of "<<i<< " element.";
      cout << "Host : " << hostRef[i] << " | GPU : " << devRef[i] << endl; 
      break;
    }
  }
  if(match){
    cout <<"Arrays match.";
  }
  cout << endl;
  return;
}


void initializeData(float* A,int size){
  //乱数で値を初期化します。
  time_t t;
  int i;
  srand((unsigned int)time(&t));

  for(i=0;i<size;i++){
    A[i] = (float)(rand()) / 10.0F;
  }
  return;
}

void print(float* Def,float* Src,float* Rst,const int elem){
	for(int i=0;i<elem;i++){
		cout << "\t" <<i << " | " << Def[i] << " | " <<Src[i] << " | "<<Rst[i] << endl;
	}
}

void Host1DStencil(float* Src,float* Dst){
	for(int st=0;st<STEP;st++){
		for(int i=0;i<ELEM;i++){
			if(i!=0 && i<ELEM-1)
				Dst[i] = 0.6*Src[i] + 0.2*(Src[i-1]+Src[i+1]);
		}
		swap(Src,Dst);
	}
}

void checkDeviceMemory(float* d_Src,int Dev,int CalcNum,dim3 grid,dim3 block){
	//GPUの中身確認したい時に使う関数

	for(int i=0;i<GPUNUM;i++){
		if(i==Dev){
			cout << i << endl;
			printDeviceMemory<<<grid,block>>>(d_Src,CalcNum);
			cout << endl;
		}
		CHECK(cudaDeviceSynchronize());
	}

}
int main(int argc,char** argv){
	float* Src = new float[ELEM];
	float* Dst = new float[ELEM];
	float* Rst = new float[ELEM];
	float* Def = new float[ELEM];//Default
	//Srcを乱数で初期化
	initializeData(Src,ELEM);
	memcpy(Dst,Src,sizeof(float)*ELEM);
	memcpy(Def,Src,sizeof(float)*ELEM);
	omp_set_num_threads(GPUNUM);

	//Dev*SLV
	float* Left = new float[SLV*GPUNUM];
	float* Right = new float[SLV*GPUNUM];
	
	//Deviceメモリの確保
	#pragma omp parallel
	{
		size_t CalcNum = ELEM/GPUNUM+2*SLV;
		//1GPUあたりのメイン領域
		size_t MainNum = ELEM/GPUNUM;
		size_t MainSize = MainNum*sizeof(float);
		//他のGPUから必要になる袖領域 Simple Ver.
		size_t SleeveSize = SLV*sizeof(float);
		//合計のメモリサイズ
		size_t DeviceMemorySize = MainSize + 2*SleeveSize;
		
		int Dev = omp_get_thread_num();
		CHECK(cudaSetDevice(Dev));

		size_t MainAddress = Dev*ELEM/GPUNUM;

		float *d_Src,*d_Dst;
		CHECK(cudaMalloc(&d_Src,DeviceMemorySize));
		CHECK(cudaMalloc(&d_Dst,DeviceMemorySize));

		//袖領域の関係上一回0でinit
		CHECK(cudaMemset(d_Src,0,DeviceMemorySize));
		CHECK(cudaMemset(d_Dst,0,DeviceMemorySize));

		if(Dev==0){
			CHECK(cudaMemcpy(&d_Src[SLV],&Src[MainAddress],MainSize+SleeveSize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[SLV],&Src[MainAddress],MainSize+SleeveSize,cudaMemcpyHostToDevice));
		}else if(Dev==GPUNUM-1){
			CHECK(cudaMemcpy(&d_Src[0],&Src[MainAddress-SLV],MainSize+SleeveSize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[0],&Src[MainAddress-SLV],MainSize+SleeveSize,cudaMemcpyHostToDevice));						
		}else{
			CHECK(cudaMemcpy(&d_Src[0],&Src[MainAddress-SLV],DeviceMemorySize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[0],&Src[MainAddress-SLV],DeviceMemorySize,cudaMemcpyHostToDevice));			
		}
		dim3 block(BLOCK);
		dim3 grid((CalcNum+block.x-1)/block.x);

		//checkDeviceMemory(d_Src,Dev,CalcNum,block,grid);
		
		cout << "block : "<< block.x << " | grid : " << grid.x << endl;
		
		for(int st=0;st<STEP;st++){
			//Stencil Calc
			StencilOneStep<<<grid,block>>>(d_Src,d_Dst,MainNum,SLV,Dev);
			swap(d_Src,d_Dst);
			if(Dev!=0){
				CHECK(cudaMemcpy(&Left[Dev*SLV],&d_Src[SLV],SleeveSize,cudaMemcpyDeviceToHost));
			}
			if(Dev!=GPUNUM-1){
				CHECK(cudaMemcpy(&Right[Dev*SLV],&d_Src[MainNum],SleeveSize,cudaMemcpyDeviceToHost));
			}
			#pragma omp barrier
			//cudaMemcpyはホストと同期的に行う関数だからいらないかも
			//CHECK(cudaDeviceSynchronize());
			if(Dev!=0){
				CHECK(cudaMemcpy(&d_Src[0],&Right[(Dev-1)*SLV],SleeveSize,cudaMemcpyHostToDevice));
			}
			if(Dev!=GPUNUM-1){
				CHECK(cudaMemcpy(&d_Src[SLV+MainNum],&Left[(Dev+1)*SLV],SleeveSize,cudaMemcpyHostToDevice));
			}
		}
		
		CHECK(cudaMemcpy(&Rst[Dev*MainNum],&d_Src[SLV],MainSize,cudaMemcpyDeviceToHost));

		CHECK(cudaGetLastError());

//		CHECK(cudaMemcpy(Rst,d_Src,DeviceMemorySize,cudaMemcpyDeviceToHost));
		CHECK(cudaFree(d_Src));
		CHECK(cudaFree(d_Dst));
	}
	Host1DStencil(Src,Dst);
	checkResult(Src,Rst,ELEM);

	//print(Def,Src,Rst,ELEM);
	delete Src;
	delete Dst;
	delete Rst;
	delete Def;
	return 0;
}