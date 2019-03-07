#include <iostream>
#include <cstring>
#include <cstdlib>
#include <utility>	//C++11
#include <cuda.h>
#define BLOCK 32
#define ELEM 64
#define STEP 32
using namespace std;

#define CHECK(call)										\
{															\
	const cudaError_t error = call;						\
	if(error != cudaSuccess){								\
		cerr << "Error:" << __FILE__ << endl; \
		cerr << "code : "<< error << " reason : "<<cudaGetErrorString(error) << endl;	\
	}														\
}

void checkResult(float* hostRef,float* devRef,const int N){
  float epsilon = 1.0E-4;
  bool match = 1;
  int i;
  for(i=0;i<N;i++){
    //printf("host:%d,device:%d\n",hostRef[i],devRef[i]);
    if(abs(hostRef[i]-devRef[i])>epsilon){
      match = 0;
      cout << "Arrays don't match.on count of "<<i<< " element.";
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

__global__ void StencilOneStep(float* Src,float* Dst,const int elem){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index!=0&&index<elem-1)
		Dst[index] = 0.6*Src[index] + 0.2*(Src[index-1]+Src[index+1]);
}

int main(int argc,char** argv){
	float* Src = new float[ELEM];
	float* Dst = new float[ELEM];
	float* Rst = new float[ELEM];
	float* Def = new float[ELEM];
	//Srcを乱数で初期化
	initializeData(Src,ELEM);
	memcpy(Dst,Src,sizeof(float)*ELEM);
	memcpy(Def,Src,sizeof(float)*ELEM);
	//Deviceメモリの確保
	size_t DeviceMemorySize = ELEM*sizeof(float);
	float *d_Src,*d_Dst;
	CHECK(cudaSetDevice(0));
	CHECK(cudaMalloc(&d_Src,DeviceMemorySize));
	CHECK(cudaMalloc(&d_Dst,DeviceMemorySize));

	CHECK(cudaMemcpy(d_Src,Src,DeviceMemorySize,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Dst,Src,DeviceMemorySize,cudaMemcpyHostToDevice));

	dim3 block(BLOCK);
	dim3 grid((ELEM+block.x-1)/block.x);

	cout << "block : "<< block.x << " | grid : " << grid.x << endl;

	for(int st=0;st<STEP;st++){
		StencilOneStep<<<grid,block>>>(d_Src,d_Dst,ELEM);
		swap(d_Src,d_Dst);
	}
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(Rst,d_Src,DeviceMemorySize,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_Src));
	CHECK(cudaFree(d_Dst));

	Host1DStencil(Src,Dst);
	checkResult(Src,Rst,ELEM);

	print(Def,Src,Rst,ELEM);
	delete Src;
	delete Dst;
	delete Rst;
	delete Def;
	return 0;
}