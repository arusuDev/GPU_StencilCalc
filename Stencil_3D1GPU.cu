#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <utility>	//C++11

#define BLOCK 32

#define X 32
#define Y 32
#define Z 32

#define ELEM (size_t)(X*Y*Z)
#define STEP 128
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
  float epsilon = 1e-3;
  bool match = 1;
  int i;
  float Ref;
  float Host;
  float Dev;
  for(i=0;i<N;i++){
  	Host = hostRef[i];
  	Dev = devRef[i];
  	Ref = Host-Dev;

    //printf("host:%d,device:%d\n",hostRef[i],devRef[i]);
    if((float)fabsf(Ref)>epsilon){
      match = 0;
      cout << "Arrays don't match.on count of "<<i<< " element." <<endl;
      cout << "Elapsed : " << Ref << " Host : " << Host << " | GPU : " << Dev << endl; 
      printf("Elapsed : %f Host : %f GPU : %f\n",Ref,Host,Dev );
      break;
    }
  }
  if(match){
    cout <<"Arrays match.";
  }
  cout << endl;
  return;
}

void initializeData(float* A,const int size){
  //乱数で値を初期化します。
  time_t t;
  int i;
  srand((unsigned int)time(&t));

  for(i=0;i<size;i++){
    A[i] = (float)(rand()&0xFFFF) / 0xFFFF;
  }
  return;
}
void print(float* Src){
	for(int i=0;i<ELEM;i++){
		cout << Src[i] << " ";
		if((i+1)%X==0)
			cout << endl;
	}
}
void print(float* Def,float* Src,float* Rst,const int elem){
	for(int i=0;i<elem;i++){
		cout << "\t" <<i << " | " << Def[i] << " | " <<Src[i] << " | "<<Rst[i] << endl;
	}
}

void Host3DStencil(float* Src,float* Dst){
	for(int time_step=0;time_step<STEP;time_step++){
		for(int all_loop=0;all_loop<ELEM;all_loop++){
			int mat_x = all_loop%X;
			int mat_y = all_loop/X;
			int mat_z = all_loop/(X*Y);

			if(mat_x!=0 && mat_x!=X-1 && mat_y!=0 && mat_y!= Y-1 && mat_z!=0 && mat_z!=Z-1){
				Dst[all_loop] = 0.4*Src[all_loop] + 0.1*(Src[all_loop-1]+Src[all_loop+1]+Src[all_loop-X]+Src[all_loop+X]+Src[all_loop-X*Y]+Src[all_loop+X*Y]);
			}
		}
		swap(Src,Dst);
	}
}

//Single版
__global__ void StencilOneStep(float* Src,float* Dst){
	size_t  index = threadIdx.x + blockDim.x * blockIdx.x;
	size_t mat_x = index % X; //X成分
	size_t mat_y = index / X; //Y成分
	size_t mat_z = index / (X*Y);
	if(mat_x != 0 && mat_x != X-1 && mat_y != 0 && mat_y != Y-1 && mat_z != 0 && mat_z != Z-1){
		Dst[index] = 0.4*Src[index] + 0.1*(Src[index-1] + Src[index+1] + Src[index+X] + Src[index-X] + Src[index+X*Y] + Src[index-X*Y]);
	}
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
		StencilOneStep<<<grid,block>>>(d_Src,d_Dst);
		swap(d_Src,d_Dst);
	}
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(Rst,d_Src,DeviceMemorySize,cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_Src));
	CHECK(cudaFree(d_Dst));
	Host3DStencil(Src,Dst);
//	print(Src);
	checkResult(Src,Rst,ELEM);

//	print(Def,Src,Rst,ELEM);
	delete Src;
	delete Dst;
	delete Rst;
	delete Def;
	return 0;
}