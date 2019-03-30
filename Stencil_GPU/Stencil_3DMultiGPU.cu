#include <iostream>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <utility>	//C++11
#include "../time/seconds.h"

#define BLOCK 32
#define X 128
#define Y 128
#define Z 128
#define ELEM (size_t)(X*Y*Z)
#define STEP 1024
#define GPUNUM 4
#define SLV (1*X*Y)

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
 
//Multi版
__global__ void StencilOneStep(float* Src,float* Dst,const int MainElem,const int Dev){
	size_t index = threadIdx.x + blockDim.x * blockIdx.x;
	size_t mat_x = index % X;
	size_t mat_y = index / X;
	
	//デバイス番号によって動作が変わる
	switch(Dev){
		case 0:
			if(index>SLV+X*Y && index<MainElem+2*SLV-X*Y && mat_x != 0 && mat_x != X-1 && mat_y != 0 && mat_y != Y-1){
				Dst[index] = 0.4*Src[index] + 0.1*(Src[index-1]+Src[index+1]+Src[index+X]+Src[index-X]+Src[index+X*Y]+Src[index-X*Y]);
			}
			break;
		case GPUNUM-1:
			if(index>X*Y && index<MainElem+SLV-X*Y && mat_x != 0 && mat_x != X-1 && mat_y != 0 && mat_y != Y-1){
				Dst[index] = 0.4*Src[index] + 0.1*(Src[index-1]+Src[index+1]+Src[index+X]+Src[index-X]+Src[index+X*Y]+Src[index-X*Y]);
			}
			break;
		default:
			if(index>X*Y && index<MainElem+2*SLV-X*Y && mat_x != 0 && mat_x != X-1 && mat_y != 0 && mat_y != Y-1){
				Dst[index] = 0.4*Src[index] + 0.1*(Src[index-1]+Src[index+1]+Src[index+X]+Src[index-X]+Src[index+X*Y]+Src[index-X*Y]);
			}
			break;
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
	//HostTemp SLV
	float* Left = new float[SLV*GPUNUM];
	float* Right = new float[SLV*GPUNUM];
	double start,end;
	start = seconds();

	omp_set_num_threads(GPUNUM);

	#pragma omp parallel
	{
		size_t MainElem = ELEM/GPUNUM;
		size_t CalcElem = MainElem + 2*SLV;
		size_t MainSize = MainElem * sizeof(float);
		size_t SleeveSize = SLV*sizeof(float);//Single
		size_t DeviceMemorySize = CalcElem * sizeof(float);

		//Device番号の取得
		int Dev = omp_get_thread_num();
		CHECK(cudaSetDevice(Dev));
		
		 //cout << Dev << " : MainElem -> " <<MainElem << " : CalcElem -> " <<CalcElem << " : SLV -> " << SLV << endl;
		//実行定義
		dim3 block(BLOCK);
		dim3 grid((CalcElem+block.x-1)/block.x);

		//開始のアドレス(要素番号)
		size_t MainAddress = Dev*MainElem;

		// cout << Dev << " : StartAddress -> " << MainAddress << endl;

		float *d_Src,*d_Dst;
		CHECK(cudaMalloc(&d_Src,DeviceMemorySize));
		CHECK(cudaMalloc(&d_Dst,DeviceMemorySize));
		//Init
		CHECK(cudaMemset(d_Src,0,DeviceMemorySize));
		CHECK(cudaMemset(d_Dst,0,DeviceMemorySize));

		//Memcpy
		if(Dev==0){
			// cout << "Copy GPU : " << Dev << " : Src ["<<MainAddress<<"]" << " : "<< MainSize+SleeveSize <<"Byte" <<endl;
			CHECK(cudaMemcpy(&d_Src[SLV],&Src[MainAddress],MainSize+SleeveSize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[SLV],&Src[MainAddress],MainSize+SleeveSize,cudaMemcpyHostToDevice));
		}else if(Dev==GPUNUM-1){
			//cout << "Copy GPU : " << Dev << " : Src ["<<MainAddress<<"]" << " : "<< MainSize+SleeveSize <<"Byte" <<endl;
			CHECK(cudaMemcpy(&d_Src[0],&Src[MainAddress-SLV],MainSize+SleeveSize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[0],&Src[MainAddress-SLV],MainSize+SleeveSize,cudaMemcpyHostToDevice));
		}else{
			//cout << "Copy GPU : " << Dev << " : Src ["<<MainAddress<<"]" << " : "<< MainSize+2*SleeveSize <<"Byte" <<endl;
			CHECK(cudaMemcpy(&d_Src[0],&Src[MainAddress-SLV],MainSize+2*SleeveSize,cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(&d_Dst[0],&Src[MainAddress-SLV],MainSize+2*SleeveSize,cudaMemcpyHostToDevice));
		}
//		cout << "block : "<< block.x << " | grid : " << grid.x << endl;
		for(int st=0;st<STEP;st++){
			//Stencil Calc
			//cout << Dev <<" : iter -> " << st << endl;
			StencilOneStep<<<grid,block>>>(d_Src,d_Dst,MainElem,Dev);
			swap(d_Src,d_Dst);
			if(Dev!=0){
				CHECK(cudaMemcpy(&Left[Dev*SLV],&d_Src[SLV],SleeveSize,cudaMemcpyDeviceToHost));
			}
			if(Dev!=GPUNUM-1){
				CHECK(cudaMemcpy(&Right[Dev*SLV],&d_Src[MainElem],SleeveSize,cudaMemcpyDeviceToHost));
			}

			#pragma omp barrier

			if(Dev!=0){
				CHECK(cudaMemcpy(&d_Src[0],&Right[(Dev-1)*SLV],SleeveSize,cudaMemcpyHostToDevice));
			}
			if(Dev!=GPUNUM-1){
				CHECK(cudaMemcpy(&d_Src[SLV+MainElem],&Left[(Dev+1)*SLV],SleeveSize,cudaMemcpyHostToDevice));
			}
		}

		CHECK(cudaMemcpy(&Rst[Dev*MainElem],&d_Src[SLV],MainSize,cudaMemcpyDeviceToHost));
		CHECK(cudaGetLastError());

		CHECK(cudaFree(d_Src));
		CHECK(cudaFree(d_Dst));
	}
	end = seconds();

	//Host3DStencil(Src,Dst);
	//checkResult(Src,Rst,ELEM);

	printf("------------------------------------------------\n");
	printf("Program : %s\n ",argv[0]);
	printf("STEPS : %d\n X:%d Y:%d Z:%d", STEP,X,Y,Z);
	printf("GPU : %d | ELEMENTS : %d  \n",GPUNUM,ELEM);
	printf("Elapsed Time : %lf\n",end-start);
	printf("------------------------------------------------\n");


//	print(Def,Src,Rst,ELEM);
	delete Src;
	delete Dst;
	delete Rst;
	delete Def;
	return 0;
}
