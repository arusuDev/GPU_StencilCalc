#include <iostream>
#include <chrono> //C++11
#include <cstring>
#include <cstdlib>
#include <utility>	//C++11

#define BLOCK 32

#define X 1024
#define Y 1024

#define ELEM (size_t)(X*Y)
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
/*
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
*/

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
/*
void Host1DStencil(float* Src,float* Dst){
	for(int st=0;st<STEP;st++){
		for(int i=0;i<ELEM;i++){
			if(i!=0 && i<ELEM-1)
				Dst[i] = 0.6*Src[i] + 0.2*(Src[i-1]+Src[i+1]);
		}
		swap(Src,Dst);
	}
}
*/

void Host2DStencil(float* Src,float* Dst){
	for(int time_step=0;time_step<STEP;time_step++){
		for(size_t all_loop=0;all_loop<ELEM;all_loop++){
			int mat_x=all_loop%X;//X成分
			int mat_y=all_loop/X;//Y成分
			//cout << "Time: "<<time_step<< " | X:" << mat_x << " | Y:"<<mat_y ;
			if(mat_x!=0 && mat_x!=X-1 && mat_y!=0 && mat_y!=Y-1){
				//端ならば計算しない
				Dst[all_loop] = 0.6*Src[all_loop] + 0.1*(Src[all_loop-1]+Src[all_loop+1]+Src[all_loop-X]+Src[all_loop+X]);
			}
			cout << endl;
		}
		swap(Src,Dst);
	}
}

int main(int argc,char** argv){
	float* Src = new float[ELEM];
	float* Dst = new float[ELEM];

//	float* Rst = new float[ELEM];
//	float* Def = new float[ELEM];
	//Srcを乱数で初期化
	initializeData(Src,ELEM);

	memcpy(Dst,Src,sizeof(float)*ELEM);

//	memcpy(Def,Src,sizeof(float)*ELEM);
	//Deviceメモリの確保
	/*
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
*/
	Host2DStencil(Src,Dst);
//	print(Src);
//	checkResult(Src,Rst,ELEM);

//	print(Def,Src,Rst,ELEM);
	delete Src;
	delete Dst;
//	delete Rst;
//	delete Def;
	return 0;
}