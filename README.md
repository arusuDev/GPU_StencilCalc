# CollegeResearch
## 研究で利用したStencil計算のあれこれ+GPUDirectのプログラムをおいてます。
GPUDirectを用いない従来手法のプログラムはフォルダにまとめてます。

### 内容物
ディレクトリ  
Binary - 4GPUで動作するバイナリファイルをおいています。アンダーバー以降は、3次元配列のXYZのサイズ、ステンシル計算のステップ回数を表しています。  
Stencil_GPU - P2Pを利用しない各種ステンシル計算が入っています。  
time - 時間を返す関数が入っているヘッダファイルが入っています。

ファイル
Makefile - Stencil_P2P.cuをコンパイルします。
Stencil3D.sh - ./Binary以下の各種バイナリファイルを実行します。
Stencil3D_nvprof.sh - ./Binary以下のバイナリファイルをnvprofオプションを付けて実行します。
