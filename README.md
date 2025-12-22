# Starlab 7년차 <br />
## Mesh Segmentation <br />
### Set Environment 🚀
```
conda env create -f environment.yml
conda activate mcnenv
```

### Download Princeton Segmentation Benchmark Dataset for Model 📦️
[Download Dataset](https://drive.google.com/file/d/1-ONm3XcOQawvkIlS_XXmtZj32-AjGGap/view?usp=drive_link) 링크를 눌러 다운 받으신 후 해당 repository의 가장 상위 폴더에 datasets가 위치하게 압축 해제 해두시면 됩니다.

### Create Noise to Dataset 💥
이름은 noise_pclass{번호}로 통일됩니다.
1. 기존 dataset의 동일 pclass 내부에서 가져와야하는 파일들
   1. classes.txt, mean_std_cache.p 파일 생성
   + classes.txt 및 mean_std_cache.p는 run_train을 돌리면 생성되며 복사해서 옮겨주시면 됩니다.
   2. test 폴더
   + test mesh가 들어있는 test폴더도 복사해주시면 됩니다.
2. seg, sseg는 add_noise_to_label.py를 실행해주면 현재 동일 pclass의 seg, sseg로부터 3%의 noise가 추가된 segmentation이 생성됩니다.

### Train Mesh Segmentation Model using PSB 🏋️
```
bash ./run_train.sh
```


### Test Mesh Segmentation Model using PSB 🧪
```
bash ./run_test1.sh
bash ./run_test2.sh
```

### Explanation of Direct Functions 💡

**1. prince_seg_preprocess.py** <br />
+ simplified mesh and label을 가지고 edge label, soft edge label을 계산하는 프로그램이다

**2. prince_seg_preprocess_disconnected.py** <br />
+ simplified mesh에서 모든 face가 disconnected인 경우<br />
다시 face를 계산 후 저장해줌 + edge label, soft edge label을 계산 및 저장해줌

**3. simp_visualize.py (DEBUG)** <br />
+ simplified 된 mesh의 visualization 결과를 보여줍니다.<br />
하나의 모델 당 여러 사람이 분류한 segmentation gt가 존재합니다.<br />
이중 균일한 gt를 사용해야하기 때문에 확인용으로 만들어두었습니다.

+ 모든 Segmentation 확인하기<br />
L52-56을 돌리면 각 class의 mesh별 모든 segmentation을 볼 수 있습니다.<br />
Segmentation이 21개 이하인 경우만 볼 수 있고 더 많은 경우에는 스킵하도록 설정되어있습니다.

+ List를 기반으로 균일한 Segmentation 확인하기<br />
L47-51를 돌리면 각 class에서 선택한 하나의 균일한 segmentation을 볼 수 있습니다.<br />
simp_seg_label 파일에 txt파일로 각 클래스별 segmentation division이 저장되어 있습니다.<br />
해당되는 클래스의 list를 복사하여 dictionary부분에 복사하면 됩니다.

**4. simp_save_seg.py (DEBUG)** <br />
mesh의 label을 재배치 (레이블 숫자만 바꿔줌, gt segmentation은 바꾸지 않는다)<br />
현재는 debug 모드로 돌리게 되어있으며 visu.vis_face_seg(points, faces, new_seg) (L98)을 주석처리하면 명령어로 돌릴 수 있다.

**4.1 분류된 segmentation dictionary 및 gt 파일 이름** <br />
simp_seg_label 파일에 txt파일로 각 클래스별 segmentation division이 저장되어 있습니다.<br />
해당되는 클래스의 dictionary를 복사하여 dictionary부분에 복사하면 됩니다.

**4.2. L42 Loop 범위 정하기** <br />
아래 Class에 해당되는 두 숫자를 range안에 적어주면 됩니다.
```
class1: 0,20
class2: 20,40
class3: 40,60
class4: 60,80
class5: 80,100
...
class13: 240-260 (241-260)
(20개 빠져있음)
class14: 260,280 (281-300)
...
class19: 360,380 (381-400)
```
**4.3. 생성된 seg, sseg 결과 옮기기** <br />
prince_simp_1000파일 내부에 seg, sseg파일이 생성됩니다.<br />
이 두 파일을 복사하여 해당하는 pclass의 하위 폴더에 넣으면 됩니다.

### Explanation of Indirect Functions ✨
1. pre_util.py: 파일 불러오기 및 저장 관련 함수들
2. edge_label.py: eseg, seseg 계산 및 weld vertex, weld faces 계산해주는 함수들
3. visualize.py: mesh visualization 관련 함수들

## (Optional) Additional Code for Data Preprocessing<br />
### Steps to Maintain Consistent Number of Points, Edge, Face<br />
1. Create /data folder inside dataprep
```
mkdir ./scripts/dataprep/data
```
3. Download [Princeton Benchmark](https://segeval.cs.princeton.edu/)
4. Unzip the downloaded benchmark
5. Place the directory 'gt', 'seg' inside /data
6. Download [blender](https://www.blender.org/download/) from official site.
7. Run script<br />
   ```
   bash ./scripts/dataprep/simplify_prince_with_label.sh
   ```
   (install related library if not installed)
8. The processed object file(.obj) and label files(.npz) are created in 'gt_simp' directory.<br />
If you want to separate these two files in separate folder, use below command<br />
   ```
   mkdir -p prince_simp_vertnum/seg_simp
   mv -n gt_simp/*.npz prince_simp_vertnum/seg_simp
   ```
