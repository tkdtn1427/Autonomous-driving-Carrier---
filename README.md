## 📌 보행 장애인을 위한 자율주행 캐리어
> ### 혼자서 거동이 힘든 분들을 위해 뱃지를 통해 스스로 보행자를 따라오는 자율 주행 캐리어


![](https://user-images.githubusercontent.com/81614803/200113921-a4debf27-a91f-4074-ace0-d1b90a928b27.jpg)

</br></br>

### 개발기간: 2021.03.01 - 2021.06.13

</br></br>

## 📌About

### - 보행 장애인의 경우 여행이나 출장 등 먼곳에 가야할 때, 휠체어에 앉거나 목발을 짚은 사람들은 혼자 캐리어를 들고가기 어려운 상황이 많습니다. 
### - 따라서, 이러한 분들이 먼곳을 혼자 다녀올 수 있게 사용자를 따라 다니는 자율주행 캐리어를 개발하였습니다. 
<p>
  <img src="https://user-images.githubusercontent.com/81614803/200116843-2102afe9-0f18-4364-bf47-9ad83d263aaf.gif">
</p>

<br>

## 👨‍👩‍👧‍👦 Team Members Introduce
- ### 이상수 (팀원)   
- ### 이하윤 (팀원)                  
- ### 방현우 (팀장) 



<br>

## 👩‍💻 구상도

<p align="center">
  <img src="https://user-images.githubusercontent.com/81614803/200114537-fd0d3258-61a4-4f2f-bf24-51c08ce14b6c.png">
  <img src="https://user-images.githubusercontent.com/81614803/200114607-39b4af68-f8e3-445c-b1b4-345148041a90.png">
</p>

 <br>


## 🏡Primary Contents

### 🎨 활용한 Ai 기능 및 정리
</br>

<details>
<summary><b>사용한 Ai 기술들</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200114692-ed648fff-caa2-4f59-8b3c-f55109ebf287.png)
 - **Reference** :pushpin:
   - yolo3 -> tensorflow lite 모델까지 여러 문제점 때문에 여러번의 모델 교체
   - 최종적으로 coral에서 작동하는 edgetpu용 tflite 파일로 변환이 가능한 mobilenet SSD v2 사용
 
</div>
</details>
</br>

<details>
<summary><b>Mobilenet SSD v2 데이터 수집 및 라벨링</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200114958-d44c74b1-56ce-4d55-a5ca-983745693c86.png)
 - **Reference** :pushpin:
   - 뱃지로 사용한 이미지를 다양한 환경에서 이미지 데이터셋을 만듭니다.
   - 이후, 학습에 사용한 데이터로 사용하기 위하여 이미지가 있는 경우 1, 없는 경우 0 으로 데이터 라벨링 작업
 
</div>
</details>
</br>

<details>
<summary><b>모델 학습 과정</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115019-3993a7fb-13d6-44fe-a3fc-5b802ce47f6f.png)
 - **Reference** :pushpin:
   - anaconda prompt를 통해 model의 하이퍼파라미터를 수정해가며 최적의 학습 모델을 찾는다.
   - 약 6천 step 정도에 loss값 0.4 정도의 모델을 사용
 
</div>
</details>
</br>

<details>
<summary><b>coral에서 동작가능한 tflite 파일로 학습모델 변환</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115229-28699166-558f-421b-aecd-5926ab77b0e8.png)
 - **Reference** :pushpin:
   - coral을 사용하지 않을시, fps가 10이하로 낮은 수치로 동작하여 이를 해결하기 위해 coral을 사용
   - coral에서는 tflite모델을 사용하여 속도를 높이기에 변환과정
   - 추가적으로 edgetpu용 8비트 양자화 모델로 변환 
 
</div>
</details>
</br>

<details>
<summary><b>8비트 양자화 tflite파일로 변환된 fps</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115389-82524ad1-5340-4b90-bdb3-ad7ebe2f0cf1.png)
 - **Reference** :pushpin:
   - 기존 tpu 없이 사용한 fps의 경우 5정도가 나오지면 변환 후 20~30 사이의 fps가 유지됨
 
</div>
</details>
</br>


## 📌 MobileNet SSD V2 정리
<br>
<details>
<summary><b>Model Explain</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200116050-e47a0098-b095-43c7-8f0f-fb6a4d8ae6bf.png)
 - **Reference** :pushpin:
   - 특징 맵을 24 -> 144로 팽창시켜주고 이 특징맵을 이용하여 convolution 연산을 수행
   - 이후, 특징 추출을 위해 사용한 layer를 다시 24로 줄여줌으로써 연산량을 감소시켜줌
 
</div>
</details>
</br>

<details>
<summary><b>SSD Explain(Single Shot MultiBox Detector)</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200116369-28b5c54a-0915-476c-a89c-6355a4269ece.png)
 - **Reference** :pushpin:
   - 8x8 특징맵을 작은 물체, 4x4 특징맵을 큰 물체를 검출하는데 사용하여 정확성을 높인다. 
   - 찾은 박스에서 중복되는 박스들을 제거하고 높은 신뢰도가 있는 박스만을 사용
 
</div>
</details>
</br>

## 🍵 시연영상 링크

- https://youtu.be/ZbI0N2Nszs4
