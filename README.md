## ๐ ๋ณดํ ์ฅ์ ์ธ์ ์ํ ์์จ์ฃผํ ์บ๋ฆฌ์ด
> ### ํผ์์ ๊ฑฐ๋์ด ํ๋  ๋ถ๋ค์ ์ํด ๋ฑ์ง๋ฅผ ํตํด ์ค์ค๋ก ๋ณดํ์๋ฅผ ๋ฐ๋ผ์ค๋ ์์จ ์ฃผํ ์บ๋ฆฌ์ด


![](https://user-images.githubusercontent.com/81614803/200113921-a4debf27-a91f-4074-ace0-d1b90a928b27.jpg)

</br></br>

### ๊ฐ๋ฐ๊ธฐ๊ฐ: 2021.03.01 - 2021.06.13

</br></br>

## ๐About

### - ๋ณดํ ์ฅ์ ์ธ์ ๊ฒฝ์ฐ ์ฌํ์ด๋ ์ถ์ฅ ๋ฑ ๋จผ๊ณณ์ ๊ฐ์ผํ  ๋, ํ ์ฒด์ด์ ์๊ฑฐ๋ ๋ชฉ๋ฐ์ ์ง์ ์ฌ๋๋ค์ ํผ์ ์บ๋ฆฌ์ด๋ฅผ ๋ค๊ณ ๊ฐ๊ธฐ ์ด๋ ค์ด ์ํฉ์ด ๋ง์ต๋๋ค. 
### - ๋ฐ๋ผ์, ์ด๋ฌํ ๋ถ๋ค์ด ๋จผ๊ณณ์ ํผ์ ๋ค๋์ฌ ์ ์๊ฒ ์ฌ์ฉ์๋ฅผ ๋ฐ๋ผ ๋ค๋๋ ์์จ์ฃผํ ์บ๋ฆฌ์ด๋ฅผ ๊ฐ๋ฐํ์์ต๋๋ค. 
<p>
  <img src="https://user-images.githubusercontent.com/81614803/200116843-2102afe9-0f18-4364-bf47-9ad83d263aaf.gif">
</p>

<br>

## ๐จโ๐ฉโ๐งโ๐ฆ Team Members Introduce
- ### ์ด์์ (ํ์)   
- ### ์ดํ์ค (ํ์)                  
- ### ๋ฐฉํ์ฐ (ํ์ฅ) 



<br>

## ๐ฉโ๐ป ๊ตฌ์๋

<p align="center">
  <img src="https://user-images.githubusercontent.com/81614803/200114537-fd0d3258-61a4-4f2f-bf24-51c08ce14b6c.png">
  <img src="https://user-images.githubusercontent.com/81614803/200114607-39b4af68-f8e3-445c-b1b4-345148041a90.png">
</p>

 <br>


## ๐กPrimary Contents

### ๐จ ํ์ฉํ Ai ๊ธฐ๋ฅ ๋ฐ ์ ๋ฆฌ
</br>

<details>
<summary><b>์ฌ์ฉํ Ai ๊ธฐ์ ๋ค</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200114692-ed648fff-caa2-4f59-8b3c-f55109ebf287.png)
 - **Reference** :pushpin:
   - yolo3 -> tensorflow lite ๋ชจ๋ธ๊น์ง ์ฌ๋ฌ ๋ฌธ์ ์  ๋๋ฌธ์ ์ฌ๋ฌ๋ฒ์ ๋ชจ๋ธ ๊ต์ฒด
   - ์ต์ข์ ์ผ๋ก coral์์ ์๋ํ๋ edgetpu์ฉ tflite ํ์ผ๋ก ๋ณํ์ด ๊ฐ๋ฅํ mobilenet SSD v2 ์ฌ์ฉ
 
</div>
</details>
</br>

<details>
<summary><b>Mobilenet SSD v2 ๋ฐ์ดํฐ ์์ง ๋ฐ ๋ผ๋ฒจ๋ง</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200114958-d44c74b1-56ce-4d55-a5ca-983745693c86.png)
 - **Reference** :pushpin:
   - ๋ฑ์ง๋ก ์ฌ์ฉํ ์ด๋ฏธ์ง๋ฅผ ๋ค์ํ ํ๊ฒฝ์์ ์ด๋ฏธ์ง ๋ฐ์ดํฐ์์ ๋ง๋ญ๋๋ค.
   - ์ดํ, ํ์ต์ ์ฌ์ฉํ ๋ฐ์ดํฐ๋ก ์ฌ์ฉํ๊ธฐ ์ํ์ฌ ์ด๋ฏธ์ง๊ฐ ์๋ ๊ฒฝ์ฐ 1, ์๋ ๊ฒฝ์ฐ 0 ์ผ๋ก ๋ฐ์ดํฐ ๋ผ๋ฒจ๋ง ์์
 
</div>
</details>
</br>

<details>
<summary><b>๋ชจ๋ธ ํ์ต ๊ณผ์ </b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115019-3993a7fb-13d6-44fe-a3fc-5b802ce47f6f.png)
 - **Reference** :pushpin:
   - anaconda prompt๋ฅผ ํตํด model์ ํ์ดํผํ๋ผ๋ฏธํฐ๋ฅผ ์์ ํด๊ฐ๋ฉฐ ์ต์ ์ ํ์ต ๋ชจ๋ธ์ ์ฐพ๋๋ค.
   - ์ฝ 6์ฒ step ์ ๋์ loss๊ฐ 0.4 ์ ๋์ ๋ชจ๋ธ์ ์ฌ์ฉ
 
</div>
</details>
</br>

<details>
<summary><b>coral์์ ๋์๊ฐ๋ฅํ tflite ํ์ผ๋ก ํ์ต๋ชจ๋ธ ๋ณํ</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115229-28699166-558f-421b-aecd-5926ab77b0e8.png)
 - **Reference** :pushpin:
   - coral์ ์ฌ์ฉํ์ง ์์์, fps๊ฐ 10์ดํ๋ก ๋ฎ์ ์์น๋ก ๋์ํ์ฌ ์ด๋ฅผ ํด๊ฒฐํ๊ธฐ ์ํด coral์ ์ฌ์ฉ
   - coral์์๋ tflite๋ชจ๋ธ์ ์ฌ์ฉํ์ฌ ์๋๋ฅผ ๋์ด๊ธฐ์ ๋ณํ๊ณผ์ 
   - ์ถ๊ฐ์ ์ผ๋ก edgetpu์ฉ 8๋นํธ ์์ํ ๋ชจ๋ธ๋ก ๋ณํ 
 
</div>
</details>
</br>

<details>
<summary><b>8๋นํธ ์์ํ tfliteํ์ผ๋ก ๋ณํ๋ fps</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200115389-82524ad1-5340-4b90-bdb3-ad7ebe2f0cf1.png)
 - **Reference** :pushpin:
   - ๊ธฐ์กด tpu ์์ด ์ฌ์ฉํ fps์ ๊ฒฝ์ฐ 5์ ๋๊ฐ ๋์ค์ง๋ฉด ๋ณํ ํ 20~30 ์ฌ์ด์ fps๊ฐ ์ ์ง๋จ
 
</div>
</details>
</br>


## ๐ MobileNet SSD V2 ์ ๋ฆฌ
<br>
<details>
<summary><b>Model Explain</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200116050-e47a0098-b095-43c7-8f0f-fb6a4d8ae6bf.png)
 - **Reference** :pushpin:
   - ํน์ง ๋งต์ 24 -> 144๋ก ํฝ์ฐฝ์์ผ์ฃผ๊ณ  ์ด ํน์ง๋งต์ ์ด์ฉํ์ฌ convolution ์ฐ์ฐ์ ์ํ
   - ์ดํ, ํน์ง ์ถ์ถ์ ์ํด ์ฌ์ฉํ layer๋ฅผ ๋ค์ 24๋ก ์ค์ฌ์ค์ผ๋ก์จ ์ฐ์ฐ๋์ ๊ฐ์์์ผ์ค
 
</div>
</details>
</br>

<details>
<summary><b>SSD Explain(Single Shot MultiBox Detector)</b></summary>
<div markdown="1">
 
![](https://user-images.githubusercontent.com/81614803/200116369-28b5c54a-0915-476c-a89c-6355a4269ece.png)
 - **Reference** :pushpin:
   - 8x8 ํน์ง๋งต์ ์์ ๋ฌผ์ฒด, 4x4 ํน์ง๋งต์ ํฐ ๋ฌผ์ฒด๋ฅผ ๊ฒ์ถํ๋๋ฐ ์ฌ์ฉํ์ฌ ์ ํ์ฑ์ ๋์ธ๋ค. 
   - ์ฐพ์ ๋ฐ์ค์์ ์ค๋ณต๋๋ ๋ฐ์ค๋ค์ ์ ๊ฑฐํ๊ณ  ๋์ ์ ๋ขฐ๋๊ฐ ์๋ ๋ฐ์ค๋ง์ ์ฌ์ฉ
 
</div>
</details>
</br>

## ๐ต ์์ฐ์์ ๋งํฌ

- https://youtu.be/ZbI0N2Nszs4
