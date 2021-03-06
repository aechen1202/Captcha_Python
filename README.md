# Captcha_Python
## 基本說明
利用Tensorflow Keras模組CNN卷積學習辨識Captcha驗證碼圖片 

## 系統流程
0.Captcha_MLP.py:多層感知機(辨識成功率70%) , Captcha_CNN.py:CNN卷積學習(辨識成功率100%) </br>
1.先將產生出的Captcha訓練與驗證圖片放入本機資料夾內 , 產生圖片程式可參考以下連結: </br>
https://github.com/aechen1202/CaptchaApp </br>
2.訓練圖片0 ~ 8 大小50x50各1000張 , 9大小50x50 3000張 </br>
3.驗證圖片0 ~ 9 大小50x50各100張 </br>
4.替換程式訓練(C:\CaptchaImg\Train)與驗證圖片(C:\CaptchaImg\Test)路徑 </br>
5.讀取訓練驗證圖片與資料轉換 </br>
6.建立CNN卷積學習模型 </br>
7.利用訓練資料訓練模型 , 發現1輪後開始快速收斂</br>
8.利用另外的驗證圖片0 ~ 9 各100張驗證訓練結果 , 發現正確率100% </br>
9.抽出每個數字的前3筆資料顯現查看資料 </br>

## 總結
此為類似Mnist的簡單程式 , 只是看完Mnist自己動手做的範例 , 網路上應該也很多類似的範例 , 但是我是在完全沒參考網路素材下自己研究出來 , 在過程中發現9時常會辨識成0 , 而8時常辨識成3 , 所以我不斷的反覆嘗試把某些數字訓練資料拿掉或加回來 , 最後發現只要9的訓練資料多兩倍則可非常成功 , 百思不得其解為甚麼加強9的訓練資料會連帶修正8跟9的辨識錯誤 , 直到後來才了解原來是加重8與9的中間那一撇連接處 , 光是試出這結果就花了我兩週時間 , 所以我想不斷的嘗試錯誤與推論假設正是機器學習人員需要長時間面對的狀況與苦悶吧

## 結果圖片
![image](https://raw.githubusercontent.com/aechen1202/Captcha_Python/master/ResultImage/1539335614612.jpg) </br>
![image](https://raw.githubusercontent.com/aechen1202/Captcha_Python/master/ResultImage/1539335645036.jpg) </br>
![image](https://raw.githubusercontent.com/aechen1202/Captcha_Python/master/ResultImage/1539335701387.jpg) </br>
![image](https://raw.githubusercontent.com/aechen1202/Captcha_Python/master/ResultImage/1539335770510.jpg) </br>

## 參考書籍
TensorFlow+Keras深度學習人工智慧實務應用: </br>
https://www.books.com.tw/products/0010754327




