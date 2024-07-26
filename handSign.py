#Deteksi bahasa tangan American Sign Language dengan python dan data training

#import modul cv2 digunakan untuk pengelolaan citra digital

import cv2

#import modul cvzone untuk deteksi gerakan tangan

from cvzone.HandTrackingModule import HandDetector

#import modul numpy digunakan untuk melakukah hitungan himpunan matriks dan kalkulasi numerik 

import numpy as np

#modul math digunakna untuk perhitungan matematis seperti menghitung kumpulan metode dan konstanta
import math

#modul untuk melacak tanggal dan waktu pada sebuah mesin
import time

#nilai yang berisi variable dengan nama cap lalu dengan menggunakan modul cv2 kita melakukan capture dan menampilkan video
cap = cv2.VideoCapture(0)

#kita membatasi deteksi pada tangan hanya 1 tangan saja
detector = HandDetector(maxHands=1)

#menentukan nilai offset dengan default value 20
offset = 20

#variable image yang berisikan nilai 300 untuk default gambar yang bakal di tampilkan
imgSize = 300

#kita akan memasukkan data training di dalam folder data 
folder = "Data/Y"

#counter di set dari 0 karena akan dilakukan perulangan dimana perulangan akan dimulai dari 0

#perulangan ini akan dilakukan untuk memasukkan data training ke dalam folder
counter = 0

#lakukan perulangan dimana jika benar
while True:

    #defenisikan nilai success dan img lalu isikan nilai variable cap untuk membaca video capture
    success, img = cap.read()

    #defenisikan variable hands dan img untuk mendeteksi tangan  lalu di capture dan nanti bakal di masukkan ke dalam folder
    hands, img = detector.findHands(img)

    #jika variable hands berisikan value
    if hands:

        #maka variable hands kita isikan nilai sebuah array yang berisikan 0 
        hand = hands[0]

        #variable yang berisikan value berisi informasi yang akan kita dapatkan pada saat membounding kotak pada gambar
        x, y, w, h = hand['bbox']

        #ini kita meng crop/memotong gambar img white 

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        #img crop berisikan value yang akan di crop nilai offset yang sudah di set sebesar 20
        #nilai variable y di kurang dengan nilai offset 20 lalu di bagi dengan nilai y itu sendiri dst
        #dan terakhir ditambah dengan nilai offset 
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        #proses cropping image
        imgCropShape = imgCrop.shape

        #menentukan aspek rasio pada gambar
        aspectRatio = h/w

        ##################################

        #proses percabangan untuk menentukan jika aspek rasionya lebih dari 1
        if aspectRatio > 1:

            #defenisikan variable k yang isinya cropping image size dibagi dengan variable h hasil bbox gambar

            k = imgSize/h

            #math ceil berfungsi untuk mengembalikan nilai integer terkecil yang lebih besar. Variable k yang isi nya sumbu x lalu di kali dengan 
            #sumbu y pada sebuah koordinat kartesius
            wCal = math.ceil(k*w)

            #proses mengubah gambar yang akan di crop 
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            #mengubah tepian siku pada gambar dan diresize 
            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize-wCal)/2)

            imgWhite[:, wGap:wCal+wGap] = imgResize

        #jika aspek rasio kurang dari 0

        else:

            # variable img size dibgai variable w 
            k = imgSize/w

            #math ceil berfungsi untuk mengembalikan nilai integer terkecil yang lebih besar. Variable k yang isi nya sumbu x lalu di kali dengan 
            #sumbu y pada sebuah koordinat kartesius
            hCal = math.ceil(k * h)
            #proses mengubah gambar yang akan di crop
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            #mengubah tepian siku pada gambar dan diresize
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize-hCal)/2)

            imgWhite[hGap:hCal+hGap,:] = imgResize
             
        #modul cv2 untuk menampilkan gambar yang telah di cropping
        cv2.imshow("ImageCrop", imgCrop)
        #modul cv2 unutk menampilkan gambar utuh
        cv2.imshow("ImageWhite", imgWhite)

    #cv2 image show 

    cv2.imshow("Image", img)

    #cv2 waitkey jika ada proses menekan tombol pada keyboard 
    key = cv2.waitKey(1)

    #jika menekan tombol s pada keyboard
    if key == ord("s"):
        
        #maka counter 0 tadi akan di loop dan dilakukan perulangan +1
        counter += 1

        #ini adalah code untuk meletekkan dan memberikan nama file ketika proses cropping ke dalam folder yang telah di tentukan
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)

        #menampilkan counter +1 pada looping
        print(counter)  