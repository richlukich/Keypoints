# Human Pose Estimation

## Введение

В данном репозиториии представлена реализация нейросети для решение задачи Human Pose Estimation. Learning Feature Pyramids for Human Pose Estimation: https://arxiv.org/pdf/1708.01101.pdf . Также проделаны эксперименты с разными подходами обработки тепловых карт (heatmaps).

## Нейросеть
Данная нейросеть выглядит следующим образом:
<p>--Тут должно быть фото нейросети--
<p>Как видно нейросеть основана на PRM блоках. Они бывают различных видов PRM-A,PRM-B,PRM-C,PRM-D. В данном репозитории использовался PRM-A.
<p>--Тут должно быть фото PRM--
<p>  То есть пусть у нас входные данные имеют вид 
![first equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7Ba%7D%7Bb%7D)

