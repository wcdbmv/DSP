# Лабораторная работа № 3. Частотный алгоритм вычисления свёртки двух функций

Вычислить свёртку следующих функций:
1. rect(x) и rect(x)
2. rect(x) и функция Гаусса
3. функция Гаусса и функция Гаусса

Результат вычислений отобразить в графиках

Для вычисления свёртки использовать частотный алгоритм, представленный ниже.
Этапы алгоритма:
1. Дополнить исходную функцию нулями следующим образом:

   \tilde U(n) = {U(n) [0 <= n <= N - 1], 0 [N <= n <= 2N - 1]}.
2. Дополнить нулями и вторую функцию аналогичным образом.
3. Вычислить БПФ от каждой функции — получим 2 спектра \tile V1(k) и \tilde V2(k).
4. Вычислить произведение

   \tilde W(k) = \tilde V1(k) * \tilde V2(k).
5. Вычислить обратное БПФ от функции \tilde W(k) (есть в Matklab).

Всего должно быть 3 графика.
