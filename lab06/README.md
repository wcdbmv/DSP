# Лабораторная работа № 6. Изучение фильтра Винера

1. Исходные данные — сигнал Гаусса
2. Помеха 2-х видов — импульсная и Гаусса (см. лабораторные работы 4, 5)
3. Фильтрация выполняется с помощью фильтра Винера с передаточной функцией, описываемой по следующей формуле

   Hk = (|Vk|² - |βk|²) / |Vk|²

   где Vk — отсчёты спектра исходного искажённого сигнала, βk — отсчёты спектра помехи, k = 0, 1, 2, ..., N - 1

   Vk = Vk^{(0)} + βk

   Vk^{(0)} — спектр идеалього неискажённого сигнала.

Замечание: отдельно рассмотреть сигнал, искажённый импульсной помехой, и отдельно — Гауссовой помехой.