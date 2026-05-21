import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os

sizes = [200, 400, 800, 1200, 1600, 2000]
threads_list = [1, 2, 4, 8, 16]
results = {t: [] for t in threads_list}

exe_name = "Lab1.exe" 

if not os.path.exists(exe_name):
    print(f"ОШИБКА: Файл {exe_name} не найден")
    exit(1)

for N in sizes:
    print(f"Обработка матрицы размером {N}x{N}...")
    
    A = np.random.randint(0, 10, size=(N, N), dtype=np.int32)
    B = np.random.randint(0, 10, size=(N, N), dtype=np.int32)
    np.savetxt("a.txt", A, fmt='%d')
    np.savetxt("b.txt", B, fmt='%d')

    C_py = A @ B

    for threads in threads_list:
        print(f"Вычисление на {threads} потоке(ах)...", end=" ")
        
        result = subprocess.run(
            [exe_name, str(N), "a.txt", "b.txt", "c.txt", str(threads)], 
            capture_output=True, text=True
        )
        
        try:
            time_ms = int(result.stdout.strip())
            results[threads].append(time_ms)
        except ValueError:
            print(f"\nОШИБКА: C++ программа упала или вернула не число. Вывод программы:\n{result.stderr}")
            exit(1)

        C_cpp = np.loadtxt("c.txt", dtype=np.int32)
        
        if np.array_equal(C_cpp, C_py):
            print(f"Время выполнения C++: {time_ms} мс")
        else:
            print("Результаты не совпадают.")
            exit(1)

plt.figure(figsize=(10, 6))
for threads in threads_list:
    plt.plot(sizes, results[threads], marker='o', linestyle='-', label=f'{threads} потоков')

plt.title('Зависимость времени выполнения от размера матрицы и числа потоков (OpenMP)')
plt.xlabel('Размер матрицы (N)')
plt.ylabel('Время выполнения (мс)')
plt.legend()
plt.grid(True)

header = "| Потоки \\ Размер | " + " | ".join([str(s) for s in sizes]) + " |"
print(header)
print("|" + "-" * 17 + "|" + "|".join(["-" * len(str(s)) for s in sizes]) + "|")

for threads in threads_list:
    row = f"| {threads:14} | " + " | ".join([str(t) for t in results[threads]]) + " |"
    print(row)

plt.show()