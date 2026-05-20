import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os

sizes = [200, 400, 800, 1200, 1600, 2000]
times = []

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

    result = subprocess.run(
        [exe_name, str(N), "a.txt", "b.txt", "c.txt"], 
        capture_output=True, text=True
    )
    
    try:
        time_ms = int(result.stdout.strip())
        times.append(time_ms)
    except ValueError:
        print("ОШИБКА: C++ программа упала или вернула не число. Вывод программы:")
        print(result.stderr)
        break

    C_cpp = np.loadtxt("c.txt", dtype=np.int32)
    C_py = A @ B
    
    if np.array_equal(C_cpp, C_py):
        print(f"Время выполнения C++: {time_ms} мс")
    else:
        print("Результаты не совпадают.")

plt.figure(figsize=(8, 5))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.title('Зависимость времени последовательного умножения от размера матрицы')
plt.xlabel('Размер матрицы (N)')
plt.ylabel('Время выполнения (мс)')
plt.grid(True)

print("| Размер матрицы (N) | Время (мс) |")
print("|--------------------|------------|")
for s, t in zip(sizes, times):
    print(f"| {s}x{s} | {t} |")

plt.show()
