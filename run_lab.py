import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
import statistics

sizes = [200, 400, 800, 1200, 1600, 2000]
blocks_configs = [(8,8), (16,8), (16,16), (32,8), (32,16), (32,32)]
repeats = 3
results = {f"{x}x{y}": [] for x, y in blocks_configs}

exe_name = "CudaLab4.exe" 

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

    for x, y in blocks_configs:
        config_name = f"{x}x{y}"
        times_run = []

        for r in range(repeats):
            result = subprocess.run(
                [exe_name, str(N), "a.txt", "b.txt", "c.txt", str(x), str(y)], 
                capture_output=True, text=True
            )
            
            try:
                time_ms = float(result.stdout.strip().split('\n')[-1])
                times_run.append(time_ms)
            except ValueError:
                print(f"ОШИБКА: C++ программа упала или вернула не число. Вывод программы:")
                print(result.stderr)
                exit(1)
                
        C_cpp = np.loadtxt("c.txt", dtype=np.int32)
        if not np.array_equal(C_cpp, C_py):
            print("Результаты не совпадают.")
            exit(1)
            
        median_time = statistics.median(times_run)
        results[config_name].append(median_time)
        print(f"Блок {config_name:7} | Потоков: {x*y:<4} | Медиана: {median_time:.4f} ms")

plt.figure(figsize=(10, 6))
for config in results:
    plt.plot(sizes, results[config], marker='o', linestyle='-', label=f'Блок {config}')

plt.title('Зависимость времени выполнения CUDA от размера матрицы и сетки блоков')
plt.xlabel('Размер матрицы (N)')
plt.ylabel('Время выполнения (мс)')
plt.legend()
plt.grid(True)

header = "| Сетка блоков | " + " | ".join([f"{s}x{s}" for s in sizes]) + " |"
print(header)
print("|" + "-" * 14 + "|" + "|".join(["-" * (len(str(s))*2+1) for s in sizes]) + "|")

for x, y in blocks_configs:
    config_name = f"{x}x{y} ({x*y})"
    row = f"| {config_name:12} | " + " | ".join([f"{t:.4f}" for t in results[f"{x}x{y}"]]) + " |"
    print(row)

plt.show()