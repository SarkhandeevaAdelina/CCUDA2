#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для построения графиков зависимости времени выполнения от параметра N
для решателя дифракции на ленте
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Настройка шрифтов для поддержки русского языка
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_timing_from_csv(csv_file='timing_data.csv'):
    """
    Построение графиков из CSV файла
    
    Формат CSV:
    N,MatrixTime_ms,SolveTime_ms,TotalTime_ms
    10,8.234,0.156,8.390
    ...
    """
    try:
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1, 
                            names=['N', 'MatrixTime', 'SolveTime', 'TotalTime'])
        
        N = data['N']
        matrix_time = data['MatrixTime']
        solve_time = data['SolveTime']
        total_time = data['TotalTime']
        
    except Exception as e:
        print(f"Ошибка чтения файла {csv_file}: {e}")
        print("\nИспользую демонстрационные данные")
        
        # Примерные данные для демонстрации
        N = np.array([10, 20, 30, 40, 50])
        matrix_time = N**2 * 0.08  # O(N^2) для построения матрицы
        solve_time = N**3 * 0.0001  # O(N^3) для решения СЛАУ
        total_time = matrix_time + solve_time
    
    # Создание фигуры с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Все компоненты времени
    ax1.plot(N, matrix_time, 'o-', linewidth=2, markersize=8, 
             label='Построение матрицы', color='#2E86AB')
    ax1.plot(N, solve_time, 's-', linewidth=2, markersize=8, 
             label='Решение СЛАУ', color='#A23B72')
    ax1.plot(N, total_time, '^-', linewidth=2, markersize=8, 
             label='Общее время', color='#F18F01')
    
    ax1.set_xlabel('Параметр усечения N', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время (мс)', fontsize=12, fontweight='bold')
    ax1.set_title('Зависимость времени выполнения от N', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(N[0] - 2, N[-1] + 2)
    
    # Добавление значений на график
    for i, (n, t) in enumerate(zip(N, total_time)):
        ax1.annotate(f'{t:.1f}', 
                    xy=(n, t), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.7)
    
    # График 2: Доля времени на каждую операцию
    percentages_matrix = (matrix_time / total_time) * 100
    percentages_solve = (solve_time / total_time) * 100
    
    width = 3
    ax2.bar(N - width/2, percentages_matrix, width, 
            label='Построение матрицы', color='#2E86AB', alpha=0.8)
    ax2.bar(N + width/2, percentages_solve, width, 
            label='Решение СЛАУ', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Параметр усечения N', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Доля времени (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Распределение времени по операциям', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, 105)
    ax2.set_xlim(N[0] - 5, N[-1] + 5)
    
    # Добавление процентов на столбцы
    for i, n in enumerate(N):
        ax2.text(n - width/2, percentages_matrix[i] + 2, 
                f'{percentages_matrix[i]:.1f}%',
                ha='center', va='bottom', fontsize=9)
        ax2.text(n + width/2, percentages_solve[i] + 2, 
                f'{percentages_solve[i]:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Сохраняем в текущую директорию
    import os
    output_file = os.path.join(os.getcwd(), 'timing_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен: {output_file}")
    plt.close()
    
    # Дополнительный график: сравнение теоретической и реальной сложности
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    # Нормализация для сравнения
    matrix_normalized = matrix_time / matrix_time[0]
    theoretical_n2 = (N / N[0])**2
    
    solve_normalized = solve_time / solve_time[0]
    theoretical_n3 = (N / N[0])**3
    
    ax3.plot(N, matrix_normalized, 'o-', linewidth=2, markersize=8,
            label='Построение матрицы (реальное)', color='#2E86AB')
    ax3.plot(N, theoretical_n2, '--', linewidth=2, 
            label='Теоретическая сложность O(N²)', color='#2E86AB', alpha=0.5)
    
    ax3.plot(N, solve_normalized, 's-', linewidth=2, markersize=8,
            label='Решение СЛАУ (реальное)', color='#A23B72')
    ax3.plot(N, theoretical_n3, '--', linewidth=2,
            label='Теоретическая сложность O(N³)', color='#A23B72', alpha=0.5)
    
    ax3.set_xlabel('Параметр усечения N', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Нормализованное время (относительно N=10)', 
                   fontsize=12, fontweight='bold')
    ax3.set_title('Сравнение с теоретической сложностью алгоритмов', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(N[0] - 2, N[-1] + 2)
    
    plt.tight_layout()
    
    output_file = os.path.join(os.getcwd(), 'complexity_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_file}")
    plt.close()
    
    # Вывод статистики
    print("\nСтатистика времени выполнения")
    print(f"\n{'N':<8} {'Матрица (мс)':<15} {'СЛАУ (мс)':<15} {'Всего (мс)':<15}")
    for i in range(len(N)):
        print(f"{int(N[i]):<8} {matrix_time[i]:<15.3f} {solve_time[i]:<15.3f} {total_time[i]:<15.3f}")
    
    # Анализ роста времени
    if len(N) > 1:
        print("\nАнализ роста времени:")
        for i in range(1, len(N)):
            ratio_n = N[i] / N[i-1]
            ratio_matrix = matrix_time[i] / matrix_time[i-1]
            ratio_solve = solve_time[i] / solve_time[i-1]
            ratio_total = total_time[i] / total_time[i-1]
            
            print(f"N: {int(N[i-1])} -> {int(N[i])} (x{ratio_n:.1f})")
            print(f"  Матрица: x{ratio_matrix:.2f} (ожидается x{ratio_n**2:.2f} для O(N^2))")
            print(f"  СЛАУ:    x{ratio_solve:.2f} (ожидается x{ratio_n**3:.2f} для O(N^3))")
            print(f"  Всего:   x{ratio_total:.2f}")
            print()

def plot_timing_manual(N, matrix_time, solve_time):
    """
    Построение графиков из массивов данных
    
    Parameters:
    -----------
    N : array-like
        Значения параметра N
    matrix_time : array-like
        Время построения матрицы (мс)
    solve_time : array-like
        Время решения СЛАУ (мс)
    """
    N = np.array(N)
    matrix_time = np.array(matrix_time)
    solve_time = np.array(solve_time)
    total_time = matrix_time + solve_time
    
    # Создание временного CSV файла
    with open('timing_data_temp.csv', 'w') as f:
        f.write('N,MatrixTime_ms,SolveTime_ms,TotalTime_ms\n')
        for i in range(len(N)):
            f.write(f'{N[i]},{matrix_time[i]},{solve_time[i]},{total_time[i]}\n')
    
    plot_timing_from_csv('timing_data_temp.csv')

if __name__ == '__main__':
    import sys
    
    print("\nПостроение графиков")
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Чтение данных: {csv_file}")
        plot_timing_from_csv(csv_file)
    else:
        print("Чтение данных: timing_data.csv")
        plot_timing_from_csv('timing_data.csv')
    
    print("\nГотово")

