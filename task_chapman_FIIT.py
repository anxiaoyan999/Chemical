"""
Задание СРР №2, Часть Б — Трек ФИИТ (инженерия кода)
Модель стратосферного цикла Чепмена с каталитическим разрушением озона хлором.

ЦЕЛЬ:
  1. Реализовать правую часть системы ОДУ  chapman_rhs(t, y)
  2. Проверить закон сохранения [Cl] + [ClO] = const как ЮНИТ-ТЕСТ
  3. Проинтегрировать методом Radau и сравнить озон без хлора и с хлором

Полное описание и чеклист — на странице задания в Moodle (ТУИС).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt  # 添加了绘图库

# ─── Константы модели (менять не нужно) ───────────────────────────────────────
J1   = 3e-12    # фотолиз O2 -> 2O        (1/с)
k2   = 6e-34    # O + O2 + M -> O3 + M    (см⁶/(молекул²·с))
J3   = 5e-4     # O3 + hν -> O2 + O       (1/с)
k4   = 8e-15    # O + O3 -> 2 O2          (см³/(молекул·с))
kCl  = 1.2e-11  # Cl + O3 -> ClO + O2     (см³/(молекул·с))
kClO = 3.8e-11  # ClO + O -> Cl + O2      (см³/(молекул·с))
O2   = 3.7e17   # молекул/см³
M    = 5.0e17   # молекул/см³


def chapman_rhs(t, y):
    """
    Правая часть системы ОДУ цикла Чепмена + хлорный катализ.
    y = [O, O3, Cl, ClO]  (концентрации, молекул/см³)
    """
    O, O3, Cl, ClO = y
    
    # TODO 1: Записываем четыре производные согласно подсказкам
    # Образование O: +2*J1*O2 (фотолиз O2) + J3*O3 (фотолиз O3)
    # Гибель O: -k2*O*O2*M, -k4*O*O3, -kClO*ClO*O
    dO   = 2 * J1 * O2 + J3 * O3 - k2 * O * O2 * M - k4 * O * O3 - kClO * ClO * O
    
    # Образование O3: +k2*O*O2*M
    # Гибель O3: -J3*O3, -k4*O*O3, -kCl*Cl*O3
    dO3  = k2 * O * O2 * M - J3 * O3 - k4 * O * O3 - kCl * Cl * O3
    
    # Хлор: dCl = -kCl*Cl*O3 + kClO*ClO*O
    dCl  = -kCl * Cl * O3 + kClO * ClO * O
    
    # Оксид хлора: dClO = +kCl*Cl*O3 - kClO*ClO*O
    dClO = kCl * Cl * O3 - kClO * ClO * O
    
    return [dO, dO3, dCl, dClO]


def test_chlorine_invariant():
    """
    ЮНИТ-ТЕСТ: хлор — катализатор, он не расходуется.
    Значит d[Cl]/dt + d[ClO]/dt должно быть строго равно нулю
    в любой точке фазового пространства.
    """
    y_test = [1e7, 4e12, 5e8, 5e8]     # произвольное состояние
    dydt = chapman_rhs(0.0, y_test)
    residual = dydt[2] + dydt[3]        # dCl + dClO
    assert abs(residual) < 1e-6, f"Инвариант нарушен: dCl+dClO = {residual}"
    print(f"[OK] Инвариант [Cl]+[ClO]=const выполняется (невязка {residual:.2e})")


def main():
    # 1. Сначала прогоняем юнит-тест
    test_chlorine_invariant()

    # 2. Интегрируем 30 суток: без хлора и с хлором
    t_end = 30 * 86400.0
    O3_results = {}
    
    # Создаем фигуру для графика
    plt.figure(figsize=(10, 6))
    
    for Cl0 in (0.0, 1e9):
        y0 = [1e7, 4e12, Cl0, 0.0]          # [O, O3, Cl, ClO]
        # TODO 2: добавляем dense_output=True для построения графика
        sol = solve_ivp(chapman_rhs, [0, t_end], y0, method='Radau', dense_output=True)
        O3_results[Cl0] = sol.y[1, -1]
        print(f"[Cl]0 = {Cl0:.0e} | O3_финал = {sol.y[1,-1]:.3e} | шагов: {sol.t.size}")
        
        # Строим кривую [O3](t). Переводим время в дни для удобства.
        plt.plot(sol.t / 86400.0, sol.y[1], label=f'[Cl]0 = {Cl0:.0e}')

    # 3. Относительное падение озона (ожидается ~30%)
    drop = (1 - O3_results[1e9] / O3_results[0.0]) * 100
    print(f"Падение [O3] из-за хлора: {drop:.1f} %")

    # Настройка и показ графика
    plt.xlabel('Время (дни)')
    plt.ylabel('Концентрация O3 (молекул/см³)')
    plt.title('Динамика озона: без хлора и с хлором')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()