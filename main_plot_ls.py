import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_solution_for_t_0(solution1, solution2, solution3, node_index, time_nodes, y_label):
    mpl.rcParams["mathtext.fontset"] = 'cm'
    plt.figure(figsize=(10, 6))

    plt.plot(time_nodes, solution1[node_index, :], linewidth=1.5, color='orange',
             label=r'$ t_0=10^{-11}c$')
    plt.plot(time_nodes, solution2[node_index, :], linewidth=1.5, color='purple',
             label=r'$t_0=10^{-12}c$')
    plt.plot(time_nodes, solution3[node_index, :], linewidth=1.5, color='green',
             label=r'$ t_0=10^{-14}c$')

    plt.xlabel(r'Час $t$, с', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    x_axis = plt.gca().xaxis
    x_axis.get_offset_text().set_fontsize(12)

    plt.grid(True, color='lightgray')
    plt.legend(loc='upper left', fontsize=12)

    plt.tight_layout()

    plt.show()


def plot_solution(solution, node_indices, time_nodes, node_indices_labels, y_label,
                  legend_location):
    mpl.rcParams["mathtext.fontset"] = 'cm'
    plt.figure(figsize=(10, 6))

    graphics_count = len(node_indices)

    if graphics_count >= 1:
        plt.plot(time_nodes, solution[node_indices[0], :], linewidth=1.5, label=node_indices_labels[0],
                 markevery=256, color='brown')
    if graphics_count >= 2:
        plt.plot(time_nodes, solution[node_indices[1], :], linewidth=1.5, label=node_indices_labels[1],
                 markevery=256, color='pink')
    if graphics_count >= 3:
        plt.plot(time_nodes, solution[node_indices[2], :], linewidth=1.5, label=node_indices_labels[2],
                 markevery=256, color='green')
    if graphics_count >= 4:
        plt.plot(time_nodes, solution[node_indices[3], :], linewidth=1.5, label=node_indices_labels[3],
                 markevery=256, color='purple')
    if graphics_count >= 5:
        plt.plot(time_nodes, solution[node_indices[4], :], linewidth=1.5, label=node_indices_labels[4],
                 markevery=256, color='orange')

    plt.xlabel(r'Час $t$, с', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    x_axis = plt.gca().xaxis
    x_axis.get_offset_text().set_fontsize(12)

    plt.grid(True, color='lightgray')
    plt.legend(loc=legend_location, fontsize=12)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    L = 1e-8
    T = 1.12e-11
    n = 128
    nt = 1200
    t = np.linspace(0, T, nt + 1)

    result_folder = "results"

    theta1 = np.loadtxt(result_folder + "/10_11/theta.txt")
    theta2 = np.loadtxt(result_folder + "/10_12/theta.txt")
    theta3 = np.loadtxt(result_folder + "/10_14/theta.txt")

    node_index = n // 10
    plot_solution_for_t_0(theta1, theta2, theta3, node_index, t,
                          r'Приріст температури $\theta$, K')

    node_index = 3 * (n // 10)
    plot_solution_for_t_0(theta1, theta2, theta3, node_index, t,
                          r'Приріст температури $\theta$, K')

    sigma1 = np.loadtxt(result_folder + "/10_11/mechanical_stress.txt")
    sigma2 = np.loadtxt(result_folder + "/10_12/mechanical_stress.txt")
    sigma3 = np.loadtxt(result_folder + "/10_14/mechanical_stress.txt")

    node_index = 3 * (n // 10)
    plot_solution_for_t_0(sigma1, sigma2, sigma3, node_index, t,
                          r'Механічне напруження $\sigma$, Pa')

    theta_node_indices = [n // 10, 2 * (n // 10), 4 * (n // 10), 6 * (n // 10)]
    plot_solution(theta2, theta_node_indices, t,
                  [r'$x=0.1L$', r'$x=0.2L$', r'$x=0.4L$', r'$x=0.6L$'],
                  r'Приріст температури $\theta$, K',
                  'upper left')

    q = np.loadtxt(result_folder + "/10_12/q.txt")
    q_node_indices = [0, n // 10, 2 * (n // 10), 3 * (n // 10), 4 * (n // 10)]
    plot_solution(q, q_node_indices, t,
                  [r'$x=0$', r'$x=0.1L$', r'$x=0.2L$', r'$x=0.3L$', r'$x=0.4L$'],
                  r'Тепловий потік $q$, $J \cdot м^{-2} \cdot с^{-1}$', 'best')

    electric_field = np.loadtxt(result_folder + "/10_12/electric_field.txt")
    electric_field_node_indices = [0, n // 10, 2 * (n // 10), 3 * (n // 10)]
    plot_solution(electric_field, electric_field_node_indices, t,
                  [r'$x=0$', r'$x=0.1L$', r'$x=0.2L$', r'$x=0.3L$'],
                  r'Елекричне поле $E$, $V/м$', 'best')

    u = np.loadtxt(result_folder + "/10_12/u.txt")
    u_node_indices = [0, n // 10, 2 * (n // 10), 3 * (n // 10)]
    plot_solution(u, u_node_indices, t, [r'$x=0$', r'$x=0.1L$', r'$x=0.2L$', r'$x=0.3L$'],
                  r'Механічне зміщення $u$, $м$', 'best')

    sigma_node_indices = [2 * (n // 10), 4 * (n // 10), 6 * (n // 10), 8 * (n // 10)]
    plot_solution(sigma2, sigma_node_indices, t, [r'$x=0.2L$', r'$x=0.4L$', r'$x=0.6L$', r'$x=0.8L$'],
                  r'Механічне напруження $\sigma$, Pa', 'upper left')
