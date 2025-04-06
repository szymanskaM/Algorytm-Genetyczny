import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

# Globalna lista przechowująca aktualne trójkąty
area=0
a=0
b=0
generation_data = []  # Lista przechowująca dane dla kolejnych generacji (populacje)
current_generation_index = 0  # Index aktualnie wyświetlanej generacji
generation_times=[]

# Funkcja generująca trójkąty równoboczne
def generate_equilateral_triangle(triangles):
    max_attempts = 200
    while max_attempts > 0:
        max_attempts -= 1
        # Losowy środek ciężkości
        cx, cy = random.uniform(-a, a), random.uniform(-b, b)
        # Losowy kąt odchylenia
        angle = random.uniform(0, 2 * math.pi)
        if is_inside_ellipse(cx,cy,angle) and not check_overlap(cx,cy,angle, triangles):
            return (cx,cy,angle)
    return None

def calculate_triangle(cx,cy,theta):
    # Calculate triangle vertices
    bok = math.sqrt((4 * area) / math.sqrt(3))
    x1 = cx + bok * math.cos(theta) / math.sqrt(3)
    y1 = cy + bok * math.sin(theta) / math.sqrt(3)
    x2 = cx + bok * math.cos(theta + 2 * math.pi / 3) / math.sqrt(3)
    y2 = cy + bok * math.sin(theta + 2 * math.pi / 3) / math.sqrt(3)
    x3 = cx + bok * math.cos(theta + 4 * math.pi / 3) / math.sqrt(3)
    y3 = cy + bok * math.sin(theta + 4 * math.pi / 3) / math.sqrt(3)
    return (x1, y1, x2, y2, x3, y3)

# Funkcja sprawdzająca czy trójkąt mieści się w elipsie
def is_inside_ellipse(cx,cy,angle):
    triangle=calculate_triangle(cx,cy,angle)
    for i in range(0, 6, 2):
        x, y = triangle[i], triangle[i + 1]
        if (x**2 / a**2 + y**2 / b**2) > 1:
            return False
    return True

# Funkcja sprawdzająca, czy trójkąty się nakładają
def check_overlap(cx,cy,angle, population):
    triangle = calculate_triangle(cx, cy, angle)
    new_poly = Polygon([(triangle[0], triangle[1]), (triangle[2], triangle[3]), (triangle[4], triangle[5])])
    for i in range(0,len(population),3):
        other=calculate_triangle(population[i],population[i+1],population[i+2])
        other_poly = Polygon([(other[0], other[1]), (other[2], other[3]), (other[4], other[5])])
        if new_poly.intersects(other_poly):
            return True
    return False

def get_existing_triangles(individual, exclude_index):
    triangles = []
    for i in range(0, len(individual), 3):
        if i == exclude_index:
            continue
        cx, cy, theta = individual[i], individual[i+1], individual[i+2]
        triangles.append(cx)
        triangles.append(cy)
        triangles.append(theta)
    return triangles

def sort_individual_by_cx(individual):
    # Podzielenie wiersza na grupy trójek
    genes = [individual[i:i + 3] for i in range(0, len(individual), 3)]
    # Sortowanie grup według wartości cx
    sorted_genes = sorted(genes, key=lambda gene: gene[0])  # sortowanie po cx
    # Złączenie posortowanych trójek w jeden wiersz
    return [value for gene in sorted_genes for value in gene]

def plot_for_population(index,triangles,ax):
    ax.set_title(f"Populacja {index}")
    ax.set_xlim(-a - 1, a + 1)
    ax.set_ylim(-b - 1, b + 1)

    ellipse = Ellipse((0, 0), 2 * a, 2 * b, edgecolor='blue', facecolor='none', linewidth=1.5)
    ax.add_patch(ellipse)

    # Dodanie trójkątów triangles
    for t in range(0,len(triangles),3):
        cx,cy,theta=triangles[t],triangles[t+1],triangles[t+2]
        if not (cx==0 and cy==0 and theta==0):
            triangle=calculate_triangle(cx,cy,theta)
            x_coords = [triangle[0], triangle[2], triangle[4], triangle[0]]
            y_coords = [triangle[1], triangle[3], triangle[5], triangle[1]]
            ax.plot(x_coords, y_coords, 'r-')

def initialize_population(pop_size, max_triangle):
    tablica = [[0 for _ in range(3 * max_triangle)] for _ in range(pop_size)]
    for individual in range(len(tablica)):  # przechodzenie po każdym osobniku populacji
        triangles=[]
        for gen in range(0, len(tablica[0]), 3):  # zapisywanie każdego genu
            triangle = generate_equilateral_triangle(triangles)
            if triangle is not None:
                cx, cy, theta = triangle
                tablica[individual][gen], tablica[individual][gen + 1], tablica[individual][gen + 2] = cx, cy, theta
                triangles.append(cx)
                triangles.append(cy)
                triangles.append(theta)
        tablica[individual] = sort_individual_by_cx(tablica[individual])  # sortowanie tablicy grupami od najmniejszego cx
    return tablica

def fitness_function(tablica):
    fitness_scores = []  # Lista przechowująca wyniki fitness dla każdego osobnika
    for individual in tablica:  # Przechodzenie po każdym osobniku
        score = 0
        for gen in range(0, len(individual), 3):  # Sprawdzanie każdego trójkąta
            # Jeśli wszystkie wartości są różne od 0, uznajemy, że trójkąt istnieje
            if individual[gen] != 0 or individual[gen + 1] != 0 or individual[gen + 2] != 0:
                score += 1
        fitness_scores.append(score)
    return fitness_scores

def roulette_wheel_selection(tablica, fitness_scores,reproductive_size):
    selected_individuals = []
    selected_indices = []  # Lista indeksów wybranych osobników
    total_fitness = sum(fitness_scores)

    # Tworzenie listy prawdopodobieństw
    probabilities = [fitness / total_fitness for fitness in fitness_scores]

    # Losowanie bez powtórzeń
    while len(selected_individuals) < reproductive_size:
        rand = random.uniform(0, 1)
        cumulative_probability = 0
        for i, probability in enumerate(probabilities):
            cumulative_probability += probability
            if rand <= cumulative_probability:
                selected_individuals.append(tablica[i])
                selected_indices.append(i)  # Dodanie indeksu do listy wybranych
                break
    return selected_individuals

def crossover(parent1, parent2, crossover_rate, method):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]  # no crossover, return copies of the parents

    # Convert parents to lists to make them mutable
    child1 = list(parent1)
    child2 = list(parent2)

    if method == 'uniform':
        for i in range(0, len(parent1), 3):
            if random.random() < 0.5:
                # Swap triangles between parents
                child1[i:i+3], child2[i:i+3] = parent2[i:i+3], parent1[i:i+3]
    elif method == 'one_point':
        point = random.randint(1, len(parent1)//3 - 1) * 3
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    elif method == 'two_point':
        p1 = random.randint(1, len(parent1)//3 - 2) * 3
        p2 = random.randint(p1 + 3, len(parent1)) * 3
        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    # Now check if the children satisfy the conditions (no overlapping triangles, and inside the ellipse)
    for child in [child1, child2]:
        triangles=[]
        for i in range(0, len(child), 3):
            if check_overlap(child[i], child[i+1], child[i+2], triangles):
                # Generate a new triangle that meets the conditions
                new_triangle = generate_equilateral_triangle( triangles)
                if new_triangle is None:
                    child[i], child[i+1], child[i+2] = 0, 0, 0  # Invalidate the triangle if it's not valid
                else:
                    child[i], child[i+1], child[i+2] = new_triangle
                    triangles.append(child[i]), triangles.append(child[i+1]), triangles.append(child[i+2])
            else:
                triangles.append(child[i]), triangles.append(child[i+1]), triangles.append(child[i+2])
        child=sort_individual_by_cx(child)

    return child1, child2

def get_elite_individuals(tablica, fitness_scores, elitism_count):
    sorted_population = sorted(zip(tablica, fitness_scores), key=lambda x: x[1], reverse=True)
    elite = [individual for individual, _ in sorted_population[:elitism_count]]
    return elite
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        # Wybierz losowy trójkąt (indeks co 3)
        num_triangles = len(individual) // 3
        triangle_index = random.randint(0, num_triangles - 1)
        i = triangle_index * 3
        original_triangle = individual[i:i + 3]  # Save original triangle (before mutation)
        # Usuwamy trójkąt z listy na tym indeksie
        individual[i:i + 3] = [0, 0, 0]
        triangle = generate_equilateral_triangle(get_existing_triangles(individual,triangle_index))
        if triangle is None:
            individual[i:i + 3]=original_triangle
        else:
            cx, cy, theta=triangle
            individual[i]=cx
            individual[i+1]=cy
            individual[i + 2] = theta
            individual=sort_individual_by_cx(individual)
    return individual

def genetic_algorithm(pop_size,reproductive_size,crossover_rate,elitism_count,method,mutation_rate,tablica):
    #ocena wszystkich osobników populacji
    fitness=fitness_function(tablica)
    #Elitaryzm
    elite_individuals = get_elite_individuals(tablica, fitness, elitism_count)
    #selekcja
    selected_individuals=roulette_wheel_selection(tablica,fitness,reproductive_size)
    #Krzyżowanie
    offspring = elite_individuals
    for i in range(0, len(selected_individuals) - 1, 2):
        parent1 = selected_individuals[i]
        parent2 = selected_individuals[i + 1]
        child1, child2 = crossover(parent1, parent2,crossover_rate,method)
        #mutacja
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        offspring.append(child1)
        offspring.append(child2)

    # Nowa populacja = elita + dzieci (przycięte do wielkości populacji)
    tablica = offspring[:pop_size ]
    return tablica
def plot_fitness_variance():
    """Tworzy wykres zmian wariancji fitness i odchylenia standardowego w generacjach."""
    # Obliczamy wariancję, odchylenie standardowe i średni fitness dla każdej generacji
    variance_fitness = [np.var(fitness_function(tablica)) for tablica in generation_data]
    std_dev_fitness = [np.std(fitness_function(tablica)) for tablica in generation_data]

    # Czyszczenie istniejących wykresów
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tworzenie wykresu
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    generations = range(1, len(variance_fitness) + 1)

    # Rysujemy wariancję fitness
    ax.plot(generations, variance_fitness, marker="o", label="Wariancja fitness", color="purple")

    # Rysujemy odchylenie standardowe fitness
    ax.plot(generations, std_dev_fitness, marker="x", label="Odchylenie standardowe fitness", color="orange")

    # Ustawienia osi i legendy
    ax.set_title("Wariancja, odchylenie standardowe")
    ax.set_xlabel("Generacja")
    ax.set_ylabel(" Wariancja / Odchylenie")
    ax.legend()
    ax.grid(True)

    # Osadzanie wykresu
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def plot_parameter_distribution(generation_index):
    """Tworzy scatter plot dla parametrów trójkątów w wybranej generacji."""
    # Pobierz dane dla wybranej generacji
    tablica = generation_data[generation_index]  # Macierz generacji: wiersze -> populacje, kolumny -> parametry

    # Przygotowanie list na współrzędne i kąty
    centers_x = []
    centers_y = []
    angles = []

    # Iteracja przez populacje (wiersze macierzy)
    for population in tablica:  # Każda `population` to lista parametrów jednego osobnika
        for i in range(0, len(population), 3):  # Grupy [cx, cy, theta]
            cx = population[i]       # Współrzędna X
            cy = population[i + 1]   # Współrzędna Y
            theta = population[i + 2]  # Kąt odchylenia
            centers_x.append(cx)
            centers_y.append(cy)
            angles.append(theta)

    # Czyszczenie istniejących wykresów w `plot_frame`
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tworzenie scatter plot
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(centers_x, centers_y, c=angles, cmap="viridis", label="Trójkąty")
    fig.colorbar(scatter, ax=ax, label="Kąt odchylenia")

    # Ustawienia wykresu
    ax.set_title(f"Rozkład parametrów (Generacja {generation_index + 1})")
    ax.set_xlabel("Współrzędne X")
    ax.set_ylabel("Współrzędne Y")
    ax.legend()
    ax.grid(True)

    # Osadzanie wykresu w aplikacji tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def plot_evolution_time():
    """Tworzy wykres czasu trwania ewolucji w generacjach."""
    # Czyszczenie istniejących wykresów
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tworzymy wykres czasu ewolucji
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    generations = range(1, len(generation_times) + 1)
    ax.plot(generations, generation_times, marker="o", label="Czas ewolucji", color="blue")

    # Ustawienia wykresu
    ax.set_title("Czas trwania ewolucji w generacjach")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Czas obliczeń (s)")
    ax.legend()
    ax.grid(True)

    # Osadzanie wykresu
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()
def animate_triangle_movement():
    """Tworzy animację pokazującą rozkład parametrów dla każdej generacji."""

    fig, ax = plt.subplots(figsize=(10, 6))

    centers_x, centers_y = [], []
    scatter = ax.scatter(centers_x, centers_y, color="blue", s=50)

    ax.set_xlim(-a - 1, a + 1)
    ax.set_ylim(-b - 1, b + 1)
    ax.set_xlabel("Współrzędne X")
    ax.set_ylabel("Współrzędne Y")

    text_annotation = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                              ha="right", va="top", fontsize=12, color="black",
                              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        text_annotation.set_text("")
        return scatter, text_annotation

    def update(frame):
        tablica = generation_data[frame]

        centers_x = []
        centers_y = []

        for individual in tablica:
            for i in range(0, len(individual), 3):
                cx = individual[i]
                cy = individual[i + 1]
                centers_x.append(cx)
                centers_y.append(cy)

        coords = np.column_stack((centers_x, centers_y))
        scatter.set_offsets(coords)

        ax.set_title("Animacja rozkładu parametrów")
        text_annotation.set_text(f"Generacja: {frame + 1}")

        return scatter, text_annotation

    ani = FuncAnimation(fig, update, frames=len(generation_data), init_func=init, blit=True, interval=1000)

    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()



def plot_fitness_over_generations():
    """Tworzy wykres zmian fitness w generacjach, uwzględniając inicjalizującą populację, i wyświetla dane w polu tekstowym."""
    # Pobieramy dane fitness dla każdej generacji
    max_fitness_per_generation = [max(fitness_function(tablica)) for tablica in generation_data]
    avg_fitness_per_generation = [sum(fitness_function(tablica)) / len(tablica) for tablica in generation_data]
    min_fitness_per_generation = [min(fitness_function(tablica)) for tablica in generation_data]

    # Czyszczenie istniejących wykresów w `plot_frame`
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tworzenie figury dla wykresu
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    generations = range(1, len(max_fitness_per_generation) + 1)

    # Rysowanie danych: maks, średnia i min
    ax.plot(generations, max_fitness_per_generation, marker="o", label="Maksymalny fitness", color="red")
    ax.plot(generations, avg_fitness_per_generation, marker="x", label="Średni fitness", color="blue")
    ax.plot(generations, min_fitness_per_generation, marker="*", label="Minimalny fitness", color="green")

    # Ustawienia osi i legendy
    ax.set_title("Zmiana fitness w generacjach")
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)

    # Osadzanie figury w `plot_frame`
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Dodanie pola tekstowego i scrollbar
    text_frame = tk.Frame(plot_frame, bg="lightgray")
    text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

    text_area = tk.Text(text_frame, height=20, width=30, bg="lightgray", fg="black", wrap=tk.NONE)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_area.config(yscrollcommand=scrollbar.set)

    # Wypełnienie pola tekstowego danymi fitness
    for i, tablica in enumerate(generation_data):
        fitness_scores = fitness_function(tablica)
        text_area.insert(tk.END, f"Generacja: {i + 1}\n{fitness_scores}\n\n")

def alghoritm(pop_size, generations, area_par, a_param, b_param, max_triangle, reproductive_size, crossover_rate, elitism_count, method, mutation_rate):
    global area, a, b, generation_data, generation_times
    area = area_par
    a = a_param
    b = b_param

    generation_times = []  # Lista przechowująca czasy obliczeń dla każdej generacji

    # Inicjalizacja populacji początkowej
    tablica = initialize_population(pop_size, max_triangle)
    generation_data = [tablica]  # Pierwsza generacja (inicjalizacja populacji)

    # Pomiar czasu dla inicjalizującej populacji
    start_time = time.time()
    print("Generacja: 1")
    print(fitness_function(tablica))
    end_time = time.time()
    generation_times.append(end_time - start_time)  # Zapisujemy czas obliczeń

    # Wykonujemy dokładnie (generations - 1) iteracji, zaczynając od "Generacja 2"
    for generation in range(1, generations):  # Zaczynamy od 1 (bo 0 to inicjalizacja)
        start_time = time.time()  # Rozpoczynamy pomiar czasu
        print(f"Generacja: {generation + 1}")  # Liczymy generację poprawnie
        tablica = genetic_algorithm(pop_size, reproductive_size, crossover_rate, elitism_count, method, mutation_rate, tablica)
        generation_data.append(tablica)  # Zapisujemy dane dla kolejnej generacji
        print(fitness_function(tablica))  # Wyświetlamy dane fitness dla bieżącej generacji
        end_time = time.time()  # Kończymy pomiar czasu
        generation_times.append(end_time - start_time)  # Zapisujemy czas obliczeń

    update_plot(0)  # Wyświetlamy pierwszy wykres (inicjalizacja populacji)
def view_subplots_for_generation():
    """Przejście do widoku subplotów dla aktualnej generacji."""
    update_plot(current_generation_index)  # Wywołuje funkcję, która wyświetla subploty dla generacji

def update_plot(generation_index):
    """Aktualizuje wykres na podstawie indeksu generacji."""
    global current_generation_index
    current_generation_index = generation_index  # Aktualizujemy indeks generacji

    # Pobieramy populację dla bieżącej generacji
    tablica = generation_data[generation_index]
    pop_size = len(tablica)

    # Czyszczenie istniejących wykresów w `plot_frame`
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tworzenie subplotów
    rows = math.ceil(pop_size / 3)
    cols = min(pop_size, 3)
    fig = Figure(figsize=(15, 5 * rows))
    axes = fig.subplots(rows, cols)

    if pop_size == 1:  # Obsługa przypadku z 1 osobnikiem
        axes = [axes]

    axes = axes.flatten()
    for i in range(pop_size):
        plot_for_population(i, tablica[i], axes[i])  # Rysujemy trójkąty dla każdego osobnika

    # Ukrywanie nieużywanych subplotów (jeśli liczba subplotów > pop_size)
    for i in range(pop_size, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Generacja {generation_index + 1}")  # Tytuł wykresu

    # Osadzanie figury w `plot_frame`
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()


def previous_generation():
    """Przechodzi do poprzedniej generacji, jeśli to możliwe."""
    if current_generation_index > 0:
        update_plot(current_generation_index - 1)


def next_generation():
    """Przechodzi do następnej generacji, jeśli to możliwe."""
    if current_generation_index < len(generation_data) - 1:
        update_plot(current_generation_index + 1)

def add_placeholder(entry, placeholder):
    entry.insert(0, placeholder)
    entry.config(fg="grey")

    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")

    def on_focus_out(event):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.config(fg="grey")

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

def run_algorithm():
    try:
        # Pobieranie wartości z pól tekstowych
        pop_size = int(entry_pop_size.get())
        generations = int(entry_generations.get())
        area_par = float(entry_area.get())
        a_param = float(entry_a.get())
        b_param = float(entry_b.get())
        max_triangle = int(entry_max_triangles.get())
        elitism_count = int(entry_elitism.get())
        reproductive_size = pop_size - elitism_count
        crossover_rate = float(entry_crossover.get())
        mutation_rate = float(entry_mutation.get())
        crossover_method = method_combobox.get()
        generation_slider.config(from_=1, to=generations)
        # Uruchamianie funkcji algorithm
        alghoritm(pop_size, generations, area_par, a_param, b_param, max_triangle,
                  reproductive_size, crossover_rate, elitism_count,
                  crossover_method, mutation_rate)
        print("Algorytm uruchomiony pomyślnie!")

    except ValueError:
        print("Proszę wprowadzić prawidłowe wartości!")

# Tworzenie interfejsu
root = tk.Tk()
root.title("Interfejs Algorytmu Genetycznego")
root.geometry("1200x800")  # Ustawiamy rozmiar okna

# Podział na ramki (parametry po lewej, wykresy po prawej)
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Ramka z możliwością przewijania dla parametrów
param_scroll_frame = tk.Frame(main_frame, padx=10, pady=10, bg="lightgray")
param_scroll_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)

# Canvas dla przewijania długości (pionowo)
param_canvas = tk.Canvas(param_scroll_frame, bg="lightgray", width=250)  # Szerokość ustawiona na stałą
param_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)

scrollbar = tk.Scrollbar(param_scroll_frame, orient=tk.VERTICAL, command=param_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

param_canvas.configure(yscrollcommand=scrollbar.set)
param_canvas.bind("<Configure>", lambda event: param_canvas.config(scrollregion=param_canvas.bbox("all")))

# Właściwa ramka parametrów wewnątrz canvas
param_frame = tk.Frame(param_canvas, padx=10, pady=10, bg="lightgray")
param_canvas.create_window((0, 0), window=param_frame, anchor="nw")

plot_frame = tk.Frame(main_frame, padx=10, pady=10, bg="white", relief=tk.SUNKEN, borderwidth=2)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Parametry wejściowe
tk.Label(param_frame, text="Rozmiar populacji:", bg="lightgray").pack(anchor="w", pady=5)
entry_pop_size = tk.Entry(param_frame)
entry_pop_size.pack(fill=tk.X, pady=5)
add_placeholder(entry_pop_size, "5")

tk.Label(param_frame, text="Liczba generacji:", bg="lightgray").pack(anchor="w", pady=5)
entry_generations = tk.Entry(param_frame)
entry_generations.pack(fill=tk.X, pady=5)
add_placeholder(entry_generations, "10")

tk.Label(param_frame, text="Powierzchnia (area):", bg="lightgray").pack(anchor="w", pady=5)
entry_area = tk.Entry(param_frame)
entry_area.pack(fill=tk.X, pady=5)
add_placeholder(entry_area, "5")

tk.Label(param_frame, text="Parametr a:", bg="lightgray").pack(anchor="w", pady=5)
entry_a = tk.Entry(param_frame)
entry_a.pack(fill=tk.X, pady=5)
add_placeholder(entry_a, "5")

tk.Label(param_frame, text="Parametr b:", bg="lightgray").pack(anchor="w", pady=5)
entry_b = tk.Entry(param_frame)
entry_b.pack(fill=tk.X, pady=5)
add_placeholder(entry_b, "4")

tk.Label(param_frame, text="Maksymalna liczba trójkątów:", bg="lightgray").pack(anchor="w", pady=5)
entry_max_triangles = tk.Entry(param_frame)
entry_max_triangles.pack(fill=tk.X, pady=5)
add_placeholder(entry_max_triangles, "7")

tk.Label(param_frame, text="Liczba elit (elitism_count):", bg="lightgray").pack(anchor="w", pady=5)
entry_elitism = tk.Entry(param_frame)
entry_elitism.pack(fill=tk.X, pady=5)
add_placeholder(entry_elitism, "1")

tk.Label(param_frame, text="Wskaźnik krzyżowania (crossover_rate):", bg="lightgray").pack(anchor="w", pady=5)
entry_crossover = tk.Entry(param_frame)
entry_crossover.pack(fill=tk.X, pady=5)
add_placeholder(entry_crossover, "0.7")

tk.Label(param_frame, text="Wskaźnik mutacji (mutation_rate):", bg="lightgray").pack(anchor="w", pady=5)
entry_mutation = tk.Entry(param_frame)
entry_mutation.pack(fill=tk.X, pady=5)
add_placeholder(entry_mutation, "0.01")

tk.Label(param_frame, text="Metoda krzyżowania:", bg="lightgray").pack(anchor="w", pady=5)
method_combobox = ttk.Combobox(param_frame, values=["uniform", "one_point", "two_point"])
method_combobox.pack(fill=tk.X, pady=5)
method_combobox.set("uniform")

# Przyciski
run_button = tk.Button(param_frame, text="Uruchom Algorytm", command=run_algorithm, bg="green", fg="white")
run_button.pack(fill=tk.X, pady=10)

animation_button = tk.Button(param_frame, text="Animacja Ruchu", command=animate_triangle_movement, bg="darkorange", fg="white")
animation_button.pack(fill=tk.X, pady=5)


subplots_button = tk.Button(param_frame, text="Wykresy Generacji", command=view_subplots_for_generation, bg="darkcyan", fg="white")
subplots_button.pack(fill=tk.X, pady=5)

# Strzałki do nawigacji między generacjami
nav_frame = tk.Frame(param_frame, bg="lightgray")
nav_frame.pack(fill=tk.X, pady=10)

prev_button = tk.Button(nav_frame, text="<", command=previous_generation, bg="orange", fg="black")
prev_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

next_button = tk.Button(nav_frame, text=">", command=next_generation, bg="orange", fg="black")
next_button.pack(side=tk.RIGHT, expand=True, fill=tk.X)
# Suwak do wyboru generacji
generation_slider = tk.Scale(param_frame, from_=1, to=1, orient=tk.HORIZONTAL, label="Wybór generacji", bg="lightgray", command=lambda x: update_plot(int(x) - 1))
generation_slider.pack(fill=tk.X, pady=10)

fitness_button = tk.Button(param_frame, text="Wykres fitness", command=plot_fitness_over_generations, bg="purple", fg="white")
fitness_button.pack(fill=tk.X, pady=10)
# Przycisk dla zmiany różnorodności populacji (wariancja fitness)
variance_button = tk.Button(param_frame, text=" Wariancja Fitness", command=plot_fitness_variance, bg="teal", fg="white")
variance_button.pack(fill=tk.X, pady=5)

# Przycisk dla rozkładu parametrów (scatter/heatmap)
parameter_distribution_button = tk.Button(param_frame, text="Rozkład Parametrów", command=lambda: plot_parameter_distribution(current_generation_index), bg="orange", fg="white")
parameter_distribution_button.pack(fill=tk.X, pady=5)

# Przycisk dla czasu trwania ewolucji
evolution_time_button = tk.Button(param_frame, text="Czas Ewolucji", command=plot_evolution_time, bg="blue", fg="white")
evolution_time_button.pack(fill=tk.X, pady=5)


root.mainloop()

