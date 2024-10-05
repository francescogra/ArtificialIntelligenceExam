import random
import os
import matplotlib.pyplot as plt
import image_test

TABU_LIST_LENGTH = 200
POL_SIZE = 3
POL_COUNT = 200
ITER_NUM = 1200000

NUM_OF_PARAMS = POL_COUNT * (POL_SIZE * 2 + 4)

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

TestImage = image_test.TestImage("MonaLisa.png", POL_SIZE)

BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0

def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

def getDiff(individual):
    return TestImage.getDifference(individual, "MSE"),

def saveImage(gen, polygonData):
    if gen % 100 == 0:
        folder = "images/results/run-{}-{}".format(POL_SIZE, POL_COUNT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        TestImage.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))

def perturbation(elem, tabu_list, exploration=False):
    perturbed = list(elem)

    # Perturbazione casuale delle coordinate dei vertici
    for i in range(POL_COUNT):
        for j in range(POL_SIZE * 2):
            perturbed[i * (POL_SIZE * 2 + 4) + j] += random.uniform(-0.005, 0.005)

    # Perturbazione casuale dei valori di colore e trasparenza
    for i in range(POL_COUNT):
        for j in range(POL_SIZE * 2, POL_SIZE * 2 + 4):
            perturbed[i * (POL_SIZE * 2 + 4) + j] += random.uniform(-0.05, 0.05)

            # Limitiamo i valori all'intervallo [0, 1]
            if perturbed[i * (POL_SIZE * 2 + 4) + j] < 0:
                perturbed[i * (POL_SIZE * 2 + 4) + j] = 0
            elif perturbed[i * (POL_SIZE * 2 + 4) + j] > 1:
                perturbed[i * (POL_SIZE * 2 + 4) + j] = 1

    # Infine, restituiamo la perturbazione
    perturbed_tuple = tuple(perturbed)
    if perturbed_tuple in tabu_list:
        return perturbation(elem, tabu_list, exploration)
    else:
        return perturbed_tuple

def mutation(elem):
    mutated = list(elem)  # Inizializziamo la mutazione come una copia della soluzione corrente

    # Eseguiamo le strategie specifiche per la mutazione

    # Mutazione casuale dei parametri
    for i in range(NUM_OF_PARAMS):
        mutated[i] += random.uniform(-0.002, 0.002)

        # Limitiamo i valori all'intervallo [0, 1]
        if mutated[i] < 0:
            mutated[i] = 0
        elif mutated[i] > 1:
            mutated[i] = 1

    # Infine, restituiamo la mutazione
    return tuple(mutated)

def tabu_search(ITER_NUMations):
    start_pt = randomFloat(BOUNDS_LOW, BOUNDS_HIGH)
    best_solution = start_pt
    best_evaluation = getDiff(start_pt)[0]
    tabu_list = []

    consecutive_no_improvement = 0
    for i in range(ITER_NUMations):
        # Applica perturbazione seguita da mutazione
        candidate = perturbation(best_solution, tabu_list, exploration=(consecutive_no_improvement > 10000))
        candidate = mutation(candidate)
        candidate_eval = getDiff(candidate)[0]

        if i % 1000 == 0:
            print("Iterazione ", i, " --> ", candidate_eval)
            saveImage(i, best_solution)

        tabu_list.append(candidate)
        if len(tabu_list) > TABU_LIST_LENGTH:
            tabu_list.pop(0)

        if candidate_eval <= best_evaluation:
            best_solution = candidate
            best_evaluation = candidate_eval
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

    return best_solution

best = tabu_search(ITER_NUM)
print(best)

TestImage.plotImages(TestImage.polygonDataToImage(best))
plt.show()
