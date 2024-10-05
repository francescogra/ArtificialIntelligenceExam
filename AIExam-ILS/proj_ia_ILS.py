import random
import os
import matplotlib.pyplot as plt
import image_test

ITER_NUM = 1200000
POL_SIZE = 3
POL_COUNT = 110

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = POL_COUNT * (POL_SIZE * 2 + 4)

# set the random seed:
RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

# create the image test class instance:
TestImage = image_test.TestImage("MonaLisa.png", POL_SIZE)

# calculate total number of params in chromosome:
# For each polygon we have:
# two coordinates per vertex, 3 color values, one alpha value
NUM_OF_PARAMS = POL_COUNT * (POL_SIZE * 2 + 4)

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions

# helper function for creating random real numbers uniformly distributed within a given range [low, up]
# it assumes that the range is the same for every dimension
def randomFloat(low, up):
    return [random.uniform(l, u) for l, u in zip([low] * NUM_OF_PARAMS, [up] * NUM_OF_PARAMS)]

# fitness calculation using MSE as difference metric:
def getDiff(individual):
    return TestImage.getDifference(individual, "MSE"),
    #return TestImage.getDifference(individual, "SSIM"),

# this function take polygons and save them into an image
def saveImage(gen, polygonData):
    # only every 100 generations:
    if gen % 100 == 0:
        # create folder if does not exist:
        folder = "images/results/run-{}-{}".format(POL_SIZE, POL_COUNT)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # save the image in the folder:
        TestImage.saveImage(polygonData,
                            "{}/after-{}-gen.png".format(folder, gen),
                            "After {} Generations".format(gen))


# Perturbation function
def perturbation(element):
    """
    Perturbs the given element by adding a random offset to each element.
    :param element: tuple representing an element to be perturbed
    :return: perturbed tuple
    """
    temp = []
    for item in element:
        offset = item + random.uniform(-0.1, 0.1)
        random_value = random.randint(1, 128)
        if random_value == 1:
            if offset < 0:
                offset = 0
            if offset > 1:
                offset = 1
            temp.append(offset)
        else:
            temp.append(item)
    perturbed = tuple(temp)
    return perturbed


# Iterated Local Search (ILS) algorithm
def ils(iterations):
    """
    Executes the Iterated Local Search algorithm.
    :param iterations: number of iterations to perform
    :return: best solution found
    """
    # Generate a random initial point for the search
    start_point = randomFloat(BOUNDS_LOW, BOUNDS_HIGH)
    solution = start_point
    solution_evaluation = getDiff(start_point)[0]

    for i in range(iterations):
        # Perform perturbation
        candidate = perturbation(solution)

        # Evaluate candidate
        candidate_evaluation = getDiff(candidate)[0]

        # Create and save an image every 1000 steps
        if i % 1000 == 0:
            print("Solution ", i, " --> ", solution_evaluation)
            print("Candidate ", i, " --> ", candidate_evaluation)
            saveImage(i, solution)

        # If candidate is less than or equal to the current solution, perform the substitution
        if candidate_evaluation <= solution_evaluation:
            # Store the new point
            solution, solution_evaluation = candidate, candidate_evaluation

    return solution

# run ils
best = ils(ITER_NUM)
print(best)

#plot best solution
TestImage.plotImages(TestImage.polygonDataToImage(best))
plt.show()