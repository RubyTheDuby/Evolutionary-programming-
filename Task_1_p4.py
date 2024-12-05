import random
import matplotlib.pyplot as plt
import numpy

# Function for the main algorithm
def main():
    Population = []
    MaxFitnesses = []
    AvgFitnesses = []

    GenerationIndex = 0
    global SolutionIndex
    SolutionIndex = None

    # Main algorithm
    initialise_population(Population, PopulationSize)

    while GenerationIndex < MaxGeneration:
        # Some housekeeping
        display_progress(GenerationIndex, Population)
        check_if_problem_has_been_solved(GenerationIndex, Population)
        track_fitness(MaxFitnesses, AvgFitnesses, Population)

        # Perform genetic operations
        NewPopulation = []

        for i in range(NumCrossovers):
            Parent1 = Population[random.randint(0, NumCrossovers)]
            Parent2 = Population[random.randint(0, NumCrossovers)]
            Offspring = Parent1.crossover(Parent2)
            if cointoss(MutationRate):
                Offspring.mutate()
            NewPopulation.append(Offspring)

        del Population[NumCloned - PopulationSize:]
        Population.extend(NewPopulation)

        # Some housekeeping
        Population = sorted(Population, key=fitness, reverse=True)
        GenerationIndex += 1

    # At end of main, plot results, and return number of generations required for
    # a solution (the latter is useful when running main interactively, as opposed
    # to as a script).
    plot_fitness_by_generation(MaxFitnesses, AvgFitnesses)
    return return_solution_if_found(GenerationIndex)

# Constants / hyperparameters
PopulationSize = 300 # Size of the population
CrossoverRate = 0.40 # probability of crossoverrate
MutationRate = 0.75  # Probability of mutation
NBits = 4            # Number of bits in the genotype
MaxGeneration = 100  # Maximum number of generations
PRINTING = True      # Whether to print progress during the algorithm execution

# Derived constants / hyperparameters
CloningRate = 1 - CrossoverRate
NumCloned = int(PopulationSize * CloningRate)
NumCrossovers = PopulationSize - NumCloned

# Global placeholders / accumulators
global SolutionIndex
SolutionIndex = None
global checker
checker = None

# Individual class representing individuals in the population
class Individual_t:
    def __init__(self, geno):
        self.Genotype = geno

    def get_phenotype(self):
        return str(self.Genotype)

    def __repr__(self):
        # Function to represent the individual in the letter format
        listnum = self.get_phenotype()
        string = ""
        answer1 = int(listnum[3] + listnum[8] + listnum[4] + listnum[2] + listnum[8]) - int(
            listnum[7] + listnum[6] + listnum[8] + listnum[1] + listnum[3])
        alphabet = ["X", "U", "T", "S", "N", "M", "L", "C", "A"] 
        for x in str(answer1):
            place = 0
            for i in listnum:
                if x == i:
                    string += alphabet[place]
                    break
                place += 1
        return string

    def mutate(self):
        # Function to mutate an individual
        rand = random.randint(0, 8)
        rand1 = random.randint(0, 8)
        while rand1 == rand:
            rand1 = random.randint(0, 8)

        genotype_list = list(self.Genotype)
        store = genotype_list[rand]
        genotype_list[rand] = genotype_list[rand1]
        genotype_list[rand1] = store

        self.Genotype = ''.join(genotype_list)

    def crossover(self, Individual2):
        # Function to perform crossover between two individuals
        ChiasmaLocation = random.randint(0, 8)
        my_nektionary = []
        seen_items = []  
        x = 8
        for item1, item2 in zip(self.get_phenotype(), Individual2.get_phenotype()):
            if x <= ChiasmaLocation:
                my_nektionary.append(item1)
                seen_items.append(item1)

            if x > ChiasmaLocation:
                if int(item2) in seen_items:
                    my_nektionary.append(item2)
                    seen_items.append(item2)
                else:
                    my_nektionary.append(item1)
                    seen_items.append(item1)

            x -= 1
        
        return Individual_t("".join(my_nektionary))

    @staticmethod
    def create_random():
        # Function to create a random individual
        numlist = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        random.shuffle(numlist)
        return Individual_t("".join(numlist))


# Helper functions

def fitness(Individual):
    # Function to calculate fitness of an individual
    listnum = Individual.get_phenotype()
    answer1 = int(listnum[3] + listnum[8] + listnum[4] + listnum[2] + listnum[8]) - int(
        listnum[7] + listnum[6] + listnum[8] + listnum[1] + listnum[3])
    answer2 = int(listnum[0] + listnum[5] + listnum[8] + listnum[3])
    
    if len(str(answer1)) == 4:
        ANS = sum(str(answer1)[3 - i] == str(answer2)[3 - i] for i in range(4))
        if answer1 == answer2:
            global SolutionIndex
            global checker
            checker = "win"
    else:
        ANS = 0

    return ANS
#The fitness function above gets its answer by minusing SANTA from CLAUSE and then compares the answer to XMAS by
#Going through line by line and counting the number of times each index is similiar. 

def cointoss(Bias=0.5):
    # Function for performing a coin toss
    return random.random() < Bias

def display_progress(GenerationIndex, Population):
    # Function to display progress of the algorithm
    if PRINTING: 
        print(f"Population at Generation {GenerationIndex:02d}: {Population}")

def check_if_problem_has_been_solved(GenerationIndex, Population):
    # Function to check if the problem has been solved
    global SolutionIndex
    global checker

    if checker is None:
        SolutionIndex = GenerationIndex

def track_fitness(MaxFitnesses, AvgFitnesses, Population):
    # Function to track fitness statistics
    Fitnesses = numpy.array([fitness(i) for i in Population])
    MaxFitnesses.append(Fitnesses.max())
    AvgFitnesses.append(Fitnesses.mean())

def initialise_population(Population, PopulationSize):
    # Function to initialize the population
    Population[:] = [Individual_t.create_random() for i in range(PopulationSize)]
    print(Population)
    Population.sort(key=fitness, reverse=True)

def plot_fitness_by_generation(MaxFitnesses, AvgFitnesses):
    # Function to plot fitness statistics by generation
    if PRINTING:
        plt.plot(range(MaxGeneration), MaxFitnesses, label="Max")
        plt.plot(range(MaxGeneration), AvgFitnesses, label="Avg")
        plt.legend()
        plt.title("Avg vs Max Fitness per Generation")
        plt.show(block=True)

def return_solution_if_found(GenerationIndex):
    # Function to return solution index if found
    global SolutionIndex
    if SolutionIndex is not None:
        if PRINTING: print(f"\n'XMAS' Solution found in Generation {SolutionIndex}.")
        return SolutionIndex
    else:
        if PRINTING: print(f"'XMAS' Solution not found by generation {GenerationIndex}.")
        return -1

if __name__ == '__main__':
    main()
