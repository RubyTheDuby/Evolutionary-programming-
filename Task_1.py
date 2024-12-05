import random
import matplotlib.pyplot as plt
import numpy

class GeneticAlgorithm:
    # Initiaizes the genetic Algorithm 
    def __init__(self, PopulationSize, CrossoverRate, MutationRate, NBits, PRINTING, MaxGeneration, fitness_function):
        self.PopulationSize = PopulationSize # Size of the population
        self.CrossoverRate = CrossoverRate   # probability of crossoverrate
        self.MutationRate = MutationRate     # Probability of mutation
        self.NBits = NBits                   # Number of bits in the genotype
        self.PRINTING = PRINTING             # Whether to print progress during the algorithm execution
        self.MaxGeneration = MaxGeneration   # Maximum number of generations
        self.fitness_function = fitness_function 
        # Derived constants / hyperparameters
        self.CloningRate = 1 - self.CrossoverRate
        self.NumCloned = int(self.PopulationSize * self.CloningRate)
        self.NumCrossovers = self.PopulationSize - self.NumCloned
        # Global placeholders / accumulators
        global SolutionIndex
        SolutionIndex = None

    # main function that runs the genetic algorithm 
    def run(self):
        Population = []
        MaxFitnesses = []
        AvgFitnesses = []
        GenerationIndex = 0

        self.initialise_population(Population)
        # Main loop that is run through 
        while GenerationIndex < self.MaxGeneration:
            # Some housekeeping
            self.display_progress(GenerationIndex, Population)
            self.check_if_problem_has_been_solved(GenerationIndex, Population)
            self.track_fitness(MaxFitnesses, AvgFitnesses, Population)

            # Perform genetic operations
            NewPopulation = []

            for i in range(self.NumCrossovers):
                selected_parents = self.tournament_selection(Population, 2)  # select 2 parents to play in the tournament
                Parent1, Parent2 = selected_parents[0], selected_parents[1]
                Offspring = Parent1.crossover(Parent2, self.NBits) 
                if self.cointoss(self.MutationRate):
                    Offspring.mutate()
                NewPopulation.append(Offspring)

            del Population[self.NumCloned - self.PopulationSize:]
            Population.extend(NewPopulation)

            # Some housekeeping
            Population = sorted(Population, key=self.fitness, reverse=True)
            GenerationIndex += 1

        # At end of main, plot results, and return number of generations required for
        # a solution (the latter is useful when running main interactively, as opposed
        # to as a script).
        self.plot_fitness_by_generation(MaxFitnesses, AvgFitnesses)
        return self.return_solution_if_found(GenerationIndex)
    
    # function for tournament selection 
    def tournament_selection(self, population, k):
        selected_parents = []
        for _ in range(k):
            tournament_members = random.sample(population, k)
            tournament_members.sort(key=self.fitness, reverse=True)
            selected_parents.append(tournament_members[0])  # Select the fittest individual from the tournament
        return selected_parents

    # Class representing individuals in the population
    class Individual_t:
        # Constructor to initialize Individual_t in the population 
        def __init__(self, BitString, NBits):  
            self.Genotype = BitString
            self.NBits = NBits  

        # function that gets the phenotype from the genotype
        def get_phenotype(self):
            return str(self.Genotype)

        # function that repsesents the phenotype as a string
        def __repr__(self):
            return f"{self.get_phenotype():>3s}"

        # Funtion which mutates a single individual 
        def mutate(self):
            BitFlipIndex = random.randint(0, self.NBits - 1)
            Mutant = []
            flip = lambda x: '1' if x == '0' else '0'

            for BitIndex in range(self.NBits):
                Bit = self.Genotype[BitIndex]
                Mutant.append(flip(Bit) if BitIndex == BitFlipIndex else Bit)

            self.Genotype = "".join(Mutant)

        # Function to perform crossover between two individuals
        def crossover(self, Individual2, NBits):  
            ChiasmaLocation = random.randint(0, NBits)  
            return GeneticAlgorithm.Individual_t(self.Genotype[:ChiasmaLocation] + Individual2.Genotype[ChiasmaLocation:], NBits)  # Pass NBits here

        # Static function to create a random individual
        @staticmethod
        def create_random(NBits):
            return GeneticAlgorithm.Individual_t(f"{bin(random.randint(0, 2 ** NBits - 1))[2:]:0>{NBits}s}", NBits)  # Pass NBits here

    # Function to calculate fitness of an individual
    def fitness(self, Individual):
        return self.fitness_function(self, Individual)

    # Function for performing a coin toss with a given bias
    def cointoss(self, Bias=0.5):
        return random.random() < Bias

    # Function to display the algorithm
    def display_progress(self, GenerationIndex, Population):
        if self.PRINTING:
            print(f"Population at Generation {GenerationIndex:02d}: {Population}")

    # Function to check if the problem has been solved
    def check_if_problem_has_been_solved(self, GenerationIndex, Population):
        global SolutionIndex
        if SolutionIndex is None and any(i.get_phenotype() == "0001111000001100000100001000101100010000000010001100010011100111111000011101001000010" for i in Population):
            SolutionIndex = GenerationIndex

    # Function to track fitness statistics
    def track_fitness(self, MaxFitnesses, AvgFitnesses, Population):
        Fitnesses = numpy.array([self.fitness(i) for i in Population])
        MaxFitnesses.append(Fitnesses.max())
        AvgFitnesses.append(Fitnesses.mean())

    # Function to initialize the population
    def initialise_population(self, Population):
        Population[:] = [self.Individual_t.create_random(self.NBits) for i in range(self.PopulationSize)]
        Population.sort(key=self.fitness, reverse=True)

    # Function to plot fitness statistics by generation
    def plot_fitness_by_generation(self, MaxFitnesses, AvgFitnesses):
        if self.PRINTING:
            plt.plot(range(self.MaxGeneration), MaxFitnesses, label="Max")
            plt.plot(range(self.MaxGeneration), AvgFitnesses, label="Avg")
            plt.legend()
            plt.title("Avg vs Max Fitness per Generation")
            plt.show(block=True)

    # Function to return solution index if found
    def return_solution_if_found(self, GenerationIndex):
        global SolutionIndex
        if SolutionIndex is not None:
            if self.PRINTING:
                print(f"\n'ascii image' Solution found in Generation {SolutionIndex}.")
            return SolutionIndex
        else:
            if self.PRINTING:
                print(f"'ascii image' Solution not found by generation {GenerationIndex}.")
            return -1
