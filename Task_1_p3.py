import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, PopulationSize, CrossoverRate, MutationRate, NDimensions, MaxGeneration, fitnessfunction):
        # Initialize the genetic algorithm with parameters 
        self.PopulationSize = PopulationSize # Size of the population
        self.CrossoverRate = CrossoverRate   # probability of crossoverrate
        self.MutationRate = MutationRate     # Probability of mutation
        self.NDimensions = NDimensions
        self.MaxGeneration = MaxGeneration  # Maximum number of generations
        self.fitnessfunction = fitnessfunction

    def run(self):
        # Run the genetic algorithm
        Population = []
        BestFitnesses = []
        GenerationIndex = 0

        self.initialise_population(Population)

        while GenerationIndex < self.MaxGeneration:
            # Perform genetic operations
            NewPopulation = []

            for _ in range(self.PopulationSize):
                selected_parents = self.tournament_selection(Population, 2)
                Parent1, Parent2 = selected_parents[0], selected_parents[1]
                Offspring = Parent1.crossover(Parent2)
                if self.cointoss(self.MutationRate):
                    Offspring.mutate()
                NewPopulation.append(Offspring)

            Population = NewPopulation
            Population.sort(key=self.fitness, reverse=False)
            BestFitnesses.append(self.fitness(Population[0]))

            GenerationIndex += 1

        self.plot_fitness_by_generation(BestFitnesses)
        return Population[0].get_phenotype()

    def tournament_selection(self, population, k):
        # Perform tournament selection to select parents for crossover
        selected_parents = []
        for _ in range(k):
            tournament_members = random.sample(population, k)
            tournament_members.sort(key=self.fitness, reverse=False)
            selected_parents.append(tournament_members[0])
        return selected_parents

    class Individual:
        def __init__(self, Genotype):
            # Initialize an individual with genotype
            self.Genotype = Genotype

        def get_phenotype(self):
            # Return the phenotype 
            return self.Genotype

        def mutate(self):
            # Mutate the individual genotype
            mutated = [gene + random.uniform(-0.1, 0.1) for gene in self.Genotype]
            self.Genotype = mutated

        def crossover(self, individual2):
            # Perform crossover with another individual
            crossover_point = random.randint(0, len(self.Genotype) - 1)
            new_genotype = self.Genotype[:crossover_point] + individual2.Genotype[crossover_point:]
            return GeneticAlgorithm.Individual(new_genotype)

    def fitness(self, individual):
        # Calculate fitness of an individual
        return self.fitnessfunction(individual.get_phenotype())

    def cointoss(self, Bias=0.5):
        # Perform a coin toss for mutation
        return random.random() < Bias

    def initialise_population(self, Population):
        # Initialize the population with random individuals
        Population[:] = [self.Individual(self.random_genotype()) for _ in range(self.PopulationSize)]

    def random_genotype(self):
        # Generate a random genotype within a specified range for each dimension
        return [random.uniform(-5, 5) for _ in range(self.NDimensions)]

    def plot_fitness_by_generation(self, BestFitnesses):
        # Plot the best fitness values over generations
        plt.plot(range(self.MaxGeneration), BestFitnesses, label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best Fitness per Generation")
        plt.legend()
        plt.show()

def sphere_function(x):
    # Sphere function for optimization
    return sum(xi ** 2 for xi in x)

PopulationSize = 500 # Size of the population
CrossoverRate = 0.7 # Probability of crossover
MutationRate = 0.03 # Probability of mutation
NDimensions = 2
MaxGeneration = 100 # Maximum number of generations

# Create an instance of the GeneticAlgorithm class for Sphere function
genetic_algorithm_sphere = GeneticAlgorithm(PopulationSize, CrossoverRate, MutationRate, NDimensions, MaxGeneration, sphere_function)
# Run the genetic algorithm for Sphere function
best_solution_sphere = genetic_algorithm_sphere.run()
print("Best solution found for Sphere function:", best_solution_sphere)
