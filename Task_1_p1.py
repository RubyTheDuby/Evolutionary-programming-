from Task_1 import GeneticAlgorithm  # Import the GeneticAlgorithm class from ascii_thing_3 module

# Define your parameters
PopulationSize = 200       # Size of the population
CrossoverRate = 0.70       # Size of the population
MutationRate = 0.3         # Probability of mutation
NBits = 17 * 5             # Number of bits in the genotype 
MaxGeneration = 50         # Maximum number of generations
PRINTING = True            # Whether to print progress during the algorithm execution

# Fitness function for the genetic algorithm
def fitness2(self, Individual):
    # Target image to be matched
    img = "0001111000001100000100001000101100010000000010001100010011100111111000011101001000010"
    # Calculate fitness by comparing each bit of the individual's phenotype with the target image
    ANS = sum(Individual.get_phenotype()[i] == img[i] for i in range(0, self.NBits))
    return ANS

# Create an instance of the GeneticAlgorithm class
genetic_algorithm = GeneticAlgorithm(PopulationSize, CrossoverRate, MutationRate, NBits, PRINTING, MaxGeneration, fitness2)

# Run the genetic algorithm
best_solution = genetic_algorithm.run()

# Output the best solution found
print("Best solution found:", best_solution)
