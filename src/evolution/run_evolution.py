import random
import string

ALPHABET = string.ascii_letters + " !"

def random_char(bias_lowercase):
    if random.random() < bias_lowercase / 100:
        return random.choice(string.ascii_lowercase + " ")
    return random.choice(ALPHABET)

def random_genome(length, bias_lowercase): # random string
    return "".join(random_char(bias_lowercase) for _ in range(length))

def fitness(genome, target): # letters matching target
    return sum(a == b for a, b in zip(genome, target))

def mutate(genome, mutation_rate, bias_lowercase): # mutate random letters
    return "".join(random_char(bias_lowercase) if random.random() < mutation_rate else c for c in genome)

def crossover(parent_a, parent_b): # splits down the middle and combines
    point = random.randrange(1, len(parent_a))
    return parent_a[:point] + parent_b[point:]

def run_evolution(target, mu, lambda_, generations, mutation_rate, seed, plus=True, bias_lowercase=0):
    if not plus and lambda_ < mu:
        raise ValueError("comma selection needs lambda_ >= mu")
    if not 0 <= bias_lowercase <= 100:
        raise ValueError("bias_lowercase must be a percent between 0 and 100")

    random.seed(seed)
    parents = [random_genome(len(target), bias_lowercase) for _ in range(mu)]

    for generation in range(generations):
        children = [
            mutate(crossover(random.choice(parents), random.choice(parents)), mutation_rate, bias_lowercase)
            for _ in range(lambda_)
        ]
        pool = parents + children if plus else children
        scored = sorted(((g, fitness(g, target)) for g in pool), key=lambda pair: pair[1], reverse=True)[:mu]
        parents = [g for g, _ in scored]

        best, best_score = scored[0]
        print(f"{generation}: {best}")

        if best_score == len(target):
            return best, generation

    return parents[0], generations

if __name__ == "__main__":
     # this is just a text evolver that evolves a random string to `target`
    run_evolution(
        target="this is just a very simple text evolver that turns random strings into this target string",
        mu=20,
        lambda_=100,
        generations=1000,
        mutation_rate=0.02,
        seed=67,
        plus=True, # True for mu + lambda, False for mu, lambda

        # converges faster when the target is very heavily lowercase
        # otherwise, this hurts convergence speed a lot
        bias_lowercase=100, 
    )