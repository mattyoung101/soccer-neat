import simulation
import neat
import os
import pickle
import numpy as np
import sys

PROCESSES = 4
EVALUATIONS = 3

def eval_genomes(genomes, config):
    #print("Evaluating genome")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        scores = [simulation.simulate(net, config) for x in range(EVALUATIONS)]
        genome.fitness = np.mean(scores)
    #print("Done evaluating")

# Based on https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = [simulation.simulate(net, config) for x in range(EVALUATIONS)]
    return np.mean(scores)

if __name__ == "__main__":
    print("Initialising...")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-34')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    pe = neat.ParallelEvaluator(4, eval_genome)

    print("Training started!")
    try:
        winner = p.run(pe.evaluate, 500)
    except KeyboardInterrupt:
        print("Training aborted!")
        sys.exit(0)
        # TODO save best net here 

    print("Training completed. Dumping winner.")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("winner.net", "wb") as f:
        pickle.dump(winner_net, f)