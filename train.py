import simulation
import neat
import os
import pickle
import numpy as np

def eval_genomes(genomes, config):
    print("Evaluating genome")
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        scores = [simulation.simulate(net, config) for x in range(3)]
        genome.fitness = np.mean(scores)
    print("Done evaluating")

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

    print("Training started!")
    try:
        winner = p.run(eval_genomes, 30)
    except KeyboardInterrupt:
        print("Training aborted!")
        # TODO save best net here 

    print("Training completed. Dumping winner.")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("winner.net", "wb") as f:
        pickle.dump(winner_net, f)