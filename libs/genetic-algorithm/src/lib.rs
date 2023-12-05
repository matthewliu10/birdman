pub use self::{
    chromosome::*, crossover::*, individual::*, mutation::*, selection::*, statistics::*,
};

use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;

mod chromosome;
mod crossover;
mod individual;
mod mutation;
mod selection;
mod statistics;

pub struct GeneticAlgorithm<S, C, M> {
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
}

impl<S, C, M> GeneticAlgorithm<S, C, M>
where
    S: SelectionMethod,
    C: CrossoverMethod,
    M: MutationMethod,
{
    pub fn new(selection_method: S, crossover_method: C, mutation_method: M) -> Self {
        Self {
            selection_method,
            crossover_method,
            mutation_method,
        }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> (Vec<I>, Statistics)
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let stats = Statistics::new(population);

        let mut offspring = Vec::with_capacity(population.len());
        for _ in 0..population.len() {
            let parent1 = self
                .selection_method
                .select(rng, population)
                .to_chromosome();
            let parent2 = self
                .selection_method
                .select(rng, population)
                .to_chromosome();

            let mut child = self.crossover_method.crossover(rng, parent1, parent2);

            self.mutation_method.mutate(rng, &mut child);

            offspring.push(I::from_chromosome(child));
        }

        (offspring, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn individual(genes: Vec<f32>) -> TestIndividual {
        let chromosome: Chromosome = Chromosome::new(genes);

        TestIndividual::WithChromosome {
            chromosome: chromosome,
        }
    }

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 0.5),
        );

        let mut population = vec![
            individual(vec![0.0, 0.0, 0.0]), // fitness = 0.0
            individual(vec![1.0, 1.0, 1.0]), // fitness = 3.0
            individual(vec![1.0, 2.0, 1.0]), // fitness = 4.0
            individual(vec![1.0, 2.0, 4.0]), // fitness = 7.0
        ];

        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population).0;
        }

        let expected = vec![
            individual(vec![0.6002736, 1.5194247, 4.3595104]), // fitness ~= 6.5
            individual(vec![1.0955309, 2.4240465, 3.6959934]), // fitness ~= 7.2
            individual(vec![1.2753081, 2.4675508, 3.8890047]), // fitness ~= 7.6
            individual(vec![1.0225878, 2.4240465, 4.3595104]), // fitness ~= 7.8
        ];

        assert_eq!(population, expected)
    }
}
