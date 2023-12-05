use crate::*;

#[derive(Clone, Debug)]
pub struct Statistics {
    min_fitness: f32,
    max_fitness: f32,
    avg_fitness: f32,
}

impl Statistics {
    pub fn new<I>(population: &[I]) -> Self
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let population_size = population.len();
        let mut min_fitness = population[0].fitness();
        let mut max_fitness = population[0].fitness();
        let mut sum_fitness = 0.0;

        for i in 0..population_size {
            let fitness = population[i].fitness();

            min_fitness = min_fitness.min(fitness);
            max_fitness = max_fitness.max(fitness);
            sum_fitness += fitness;
        }

        let avg_fitness = sum_fitness / population_size as f32;

        Self {
            min_fitness,
            max_fitness,
            avg_fitness,
        }
    }

    pub fn min_fitness(&self) -> f32 {
        self.min_fitness
    }

    pub fn max_fitness(&self) -> f32 {
        self.max_fitness
    }

    pub fn avg_fitness(&self) -> f32 {
        self.avg_fitness
    }
}
