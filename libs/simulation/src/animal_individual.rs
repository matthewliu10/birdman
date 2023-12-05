use crate::*;

#[derive(Debug)]
pub struct AnimalIndividual {
    chromosome: ga::Chromosome,
    fitness: f32,
}

impl AnimalIndividual {
    pub fn from_animal(animal: &Animal) -> Self {
        Self {
            fitness: animal.food_eaten as f32,
            chromosome: animal.as_chromosome(),
        }
    }

    pub fn to_animal(self, rng: &mut dyn RngCore) -> Animal {
        Animal::from_chromosome(self.chromosome, rng)
    }
}

impl ga::Individual for AnimalIndividual {
    fn from_chromosome(chromosome: ga::Chromosome) -> Self {
        Self {
            fitness: 0.0,
            chromosome,
        }
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }

    fn to_chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }
}