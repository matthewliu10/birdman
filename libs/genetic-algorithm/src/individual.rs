use crate::*;

pub trait Individual {
    fn from_chromosome(chromosome: Chromosome) -> Self;
    fn fitness(&self) -> f32;
    fn to_chromosome(&self) -> &Chromosome;
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub enum TestIndividual {
    WithChromosome { chromosome: Chromosome },
    WithFitness { fitness: f32 },
}

#[cfg(test)]
impl TestIndividual {
    pub fn new(fitness: f32) -> Self {
        Self::WithFitness { fitness }
    }
}

#[cfg(test)]
impl Individual for TestIndividual {
    fn from_chromosome(chromosome: Chromosome) -> Self {
        Self::WithChromosome { chromosome }
    }

    fn fitness(&self) -> f32 {
        match self {
            Self::WithChromosome { chromosome } => chromosome.iter().sum(),
            Self::WithFitness { fitness } => *fitness,
        }
    }

    fn to_chromosome(&self) -> &Chromosome {
        match self {
            Self::WithChromosome { chromosome } => chromosome,
            Self::WithFitness { fitness: _ } => {
                panic!("not supported for TestIndividual::WithFitness")
            }
        }
    }
}
