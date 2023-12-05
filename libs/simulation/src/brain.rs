use crate::*;

use lib_neural_network as nn;

#[derive(Debug)]
pub struct Brain {
    nn: nn::Network,
}

impl Brain {
    pub fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        Self {
            nn: nn::Network::random(rng, &Self::topology(input_size)),
        }
    }

    pub(crate) fn as_chromosome(&self) -> ga::Chromosome {
        let genes = self.nn.weights();
        ga::Chromosome::new(genes)
    }

    pub(crate) fn from_chromosome(input_size: usize, chromosome: ga::Chromosome) -> Self {
        let layer_info = Self::topology(input_size);
        let nn = nn::Network::from_weights(&layer_info, chromosome.genes);

        Self { nn }
    }

    pub(crate) fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.nn.propogate(inputs)
    }

    fn topology(input_size: usize) -> [usize; 3] {
        [input_size, 2 * input_size, 2]
    }
}
