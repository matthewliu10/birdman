use crate::*;

#[derive(Clone, Debug)]
pub struct Layer {
    pub(crate) neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    pub fn random(rng: &mut dyn rand::RngCore, input_size: usize, num_neurons: usize) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);

        for _ in 0..num_neurons {
            neurons.push(Neuron::random(rng, input_size));
        }

        Self { neurons }
    }

    pub fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let output = neuron.propogate(&inputs);
            outputs.push(output);
        }

        outputs
    }

    pub fn weights(&self) -> Vec<f32> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.weights())
            .collect()
    }

    pub fn from_weights(
        num_inputs: usize,
        num_neurons: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::from_weights(num_inputs, weights))
            .collect();

        Self { neurons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let input_size: usize = 3;
        let num_neurons: usize = 2;
        let layer = Layer::random(&mut rng, input_size, num_neurons);

        let mut actual_biases: Vec<_> = Vec::with_capacity(num_neurons);
        let mut actual_weights: Vec<_> = Vec::with_capacity(num_neurons);
        for i in 0..num_neurons {
            actual_biases.push(layer.neurons[i].bias);
            actual_weights.push(layer.neurons[i].weights.as_slice());
        }

        let expected_biases: Vec<f32> = vec![-0.6255188, 0.5238807];
        let expected_weights: Vec<&[f32]> = vec![
            &[0.67383957, 0.8181262, 0.26284897],
            &[-0.53516835, 0.069369674, -0.7648182],
        ];

        assert_relative_eq!(actual_biases.as_slice(), expected_biases.as_slice());
        assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
    }

    #[test]
    fn test_propogate() {
        let layer = Layer {
            neurons: vec![
                Neuron {
                    bias: 0.3,
                    weights: vec![0.3, 0.4, 0.1],
                },
                Neuron {
                    bias: 0.1,
                    weights: vec![0.4, 0.1, -0.1],
                },
            ],
        };

        let inputs = &[0.2, 0.4, 0.6];

        let actual = layer.propogate(inputs.to_vec());
        let expected = vec![
            f32::max((0.3 * 0.2) + (0.4 * 0.4) + (0.1 * 0.6) + 0.3, 0.0),
            f32::max((0.4 * 0.2) + (0.1 * 0.4) + (-0.1 * 0.6) + 0.1, 0.0),
        ];
        assert_relative_eq!(actual.as_slice(), expected.as_slice());

        let inputs = &[0.4, -2.0, -0.7];
        let actual = layer.propogate(inputs.to_vec());
        let expected = vec![
            f32::max((0.3 * 0.4) + (0.4 * -2.0) + (0.1 * -0.7) + 0.3, 0.0),
            f32::max((0.4 * 0.4) + (0.1 * -2.0) + (-0.1 * -0.7) + 0.1, 0.0),
        ];
        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }
}
