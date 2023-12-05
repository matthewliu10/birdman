use self::{layer::*, neuron::*};
use rand::Rng;

mod layer;
mod neuron;

#[derive(Clone, Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    // layer_info: Number of neurons in each layer of the network.
    // layer_info[0]: Size of the input given to the network.
    pub fn random(rng: &mut dyn rand::RngCore, layer_info: &[usize]) -> Self {
        assert!(layer_info.len() > 1);

        let mut built_layers = Vec::with_capacity(layer_info.len() - 1);

        for adjacent_layers in layer_info.windows(2) {
            built_layers.push(Layer::random(rng, adjacent_layers[0], adjacent_layers[1]));
        }

        Self {
            layers: built_layers,
        }
    }

    pub fn propogate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.propogate(inputs);
        }
        inputs
    }

    pub fn weights(&self) -> Vec<f32> {
        self.layers.iter().flat_map(|layer| layer.weights()).collect()
    }

    pub fn from_weights(layer_info: &[usize], weights: Vec<f32>) -> Self {
        let mut expected_num_weights = 0;
        for i in 1..(layer_info.len()) {
            expected_num_weights += layer_info[i];
            expected_num_weights += layer_info[i] * layer_info[i - 1];
        }
        assert!(layer_info.len() > 1);
        assert!(expected_num_weights == weights.len());

        let mut weights = weights.into_iter();
        let layers = layer_info.windows(2).map(|layers| Layer::from_weights(layers[0], layers[1], &mut weights)).collect();

        Self { layers }
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

        let layer_info = &[3, 2, 1];
        let network = Network::random(&mut rng, layer_info);

        assert_eq!(network.layers.len(), 2);

        assert_eq!(network.layers[0].neurons.len(), 2);
        assert_relative_eq!(network.layers[0].neurons[0].bias, -0.6255188);
        assert_relative_eq!(network.layers[0].neurons[1].bias, 0.5238807);
        assert_relative_eq!(
            network.layers[0].neurons[0].weights.as_slice(),
            &[0.67383957, 0.8181262, 0.26284897].as_slice(),
        );
        assert_relative_eq!(
            network.layers[0].neurons[1].weights.as_slice(),
            &[-0.53516835, 0.069369674, -0.7648182].as_slice(),
        );

        assert_eq!(network.layers[1].neurons.len(), 1);
        assert_relative_eq!(network.layers[1].neurons[0].bias, -0.102499366);
        assert_relative_eq!(
            network.layers[1].neurons[0].weights.as_slice(),
            &[-0.48879617, -0.19277132].as_slice(),
        );
    }

    #[test]
    fn test_propogate() {
        let network = Network {
            layers: vec![
                Layer {
                    neurons: vec![
                        Neuron {
                            bias: 0.4,
                            weights: vec![0.3, -0.8, 0.0],
                        },
                        Neuron {
                            bias: -0.8,
                            weights: vec![0.6, -0.3, -0.9],
                        },
                    ],
                },
                Layer {
                    neurons: vec![Neuron {
                        bias: 0.0,
                        weights: vec![0.9, -2.6],
                    }],
                },
            ],
        };

        let input = vec![0.9, 0.6, -0.2];
        let actual = network.propogate(input.clone());
        let expected = network.layers[1].propogate(network.layers[0].propogate(input.clone()));

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_weights() {
        let network = Network::new(vec![
            Layer::new(vec![Neuron::new(0.1, vec![0.2, 0.3, 0.4])]),
            Layer::new(vec![Neuron::new(0.5, vec![0.6, 0.7, 0.8])]),
        ]);

        let actual_weights = network.weights();
        let expected_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
    }

    #[test]
    fn from_weights() {
        let layer_info = &[3, 2];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let network = Network::from_weights(layer_info, weights.clone());
        let actual = network.weights();

        assert_relative_eq!(actual.as_slice(), weights.as_slice());
    }
}
