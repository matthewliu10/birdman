use crate::*;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub(crate) bias: f32,
    pub(crate) weights: Vec<f32>,
}

impl Neuron {
    pub fn new(bias: f32, weights: Vec<f32>) -> Self {
        Self { bias, weights }
    }

    pub fn random(rng: &mut dyn rand::RngCore, input_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let mut weights = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            weights.push(rng.gen_range(-1.0..=1.0));
        }

        Self { bias, weights }
    }

    pub fn propogate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(self.weights.len(), inputs.len());

        let mut output = 0.0;
        for i in 0..inputs.len() {
            output += inputs[i] * self.weights[i];
        }

        (output + self.bias).max(0.0)
    }

    pub fn weights(&self) -> Vec<f32> {
        let mut weights = vec![self.bias];
        weights.extend(&self.weights);
        weights
    }

    pub fn from_weights(num_weights: usize, weights: &mut dyn Iterator<Item = f32>) -> Self {
        let bias = weights.next().expect("not enough weights");
        let weights = (0..num_weights)
            .map(|_| weights.next().expect("not enough weights"))
            .collect();

        Self { bias, weights }
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
        let neuron = Neuron::random(&mut rng, 4);

        assert_relative_eq!(neuron.bias, -0.6255188);
        assert_relative_eq!(
            neuron.weights.as_slice(),
            [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
        );
    }

    #[test]
    fn test_propogate() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        // test ReLU
        assert_relative_eq!(neuron.propogate(&[-10.0, -10.0]), 0.0,);

        assert_relative_eq!(
            neuron.propogate(&[-0.4, 0.7]),
            (-0.4 * -0.3) + (0.7 * 0.8) + 0.5,
        );
    }
}
