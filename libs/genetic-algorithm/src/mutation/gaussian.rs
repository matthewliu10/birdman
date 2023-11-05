use crate::*;

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    chance: f32,
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff: f32) -> GaussianMutation {
        GaussianMutation { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            if rng.gen_bool(self.chance as _) {
                *gene += rng.gen_range(-self.coeff..=self.coeff);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn actual(chance: f32, coeff: f32) -> Vec<f32> {
        let mut child = Chromosome::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let mut rng = ChaCha8Rng::from_seed(Default::default());

        GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

        let mut genes = Vec::with_capacity(child.len());
        for i in 0..child.len() {
            genes.push(child[i]);
        }

        genes
    }

    fn no_change(chance: f32, coeff: f32) {
        let actual = actual(chance, coeff);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    fn change(chance: f32, coeff: f32, expected: Vec<f32>) {
        let actual = actual(chance, coeff);

        assert_relative_eq!(actual.as_slice(), expected.as_slice());
    }

    mod zero_chance {
        use super::*;

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn test() {
                no_change(0.0, 0.0);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn test() {
                no_change(0.0, 1.0);
            }
        }
    }

    mod half_chance {
        use super::*;

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn test() {
                no_change(0.5, 0.0);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn test() {
                change(
                    0.5,
                    1.0,
                    vec![1.0, 2.0, 3.0693698, 3.5112038, 5.2754607, 6.0, 6.638722],
                );
            }
        }
    }

    mod max_chance {
        use super::*;

        mod and_zero_coefficient {
            use super::*;

            #[test]
            fn test() {
                no_change(1.0, 0.0);
            }
        }

        mod and_nonzero_coefficient {
            use super::*;

            #[test]
            fn test() {
                change(
                    1.0,
                    1.0,
                    vec![
                        0.3744812, 2.6738396, 3.8181262, 4.262849, 5.523881, 5.464832, 7.06937,
                    ],
                );
            }
        }
    }
}
