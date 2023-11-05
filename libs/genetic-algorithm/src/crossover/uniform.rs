use crate::*;

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert!(parent_a.len() != 0);
        assert_eq!(parent_a.len(), parent_b.len());

        let gene_cnt = parent_a.len();
        let mut child: Vec<f32> = Vec::with_capacity(gene_cnt);

        for i in 0..gene_cnt {
            child.push(if rng.gen_bool(0.5) {
                parent_a[i]
            } else {
                parent_b[i]
            })
        }

        Chromosome::new(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let parent_a: Chromosome = (1..=1000).map(|n| n as f32).collect();
        let parent_b: Chromosome = (1..=1000).map(|n| -n as f32).collect();

        let uniform_crossover = UniformCrossover::new();
        let child: Chromosome = uniform_crossover.crossover(&mut rng, &parent_a, &parent_b);

        let mut diff_a = 0;
        let mut diff_b = 0;
        for i in 0..1000 {
            if child[i] > 0.0 {
                diff_b += 1;
            } else {
                diff_a += 1;
            }
        }

        assert_eq!(diff_a, 515);
        assert_eq!(diff_b, 485);
    }
}
